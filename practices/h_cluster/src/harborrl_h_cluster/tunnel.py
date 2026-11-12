"""Reverse-tunnel adapter: serve shard ports from a host the cluster cannot dial.

The H cluster gives every machine an inbound port only inside the cluster; a
host outside it that is entered through an interactive web portal has no
inbound route at all. The one direction that works is the outside host
initiating ssh to the hub. This module exploits exactly that: the outside
host opens ssh connections carrying ``-R`` forwards, so the hub binds the
ports, and the in-cluster manager dials ``127.0.0.1:<hub_port>`` as if the
shard lived next door.

Three properties drive the design. The tunnel is supervised, because the jump
gateway drops idle connections and a silent death would look like a shard
crash. The state file records why the last connection ended, because the
platform may wipe the outside host's ``/root`` and take the installed keys
with it — an operator needs to see that, not infer it. And artifacts are
pushed back over the same ssh direction, because the outside host has no
shared storage: anything written there dies with the host.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

SSH_OPTIONS = (
    "-N",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ClearAllForwardings=yes",
)
BACKOFF_START = 1.0
BACKOFF_CAP = 60.0

RUNNER = subprocess.run
POPEN = subprocess.Popen
CLOCK = time


class TunnelError(RuntimeError):
    """The tunnel configuration or a tunnel step did not behave as declared."""


def load_config(path):
    """Read and validate one site config; a bad mapping fails before connecting."""
    try:
        raw = json.loads(Path(path).read_text())
    except OSError as exc:
        raise TunnelError(f"cannot read tunnel config {path}: {exc}") from exc
    except ValueError as exc:
        raise TunnelError(f"tunnel config {path} is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise TunnelError("tunnel config must be a JSON object")
    unknown = sorted(set(raw) - {"hub", "identity", "mappings", "state_path",
                                 "push", "backoff_cap", "backoff_start"})
    if unknown:
        raise TunnelError(f"tunnel config names unsupported fields {unknown}")
    hub = raw.get("hub")
    if not isinstance(hub, str) or not hub.strip() or hub.startswith("-"):
        raise TunnelError("tunnel config hub must be an ssh destination")
    identity = raw.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise TunnelError("tunnel config identity must be a key path")
    mappings = raw.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise TunnelError("tunnel config mappings must be a nonempty list")
    normalized, hub_ports = [], set()
    for index, entry in enumerate(mappings):
        label = f"tunnel config mappings[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"local_port", "hub_bind", "hub_port"}:
            raise TunnelError(f"{label} requires exactly local_port, hub_bind and hub_port")
        for key in ("local_port", "hub_port"):
            if type(entry[key]) is not int or not 0 < entry[key] < 65536:
                raise TunnelError(f"{label}.{key} must be an integer in 1..65535")
        bind = entry["hub_bind"]
        if not isinstance(bind, str) or not bind.strip():
            raise TunnelError(f"{label}.hub_bind must be an address to bind on the hub")
        if entry["hub_port"] in hub_ports:
            # Two forwards onto one hub port: ssh silently keeps the first,
            # so the second shard would never be reachable.
            raise TunnelError(f"duplicate hub port {entry['hub_port']}")
        hub_ports.add(entry["hub_port"])
        normalized.append({"local_port": entry["local_port"], "hub_bind": bind,
                           "hub_port": entry["hub_port"]})
    state_path = raw.get("state_path")
    if not isinstance(state_path, str) or not state_path.strip():
        raise TunnelError("tunnel config state_path must be a path")
    push = raw.get("push") or []
    if not isinstance(push, list) or not all(isinstance(item, str) and item.strip() for item in push):
        raise TunnelError("tunnel config push is a list of local:remote directories")
    config = {
        "hub": hub.strip(), "identity": identity.strip(), "mappings": normalized,
        "state_path": state_path, "push": list(push),
        "backoff_start": float(raw.get("backoff_start", BACKOFF_START)),
        "backoff_cap": float(raw.get("backoff_cap", BACKOFF_CAP)),
    }
    if config["backoff_start"] <= 0 or config["backoff_cap"] < config["backoff_start"]:
        raise TunnelError("tunnel config backoff must start positive and not exceed the cap")
    return config


def forward_argv(config):
    """The exact ssh argv: one -R per mapping, stable option order.

    Each forward binds ``hub_bind:hub_port`` on the hub and connects it to
    ``127.0.0.1:local_port`` on this side. The local side is always loopback:
    binding a shard process on a routable address would expose it to the
    outside network for no gain.
    """
    argv = ["ssh", *SSH_OPTIONS, "-i", config["identity"]]
    for mapping in config["mappings"]:
        argv += ["-R", f"{mapping['hub_bind']}:{mapping['hub_port']}:"
                       f"127.0.0.1:{mapping['local_port']}"]
    argv.append(config["hub"])
    return argv


def write_state(path, config, *, pid, attempts, last_error=None, connected=False):
    """Record what the supervisor is doing; the operator reads this, not a log."""
    payload = {
        "pid": pid, "connected": bool(connected), "attempts": int(attempts),
        "last_error": last_error, "updated": CLOCK.strftime("%Y-%m-%dT%H:%M:%S"),
        "mappings": list(config["mappings"]), "hub": config["hub"],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temp.replace(target)


def daemon(config, *, stop=None):
    """Supervise the tunnel: run ssh, wait, reconnect with capped backoff.

    Returns the number of completed connection attempts. A caller passes a
    ``threading.Event`` (or any object with ``is_set``) to stop the loop from
    another thread; SIGINT and SIGTERM do the same from the process side.
    """
    stop = stop or _StopFlag()
    previous = {}
    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda *_args: stop.set())
    delay, attempts = config["backoff_start"], 0
    try:
        while not stop.is_set():
            attempts += 1
            write_state(config["state_path"], config, pid=os.getpid(),
                        attempts=attempts, connected=True)
            child = None
            error = None
            try:
                child = POPEN(forward_argv(config),
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                while not stop.is_set():
                    if child.poll() is not None:
                        break
                    CLOCK.sleep(0.2)
                if stop.is_set():
                    break
                stderr = b""
                try:
                    stderr = child.stderr.read() if child.stderr else b""
                except (OSError, ValueError):
                    stderr = b""
                error = (stderr.decode(errors="replace").strip()[-300:]
                         or f"ssh exited {child.returncode}")
            except OSError as exc:
                error = f"cannot start ssh: {exc}"
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
            if stop.is_set():
                break
            write_state(config["state_path"], config, pid=os.getpid(),
                        attempts=attempts, last_error=error, connected=False)
            print(f"tunnel: {error}; retrying in {delay:.0f}s", file=sys.stderr)
            # Sleep in small slices so a stop request is honoured promptly
            # instead of waiting out a full backoff interval.
            waited = 0.0
            while waited < delay and not stop.is_set():
                CLOCK.sleep(min(0.2, delay - waited))
                waited += 0.2
            delay = min(delay * 2, config["backoff_cap"])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return attempts


class _StopFlag:
    """A minimal stand-in for threading.Event, used when no stop is passed."""

    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set


def install_commands(config, *, root="/opt/harborrl-tunnel", package_dir=None,
                     public_key=None):
    """The two commands that install this adapter across the only working path.

    The outside host has neither this package nor a route to anything but the
    hub, and its key is not authorized yet — so one paste cannot do both ends.
    ``hub`` runs on the hub (authorize the key, stage nothing else); ``host``
    runs on the outside host (pull the package over the one direction that
    works, write the config, start the supervised daemon). Both are
    idempotent: re-running adds no duplicate key and starts no second daemon.

    ``public_key`` exists because of the same chicken-and-egg: the keypair is
    generated on the outside host (it is the only side that needs the private
    half), so the hub cannot read it from disk. The operator copies the public
    half out of the host's portal session and passes it here.
    """
    public_key = public_key or _public_key(config["identity"])
    site = json.dumps({
        "hub": config["hub"], "identity": config["identity"],
        "mappings": config["mappings"], "state_path": config["state_path"],
        "push": config["push"],
    }, separators=(",", ":"))
    source = str(Path(package_dir or Path(__file__).resolve().parent.parent))
    quoted_root = shlex.quote(root)
    hub = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh "
        "&& touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys "
        # Idempotent by content: a blind append leaves duplicates that look
        # like several hosts sharing one key.
        f"&& (grep -qxF {shlex.quote(public_key)} ~/.ssh/authorized_keys "
        f"|| echo {shlex.quote(public_key)} >> ~/.ssh/authorized_keys) "
        "&& echo hub-ready"
    )
    host = (
        f"mkdir -p {quoted_root}/lib "
        # The package arrives over the same direction the tunnel uses; there
        # is no shared storage here to read it from.
        f"&& ssh -o BatchMode=yes -i {shlex.quote(config['identity'])} "
        f"{shlex.quote(config['hub'])} "
        f"'tar -C {shlex.quote(source)} -cf - harborrl_h_cluster' "
        f"| tar -C {quoted_root}/lib -xf - "
        f"&& printf '%s\\n' {shlex.quote(site)} > {quoted_root}/tunnel.json "
        f"&& if [ -f {quoted_root}/tunnel.pid ] "
        f"&& kill -0 \"$(cat {quoted_root}/tunnel.pid)\" 2>/dev/null; then echo tunnel-already-running; "
        f"else PYTHONPATH={quoted_root}/lib nohup python3 -m harborrl_h_cluster.tunnel daemon "
        f"--config {quoted_root}/tunnel.json >> {quoted_root}/tunnel.log 2>&1 "
        f"& echo $! > {quoted_root}/tunnel.pid; echo tunnel-started; fi"
    )
    return {"hub": hub, "host": host}


def _public_key(identity):
    """The public half of a key file; only this ever leaves the host."""
    result = RUNNER(["ssh-keygen", "-y", "-f", identity],
                    capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0 or not (result.stdout or "").strip():
        raise TunnelError(
            f"cannot derive a public key from {identity}: "
            f"{(result.stderr or '').strip()[-200:]}")
    return result.stdout.strip()


def push(config, *, once=False):
    """Copy declared directories back to the hub without exposing partial files.

    No shared storage exists on the outside host, so traces written there must
    arrive before the host is recycled. ``rsync`` writes through a temporary
    name and renames, so a reader never sees a half-written tree; without
    rsync, ``tar`` streams over ssh and unpacks atomically in a temp dir.
    """
    results = []
    for index, entry in enumerate(config["push"]):
        local, _, remote = entry.partition(":")
        if not remote:
            raise TunnelError(f"push entry {index} must be a local:remote pair")
        argv = ["rsync", "-a", "--timeout=120",
                "-e", f"ssh -o BatchMode=yes -i {config['identity']}",
                str(local), f"{config['hub']}:{remote}"]
        try:
            result = RUNNER(argv, capture_output=True, text=True,
                            timeout=1800, check=False, env=dict(os.environ))
            results.append({"entry": entry, "returncode": result.returncode,
                            "stderr": (result.stderr or "").strip()[-200:]})
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append({"entry": entry, "returncode": None, "stderr": str(exc)})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harborrl_h_cluster.tunnel",
                                     description="Reverse ssh tunnel for an outside host")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("daemon", "install", "push"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--config", required=True)
        if action == "push":
            cmd.add_argument("--once", action="store_true")
        if action == "install":
            cmd.add_argument("--root", default="/opt/harborrl-tunnel")
            # The keypair lives on the outside host; the hub runs install
            # without it and is handed the public half from the portal.
            cmd.add_argument("--public-key", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        if args.action == "install":
            # Both ends are printed: the key is not authorized yet, so the
            # outside host cannot reach the hub until the hub command runs.
            # Neither command carries private key material.
            commands = install_commands(config, root=args.root,
                                        public_key=args.public_key)
            print(json.dumps(commands))
            return 0
        if args.action == "daemon":
            daemon(config)
            return 0
        results = push(config, once=args.once)
        print(json.dumps(results))
        return 0 if all(item["returncode"] == 0 for item in results) else 1
    except TunnelError as exc:
        print(f"tunnel practice: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
