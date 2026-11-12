"""rjob site adapter: obtain and release a GPU pod through the brainpp CLI.

The core speaks one provisioning contract (``harborrl.deployment.provision``):
``up`` prints one JSON record naming the machine it produced, ``down``
receives that record on stdin and releases it. This module implements that
contract for the H cluster, whose GPU jobs are submitted with ``rjob``.

Two facts about ``rjob`` shape the implementation. It has no ``exec`` action,
so the pod's command must be a script written *before* submission — the
script bootstraps sshd, publishes the pod's address, then sleeps so the pod
does not exit underneath us. And a submitted pod is billed until it is
deleted, so ``up`` refuses to submit while a previous handle is still
recorded: a double submit quietly burns a second allocation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULTS = {
    "namespace": "ailab-datafoundation",
    "charged_group": "datafoundation_gpu_pool",
    "private_machine": "group",
    "image_pull_policy": "IfNotPresent",
    "mounts": [],
    "gpu": 4,
    "cpu": 64,
    "memory": 400000,
    "priority": 1,
    "ssh_user": "root",
    "ssh_port": 22,
    "timeout_s": 1800,
    "poll_interval_s": 15,
    "alive_seconds": 21600,
}
REQUIRED = ("name_prefix", "image", "workdir", "identity")
MOUNT_TARGET = re.compile(r"^[A-Za-z0-9_./:@-]+$")

# The submission tool and the clock are indirections so the tests can drive
# the whole lifecycle without a cluster or a real wait.
RUNNER = subprocess.run
CLOCK = time


class RjobError(RuntimeError):
    """A submission, discovery or release step did not behave as declared."""


def load_config(path):
    """Read and validate one site config; never accept a half-known file."""
    try:
        raw = json.loads(Path(path).read_text())
    except OSError as exc:
        raise RjobError(f"cannot read rjob config {path}: {exc}") from exc
    except ValueError as exc:
        raise RjobError(f"rjob config {path} is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RjobError("rjob config must be a JSON object")
    unknown = sorted(set(raw) - set(DEFAULTS) - set(REQUIRED))
    if unknown:
        raise RjobError(f"rjob config names unsupported fields {unknown}")
    config = {**DEFAULTS, **raw}
    missing = [key for key in REQUIRED if not config.get(key)]
    if missing:
        raise RjobError(f"rjob config requires {missing}")
    for key in ("name_prefix", "image", "identity"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise RjobError(f"rjob config {key} must be a nonempty string")
    workdir = Path(config["workdir"])
    if not workdir.is_absolute():
        # The pod mounts this directory and reads the bootstrap from it, so a
        # relative path would resolve differently on the two sides.
        raise RjobError("rjob config workdir must be an absolute path")
    config["workdir"] = str(workdir)
    config["identity"] = str(Path(config["identity"]))
    mounts = config.get("mounts") or []
    if not isinstance(mounts, list) or not all(
            isinstance(mount, str) and MOUNT_TARGET.match(mount) for mount in mounts):
        raise RjobError("rjob config mounts is a list of local:target paths")
    config["mounts"] = list(mounts)
    for key in ("gpu", "cpu", "memory", "priority", "ssh_port", "timeout_s"):
        if type(config[key]) is not int or config[key] <= 0:
            raise RjobError(f"rjob config {key} must be a positive integer")
    if type(config["poll_interval_s"]) not in (int, float) or config["poll_interval_s"] <= 0:
        raise RjobError("rjob config poll_interval_s must be a positive number")
    if not isinstance(config["alive_seconds"], int) or config["alive_seconds"] <= 0:
        raise RjobError("rjob config alive_seconds must be a positive integer")
    return config


def _run(argv, *, timeout=None, stdin=None):
    try:
        return RUNNER(argv, capture_output=True, text=True,
                      input=stdin, timeout=timeout, check=False,
                      env=dict(os.environ))
    except OSError as exc:
        raise RjobError(f"cannot run {argv[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RjobError(f"{' '.join(argv[:2])} exceeded {timeout}s") from exc


def _public_key(identity):
    """Derive the pod-side authorized key from the manager's private key.

    Only the public half ever enters the generated script; the private key
    stays on the manager, where the core's ssh calls read it by path.
    """
    result = _run(["ssh-keygen", "-y", "-f", identity], timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        raise RjobError(
            f"cannot derive a public key from {identity}: {result.stderr.strip()[-200:]}")
    return result.stdout.strip()


def bootstrap_script(config, public_key):
    """The pod command: bring sshd up, publish the address, then stay alive.

    ``rjob`` cannot execute anything inside a running pod, so this script is
    the only channel from the site into the pod. It is idempotent — re-running
    it appends nothing twice and starts no second sshd — because a pod may be
    re-entered by hand while the acceptance run is being diagnosed.
    """
    workdir = shlex.quote(config["workdir"])
    pubkey = shlex.quote(public_key)
    return f"""#!/usr/bin/env bash
set -uo pipefail
mkdir -p /root/.ssh && chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
grep -qxF {pubkey} /root/.ssh/authorized_keys 2>/dev/null || echo {pubkey} >> /root/.ssh/authorized_keys
if ! pgrep -x sshd >/dev/null 2>&1; then
  ssh-keygen -A >/dev/null 2>&1
  /usr/sbin/sshd
fi
printf '{{"hostname":"%s","ips":"%s"}}\\n' "$(hostname)" "$(hostname -I)" > {workdir}/pod-net.json
sleep {config['alive_seconds']}
"""


def submit_argv(config, script_path):
    argv = ["rjob", "submit", f"--name={config['name_prefix']}",
            "--enable-sshd",
            f"--gpu={config['gpu']}", f"--cpu={config['cpu']}",
            f"--memory={config['memory']}", "-P", str(config["priority"]),
            f"--charged-group={config['charged_group']}",
            f"--private-machine={config['private_machine']}",
            f"--namespace={config['namespace']}",
            f"--image={config['image']}",
            f"--image-pull-policy={config['image_pull_policy']}"]
    for mount in config["mounts"]:
        argv.append(f"--mount={mount}")
    argv += ["--", "bash", str(script_path)]
    return argv


CREATED = re.compile(r"created rjob_name:\s*(\S+)")
# A replica line reads "|- replica <name>: STATE, <node>"; the trailing node
# name means the state is not at end-of-line, and a summary line may instead
# end in a human state word such as "Inqueue" — only the uppercase token is
# the machine state these commands are asked about.
STATE = re.compile(r":\s*([A-Z][A-Z_]*)\s*(?:,|$)")


def _job_name(stdout):
    matches = CREATED.findall(stdout or "")
    if not matches:
        raise RjobError("rjob submit did not report the created job name")
    return matches[-1]


def job_state(name, config):
    """The replica state of one job, or None when the job is gone.

    Three outcomes, and they must not be confused: a live job answers with a
    replica line; a deleted job answers successfully with no replica line at
    all; a broken probe exits nonzero. Treating the last as "gone" would
    release the record for a pod that is still billed, so it raises instead.
    """
    result = _run(["rjob", "get", name, f"--namespace={config['namespace']}"], timeout=120)
    if result.returncode != 0:
        raise RjobError(
            f"rjob get {name} failed with {result.returncode}: "
            f"{result.stderr.strip()[-300:]}")
    states = [match for match in STATE.findall(result.stdout or "") if match]
    # No replica line means the job no longer exists; that is a legitimate
    # answer, not an unknown state to wait on forever.
    return states[-1] if states else None


def pod_address(config):
    """The address record the bootstrap wrote into the shared workdir."""
    path = Path(config["workdir"]) / "pod-net.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    ips = str(record.get("ips") or "").split()
    # The pod network address is the first global-looking IPv4; loopback and
    # the docker bridge are never reachable from the manager.
    for candidate in ips:
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", candidate) and not candidate.startswith(
                ("127.", "172.17.")):
            return candidate
    return None


def _ssh_ok(config, host, *, timeout=30):
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ClearAllForwardings=yes", "-o", "ConnectTimeout=10",
            "-i", config["identity"], "-p", str(config["ssh_port"]),
            f"{config['ssh_user']}@{host}", "echo ok"]
    result = _run(argv, timeout=timeout)
    return result.returncode == 0 and result.stdout.strip() == "ok"


def _wait(predicate, *, timeout, interval, description):
    deadline = CLOCK.monotonic() + timeout
    last = None
    while CLOCK.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        CLOCK.sleep(interval)
    raise RjobError(f"timed out after {timeout}s waiting for {description}")


def up(config, *, record_path=None):
    """Submit one pod and return the provision record naming it."""
    handle_path = Path(config["workdir"]) / "rjob-handle.json"
    net_path = Path(config["workdir"]) / "pod-net.json"
    if net_path.exists() or handle_path.exists():
        # A leftover record means an earlier submission may still be billed;
        # refuse instead of silently stacking a second pod on the quota.
        raise RjobError(
            f"a previous pod record exists under {config['workdir']}; "
            "release it with down or remove the stale files deliberately")
    workdir = Path(config["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "bootstrap.sh"
    script.write_text(bootstrap_script(config, _public_key(config["identity"])))
    script.chmod(0o755)

    result = _run(submit_argv(config, script), timeout=600)
    if result.returncode != 0:
        raise RjobError(
            f"rjob submit failed with {result.returncode}: {result.stderr.strip()[-400:]}")
    name = _job_name(result.stdout)
    print(f"submitted {name}", file=sys.stderr)
    handle_path.write_text(json.dumps({"handle": name}) + "\n")

    _wait(lambda: job_state(name, config) in ("ACTIVE", "RUNNING"),
          timeout=config["timeout_s"], interval=config["poll_interval_s"],
          description=f"job {name} to become active")
    host = _wait(lambda: pod_address(config), timeout=config["timeout_s"],
                 interval=config["poll_interval_s"],
                 description=f"pod {name} to publish its address")
    _wait(lambda: _ssh_ok(config, host), timeout=config["timeout_s"],
          interval=config["poll_interval_s"],
          description=f"non-interactive ssh to {host}")

    record = {"host": host, "user": config["ssh_user"],
              "port": config["ssh_port"], "identity": config["identity"],
              "handle": name}
    if record_path is not None:
        Path(record_path).write_text(json.dumps(record) + "\n")
    return record


def down(config, record):
    """Delete the recorded pod; exit nonzero if it cannot be released."""
    handle = (record or {}).get("handle")
    if not isinstance(handle, str) or not handle.strip():
        raise RjobError("release record carries no handle; refusing to guess a job name")
    result = _run(["rjob", "delete", handle, f"--namespace={config['namespace']}"],
                  timeout=600)
    if result.returncode != 0:
        raise RjobError(
            f"rjob delete {handle} failed with {result.returncode}: "
            f"{result.stderr.strip()[-400:]}")
    _wait(lambda: job_state(handle, config) is None, timeout=config["timeout_s"],
          interval=config["poll_interval_s"], description=f"job {handle} to disappear")
    for name in ("rjob-handle.json", "pod-net.json"):
        with __import__("contextlib").suppress(OSError):
            (Path(config["workdir"]) / name).unlink()
    return handle


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harborrl_h_cluster.rjob",
                                     description="GPU pod provisioning through rjob")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("up", "down"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        if args.action == "up":
            record = up(config)
            print(json.dumps(record))
            return 0
        payload = sys.stdin.read().strip()
        if not payload:
            raise RjobError("down reads the provision record as JSON on stdin")
        down(config, json.loads(payload))
        return 0
    except (RjobError, ValueError) as exc:
        print(f"rjob practice: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
