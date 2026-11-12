"""Reverse tunnel adapter: mapping, supervision and pasteable installation.

Nothing here opens a socket: ssh runs through a fake subprocess layer, so
these tests pin the decisions — the exact -R strings, what a reconnect does
after a failure, and what the operator's one-line install contains.
"""
from __future__ import annotations

import json
import threading

import pytest

from harborrl_h_cluster import tunnel


def write_config(tmp_path, **overrides):
    values = {
        "hub": "manager.example",
        "identity": str(tmp_path / "id_ed25519"),
        "mappings": [
            {"local_port": 5010, "hub_bind": "127.0.0.1", "hub_port": 15010},
            {"local_port": 5011, "hub_bind": "127.0.0.1", "hub_port": 15011},
        ],
        "state_path": str(tmp_path / "state.json"),
        "push": [f"{tmp_path}/traces:~/incoming/traces"],
    }
    values.update(overrides)
    path = tmp_path / "tunnel.json"
    path.write_text(json.dumps(values))
    return path


def test_config_defaults_and_absolute_path_preservation(tmp_path):
    config = tunnel.load_config(write_config(tmp_path))
    assert len(config["mappings"]) == 2
    assert config["backoff_start"] == tunnel.BACKOFF_START
    assert config["backoff_cap"] == tunnel.BACKOFF_CAP


@pytest.mark.parametrize("mutation, message", [
    ({"hub": ""}, "ssh destination"),
    ({"hub": "-oProxyCommand=x"}, "ssh destination"),
    ({"identity": ""}, "key path"),
    ({"mappings": []}, "nonempty list"),
    ({"mappings": [{"local_port": 0, "hub_bind": "127.0.0.1", "hub_port": 1}]},
     "local_port must be an integer"),
    ({"mappings": [{"local_port": 5010, "hub_bind": "", "hub_port": 1}]}, "hub_bind"),
    ({"mappings": [{"local_port": 5010, "hub_bind": "127.0.0.1"}]}, "requires exactly"),
    ({"mappings": [
        {"local_port": 1, "hub_bind": "127.0.0.1", "hub_port": 9},
        {"local_port": 2, "hub_bind": "127.0.0.1", "hub_port": 9}]}, "duplicate hub port"),
    ({"state_path": ""}, "state_path"),
    ({"backoff_start": 0}, "backoff"),
    ({"backoff_cap": 0.1}, "backoff"),
    ({"push": [""]}, "local:remote"),
])
def test_config_validation_fails_closed(tmp_path, mutation, message):
    with pytest.raises(tunnel.TunnelError, match=message):
        tunnel.load_config(write_config(tmp_path, **mutation))


def test_forward_argv_binds_loopback_and_orders_one_forward_per_mapping(tmp_path):
    config = tunnel.load_config(write_config(tmp_path))
    argv = tunnel.forward_argv(config)
    assert argv[0] == "ssh"
    forwards = [argv[index + 1] for index, part in enumerate(argv) if part == "-R"]
    assert forwards == ["127.0.0.1:15010:127.0.0.1:5010",
                        "127.0.0.1:15011:127.0.0.1:5011"]
    # The destination is last and the local side is never a routable address.
    assert argv[-1] == "manager.example"
    assert all(forward.split(":")[2] == "127.0.0.1" for forward in forwards)
    assert "-N" in argv and "ExitOnForwardFailure=yes" in argv


def test_daemon_reconnects_with_backoff_and_records_the_error(tmp_path, monkeypatch):
    config = tunnel.load_config(write_config(tmp_path, backoff_start=0.001,
                                             backoff_cap=0.004))
    sleeps = []
    monkeypatch.setattr(tunnel.CLOCK, "sleep", lambda seconds: sleeps.append(seconds))
    errors, attempts = [], []
    monkeypatch.setattr(tunnel, "POPEN", lambda argv, **kw: (_ for _ in ()).throw(
        OSError("host key verification failed")))
    real_write = tunnel.write_state

    def capture(path, cfg, **kwargs):
        # The supervisor writes twice per attempt: once when connecting (no
        # error yet) and once when the connection ends, carrying the reason.
        if kwargs.get("last_error") is not None:
            errors.append(kwargs["last_error"])
        attempts.append(kwargs.get("attempts"))
        return real_write(path, cfg, **kwargs)

    monkeypatch.setattr(tunnel, "write_state", capture)
    stopped = threading.Event()

    def sleep_then_stop(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            stopped.set()

    monkeypatch.setattr(tunnel.CLOCK, "sleep", sleep_then_stop)
    attempts_made = tunnel.daemon(config, stop=stopped)
    assert attempts_made == 3
    # Every recorded failure names the cause, not just "ssh died".
    assert all("host key verification failed" in item for item in errors)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["connected"] is False
    assert state["mappings"] == config["mappings"]


def test_daemon_retries_a_nonzero_ssh_exit_with_capped_backoff(tmp_path, monkeypatch):
    config = tunnel.load_config(write_config(tmp_path, backoff_start=0.001,
                                             backoff_cap=0.002))
    delays, rounds = [], {"n": 0}

    class Child:
        returncode = None
        stderr = None

        def poll(self):
            return 7

        def wait(self, timeout=None):
            return 7

        def terminate(self):
            pass

        def kill(self):
            pass

    class Buf:
        def read(self):
            return b"remote port forwarding failed\n"

    Child.stderr = Buf()
    monkeypatch.setattr(tunnel, "POPEN", lambda argv, **kw: Child())
    stopped = threading.Event()

    def fake_sleep(seconds):
        delays.append(seconds)
        rounds["n"] += 1
        if rounds["n"] >= 4:
            stopped.set()

    monkeypatch.setattr(tunnel.CLOCK, "sleep", fake_sleep)
    tunnel.daemon(config, stop=stopped)
    # Backoff doubles from the start value and stops growing at the cap.
    assert delays[0] == 0.001
    assert max(delays) <= 0.002


def test_install_prints_both_ends_and_never_the_private_key(tmp_path):
    config = tunnel.load_config(write_config(tmp_path))
    local = tmp_path / "id_ed25519"
    local.write_text("PRIVATE-KEY-MATERIAL")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA test\n")

    class Run:
        returncode = 0
        stdout = "ssh-ed25519 AAAA test\n"
        stderr = ""

    monkey = tunnel.RUNNER
    tunnel.RUNNER = lambda *args, **kwargs: Run()
    try:
        commands = tunnel.install_commands(
            config, root="/opt/harborrl-tunnel",
            public_key="ssh-ed25519 AAAA test")
    finally:
        tunnel.RUNNER = monkey

    # The key is not authorized yet, so one paste cannot do both ends: the
    # hub command authorizes it, the host command can only follow.
    assert set(commands) == {"hub", "host"}
    hub, host = commands["hub"], commands["host"]
    assert "authorized_keys" in hub and "grep -qxF" in hub  # idempotent append
    assert "tunnel.json" in host and "harborrl_h_cluster.tunnel daemon" in host
    # The package travels over the same one-way path: the host has no route
    # to anything but the hub, and no shared storage to read it from.
    assert "tar -C" in host and "harborrl_h_cluster" in host
    # The mappings travel inside the site config the host command writes.
    assert json.dumps(config["mappings"], separators=(",", ":")) in host
    # Neither command may carry the private key itself.
    for command in commands.values():
        assert "PRIVATE-KEY-MATERIAL" not in command


def test_state_writes_are_atomic_and_carry_the_reason(tmp_path):
    config = tunnel.load_config(write_config(tmp_path))
    tunnel.write_state(config["state_path"], config, pid=1234, attempts=2,
                       last_error="connection reset", connected=False)
    payload = json.loads((tmp_path / "state.json").read_text())
    assert payload["pid"] == 1234 and payload["attempts"] == 2
    assert payload["last_error"] == "connection reset"
    assert payload["connected"] is False
    assert not (tmp_path / "state.tmp").exists()


def test_push_uses_rsync_with_a_bounded_timeout(tmp_path, monkeypatch):
    config = tunnel.load_config(write_config(tmp_path))
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(tunnel, "RUNNER", run)
    results = tunnel.push(config, once=True)
    assert results[0]["returncode"] == 0
    argv, kwargs = calls[0]
    assert argv[0] == "rsync" and "-a" in argv
    assert kwargs["timeout"] == 1800


def test_push_reports_a_malformed_entry_instead_of_guessing(tmp_path):
    config = tunnel.load_config(write_config(tmp_path, push=["/only/local"]))
    with pytest.raises(tunnel.TunnelError, match="local:remote"):
        tunnel.push(config, once=True)
