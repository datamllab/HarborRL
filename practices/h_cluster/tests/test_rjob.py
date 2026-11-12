"""rjob provisioning adapter: contract, polling and quota safety.

No cluster is contacted: the submission runner, the clock and the address
record are all faked, so these tests pin the decisions — what a record may
contain, when a submit is refused, and what happens when a pod never answers.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harborrl_h_cluster import rjob


def write_config(tmp_path, **overrides):
    values = {
        "name_prefix": "harborrl-test",
        "image": "registry.example/ml-base:1",
        "workdir": str(tmp_path / "work"),
        "identity": str(tmp_path / "id_ed25519"),
        "mounts": ["gpfs://gpfs1/data:/mnt/data"],
        "timeout_s": 30,
        # The clock is faked in these tests; the interval only has to be legal.
        "poll_interval_s": 0.001,
    }
    values.update(overrides)
    path = tmp_path / "rjob.json"
    path.write_text(json.dumps(values))
    return path


def fake_runtime(monkeypatch, *, states, submissions=None, ssh_ok=True, clock=None):
    """Replace the runner and clock; record every argv the adapter builds."""
    calls = []
    state_iter = iter(states)

    def run(argv, **kwargs):
        calls.append(list(argv))
        head = argv[:2]
        if head == ["rjob", "submit"]:
            if submissions is not None:
                submissions.append(list(argv))
            return SimpleNamespace(returncode=0, stdout="created rjob_name: job-42\n",
                                   stderr="")
        if head == ["rjob", "get"]:
            state = next(state_iter, None)
            if state is None:
                # A deleted job answers successfully with no replica line.
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(
                returncode=0, stdout=f"  |- replica job-42-x: {state}, node-1\n", stderr="")
        if head == ["rjob", "delete"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if head[0] == "ssh":
            return SimpleNamespace(returncode=0 if ssh_ok else 255,
                                   stdout="ok\n" if ssh_ok else "", stderr="denied")
        if head[0] == "ssh-keygen":
            return SimpleNamespace(returncode=0, stdout="ssh-ed25519 AAAA test\n", stderr="")
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(rjob, "RUNNER", run)
    monkeypatch.setattr(rjob, "_public_key", lambda identity: "ssh-ed25519 AAAA test")
    # A clock that advances with every sleep: a frozen clock would make the
    # deadline unreachable and hang the poll loop forever.
    now = [0.0]
    monkeypatch.setattr(rjob.CLOCK, "monotonic", lambda: now[0])

    def sleep(seconds):
        now[0] += max(float(seconds), 1.0)

    monkeypatch.setattr(rjob.CLOCK, "sleep", sleep)
    return calls


def test_config_requires_the_submission_fields(tmp_path):
    path = write_config(tmp_path)
    path.write_text(json.dumps({"name_prefix": "x"}))
    with pytest.raises(rjob.RjobError, match="requires"):
        rjob.load_config(path)


def test_config_rejects_a_relative_workdir(tmp_path):
    with pytest.raises(rjob.RjobError, match="absolute path"):
        rjob.load_config(write_config(tmp_path, workdir="relative/work"))


def test_config_rejects_unknown_fields(tmp_path):
    path = write_config(tmp_path)
    raw = json.loads(path.read_text())
    raw["gpuu"] = 8
    path.write_text(json.dumps(raw))
    with pytest.raises(rjob.RjobError, match="unsupported fields"):
        rjob.load_config(path)


def test_submit_argv_carries_the_quota_and_namespace_flags(tmp_path):
    config = rjob.load_config(write_config(tmp_path))
    argv = rjob.submit_argv(config, "/tmp/work/bootstrap.sh")
    assert argv[:2] == ["rjob", "submit"]
    assert "--charged-group=datafoundation_gpu_pool" in argv
    assert "--namespace=ailab-datafoundation" in argv
    assert "--enable-sshd" in argv
    assert "--mount=gpfs://gpfs1/data:/mnt/data" in argv
    # The bootstrap script is the pod command: rjob has no exec action.
    assert argv[-3:] == ["--", "bash", "/tmp/work/bootstrap.sh"]


def test_bootstrap_script_is_idempotent_and_publishes_the_address(tmp_path):
    config = rjob.load_config(write_config(tmp_path))
    script = rjob.bootstrap_script(config, "ssh-ed25519 AAAA test")
    # Appending twice would leave a duplicate key, so the guard is by content.
    assert "grep -qxF" in script and "authorized_keys" in script
    assert "pgrep -x sshd" in script
    assert "pod-net.json" in script
    assert "ssh-ed25519 AAAA test" in script


def test_up_returns_the_record_after_state_address_and_ssh(tmp_path, monkeypatch):
    config = rjob.load_config(write_config(tmp_path))
    Path(config["workdir"]).mkdir(parents=True)
    fake_runtime(monkeypatch, states=["STARTING", "ACTIVE"])
    writes = iter([None, "100.98.93.42"])
    monkeypatch.setattr(rjob, "pod_address", lambda cfg: next(writes, None))
    record = rjob.up(config)
    assert record == {"host": "100.98.93.42", "user": "root", "port": 22,
                      "identity": config["identity"], "handle": "job-42"}


def test_up_refuses_a_second_submission_before_spending_quota(tmp_path, monkeypatch):
    config = rjob.load_config(write_config(tmp_path))
    workdir = Path(config["workdir"])
    workdir.mkdir(parents=True)
    (workdir / "rjob-handle.json").write_text('{"handle": "old-job"}\n')
    calls = fake_runtime(monkeypatch, states=[])
    with pytest.raises(rjob.RjobError, match="previous pod record"):
        rjob.up(config)
    assert not [call for call in calls if call[:2] == ["rjob", "submit"]]


def test_up_fails_closed_when_the_pod_never_publishes_an_address(tmp_path, monkeypatch):
    config = rjob.load_config(write_config(tmp_path, timeout_s=1))
    Path(config["workdir"]).mkdir(parents=True)
    fake_runtime(monkeypatch, states=["ACTIVE"] * 50)
    monkeypatch.setattr(rjob, "pod_address", lambda cfg: None)
    with pytest.raises(rjob.RjobError, match="publish its address"):
        rjob.up(config)


def test_up_fails_closed_when_ssh_is_refused(tmp_path, monkeypatch):
    config = rjob.load_config(write_config(tmp_path, timeout_s=1))
    Path(config["workdir"]).mkdir(parents=True)
    fake_runtime(monkeypatch, states=["ACTIVE"] * 50, ssh_ok=False)
    monkeypatch.setattr(rjob, "pod_address",
                        lambda cfg: "100.98.93.42")
    with pytest.raises(rjob.RjobError, match="non-interactive ssh"):
        rjob.up(config)


def test_down_deletes_only_the_recorded_handle(tmp_path, monkeypatch):
    config = rjob.load_config(write_config(tmp_path))
    Path(config["workdir"]).mkdir(parents=True)
    calls = fake_runtime(monkeypatch, states=[None])
    assert rjob.down(config, {"handle": "job-42"}) == "job-42"
    deletes = [call for call in calls if call[:2] == ["rjob", "delete"]]
    assert deletes and deletes[0][2] == "job-42"


def test_down_refuses_a_record_without_a_handle(tmp_path, monkeypatch):
    config = rjob.load_config(write_config(tmp_path))
    fake_runtime(monkeypatch, states=[])
    with pytest.raises(rjob.RjobError, match="no handle"):
        rjob.down(config, {"host": "1.2.3.4"})


def test_cli_prints_one_machine_readable_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rjob, "up", lambda cfg: {"host": "h", "handle": "job-1"})
    assert rjob.main(["up", "--config", str(write_config(tmp_path))]) == 0
    assert json.loads(capsys.readouterr().out) == {"host": "h", "handle": "job-1"}
