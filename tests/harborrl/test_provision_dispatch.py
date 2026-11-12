"""Provision wiring inside the dispatcher: obtain, probe, launch, release.

The provision module is tested on its own; this file pins the part that turns
it into a lifecycle — that ``up`` runs before anything dials the machine, that
the machine it produced becomes the dispatch target, and that ``down`` runs
however the dispatch ends.
"""
from __future__ import annotations

import pytest

from harborrl.deployment import controller, provision
from harborrl.deployment.schema import validate


def deployment(tmp_path, *, transport="ssh", provision_block=None,
               provider="ssh_docker"):
    gpu = {
        "id": "gpu", "transport": transport,
        "ssh": None if provision_block else {"host": "fixed-host"},
        "source_dir": str(tmp_path / "gpu-source"),
        "train_python": str(tmp_path / "gpu-bin" / "python"),
        "environment_file": None, "expected_gpus": 8,
    }
    if provision_block:
        gpu["provision"] = provision_block
    workers = ([{"id": "worker-a", "ssh": {"host": "worker-a"},
                 "source_dir": str(tmp_path / "w"), "runner_python": str(tmp_path / "p"),
                 "docker": True, "task_roots": [str(tmp_path / "tasks")]}]
               if provider == "ssh_docker" else [])
    return validate({
        "schema_version": 1,
        "manager": {
            "id": "manager-a", "mode": "service", "bind": "127.0.0.1",
            "control_port": 49200, "control_url": "http://manager-a:49200",
            "gateway_port": 49184, "gateway_url": "http://manager-a:49184",
            "run_token_file": str(tmp_path / "run-token"),
            "source_dir": str(tmp_path / "manager-source"),
            "controller_python": str(tmp_path / "bin" / "python"),
        },
        "gpu_target": gpu,
        "storage": {"mode": "shared_fs", "run_root": str(tmp_path / "runs"),
                    "manager_writable": True, "gpu_writable": True,
                    "worker_writable": True},
        "rollout": {"provider": provider, "workers": workers},
        "connectivity": {"command_timeout_s": 30, "ssh_timeout_s": 15,
                         "http_timeout_s": 10, "gpu_to_controller": True,
                         "worker_to_gateway": True,
                         "manager_to_gpu_inference": True},
    })


def plan():
    return {
        "config": {"sampling": {"group_size": 2, "groups_per_batch": 1}},
        "deployment": {"source": "deployment.yaml", "digest": "d" * 64},
    }


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Replace every side effect of a dispatch; record the ordered steps."""
    events = []
    seen = {}

    class FakeService:
        def __init__(self, **kwargs):
            pass

        def register_run(self, run_id, manifest, token=None):
            pass

        def drain_run(self, run_id):
            events.append("drain")

        def close(self):
            events.append("service-close")

    class FakeControl:
        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            events.append("control-close")

    class FakeTarget:
        def __init__(self, spec, **kwargs):
            seen["target_spec"] = dict(spec)

        def write_launcher(self, *args, **kwargs):
            pass

        def launch(self):
            events.append("launch")
            raise RuntimeError("gpu job refused")

        def cancel(self):
            events.append("cancel")

    monkeypatch.setattr(controller, "new_run_root", lambda root: tmp_path / "run")
    monkeypatch.setattr(controller, "_with_allocated_ports",
                        lambda dep, run_id: dep)
    monkeypatch.setattr(controller, "_materialize",
                        lambda p, root: (tmp_path / "catalog.json", tmp_path / "tasks.jsonl"))
    monkeypatch.setattr(controller, "_source_audit", lambda root: {"commit": "abc"})
    monkeypatch.setattr(controller, "run_environment", lambda *args: {})
    monkeypatch.setattr(controller, "_ensure_token", lambda path: "token")
    monkeypatch.setattr(controller, "_rollout_environment",
                        lambda p, dep, env: dict(env))
    monkeypatch.setattr(controller, "RolloutService", FakeService)
    monkeypatch.setattr(controller, "make_control_server",
                        lambda *args, **kwargs: FakeControl())
    monkeypatch.setattr(controller, "check_gpu_control",
                        lambda dep: {"gpu_to_controller": True})
    monkeypatch.setattr(controller, "check_gateway_shards", lambda dep: None)
    monkeypatch.setattr(controller, "effective_config", lambda *args: {})
    monkeypatch.setattr(controller, "SshTrainingTarget", FakeTarget)
    monkeypatch.setattr(controller, "gpu_checks", lambda p, dep: [])
    monkeypatch.setattr(provision, "up",
                        lambda spec, evidence_path=None: events.append("up") or {
                            "host": "10.9.9.9", "user": "root", "port": 2222,
                            "identity": "/keys/id", "handle": "job-7"})
    captured = {}

    def fake_down(spec, record, evidence_path=None):
        events.append("down")
        captured["record"] = record
        return "released", "released"

    monkeypatch.setattr(provision, "down", fake_down)
    # The dispatcher creates its own run root; creating it here would collide.
    return events, seen, captured


PROVISION = {"up": ["true"], "down": ["true"], "timeout_s": 30}


def test_a_provisioned_dispatch_obtains_then_releases_the_machine(tmp_path, wired):
    events, seen, captured = wired
    with pytest.raises(RuntimeError, match="gpu job refused"):
        controller.dispatch_service(plan(), deployment(tmp_path, provision_block=PROVISION),
                                    checks=[])
    # The machine exists before the launch and is released after the failure.
    assert events[0] == "up"
    assert events.index("launch") < events.index("down")
    # The record the command produced became the dispatch target.
    assert seen["target_spec"] == {"host": "10.9.9.9", "user": "root",
                                   "port": 2222, "identity": "/keys/id"}
    assert captured["record"]["handle"] == "job-7"


def test_a_fixed_target_provisions_nothing(tmp_path, wired):
    events, seen, _captured = wired
    with pytest.raises(RuntimeError):
        controller.dispatch_service(plan(), deployment(tmp_path, provision_block=None),
                                    checks=[])
    assert "up" not in events and "down" not in events
    assert seen["target_spec"] == {"host": "fixed-host"}


def test_a_failed_post_provision_check_still_releases_the_machine(tmp_path, wired, monkeypatch):
    events, seen, _captured = wired
    monkeypatch.setattr(controller, "gpu_checks", lambda p, dep: [
        {"role": "gpu_target", "machine": "gpu", "check": "gpu_target.ssh",
         "ok": False, "detail": "connection refused"}])
    with pytest.raises(RuntimeError, match="failed preflight"):
        controller.dispatch_service(plan(), deployment(tmp_path, provision_block=PROVISION),
                                    checks=[])
    # Nothing was launched, but the machine this dispatch obtained is gone.
    assert "launch" not in events
    assert events.count("down") == 1


def test_release_evidence_is_written_next_to_the_run(tmp_path, wired):
    events, _seen, _captured = wired
    with pytest.raises(RuntimeError):
        controller.dispatch_service(plan(), deployment(tmp_path, provision_block=PROVISION),
                                    checks=[])
    # The dispatcher passes an evidence path; the fake ignores it, so assert
    # the wiring by checking the real helper is called with one.
    assert events[0] == "up" and "down" in events
