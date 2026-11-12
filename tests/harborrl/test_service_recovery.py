"""Restart restores durable group state, never launches an old Runner twice."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from test_rollout_service import (
    FakeRuntime,
    activated_states,
    group_request,
    write_group_irs,
)

from harborrl.rollout.runtime import CleanupAdmission, CleanupRequiredError
from harborrl.rollout.service import (
    GROUP_DIR,
    RolloutService,
    ServiceError,
    request_digest,
    validate_group_request,
)
from harborrl.trajectories.native import publish, read_json

MANIFEST = {"run_id": "run-1", "group_size": 2}


def ready(service, root):
    service.register_run("run-1", MANIFEST)
    service.attach_runtime("run-1", SimpleNamespace(drain_timeout=1), FakeRuntime(root))
    service.activate_policy("run-1", activated_states(), "1")


def test_completed_group_survives_restart_without_regeneration(tmp_path):
    calls = []
    async def executor(_config, _runtime, request):
        calls.append(request["group_id"])
        return write_group_irs(tmp_path, request)
    first = RolloutService(run_root=tmp_path, executor=executor, allow_synthetic=True)
    try:
        ready(first, tmp_path)
        original = first.submit_group("run-1", group_request())
    finally:
        first.close()
    directory = tmp_path / GROUP_DIR / "run-1/group-0"
    evidence = {p.name: p.read_bytes() for p in directory.glob("*.json")}
    async def forbidden(*_):
        raise AssertionError("recovered group must not run again")
    second = RolloutService(run_root=tmp_path, executor=forbidden, allow_synthetic=True)
    try:
        reply = second.register_run("run-1", MANIFEST)
        assert reply["existed"] and reply["recovered_groups"] == 1
        restored = second.submit_group("run-1", group_request())
        assert restored == original
        assert calls == ["group-0"]
        assert {p.name: p.read_bytes() for p in directory.glob("*.json")} == evidence
        changed = group_request()
        changed["sampling"]["max_new_tokens"] += 1
        with pytest.raises(ServiceError) as error:
            second.submit_group("run-1", changed)
        assert error.value.code == "conflict"
    finally:
        second.close()


@pytest.mark.parametrize("status,filename", [("failed", "failed.json"), ("cancelled", "cancelled.json")])
def test_failed_and_cancelled_groups_do_not_silently_retry(tmp_path, status, filename):
    request = group_request()
    directory = tmp_path / GROUP_DIR / "run-1/group-0"
    publish(directory / "request.json", request)
    publish(directory / filename, {"group_id": "group-0", "request_digest": request_digest(request),
                                   "status": status, "created_at": 1, "error": "original reason"})
    service = RolloutService(run_root=tmp_path)
    try:
        service.register_run("run-1", MANIFEST)
        recovered = service.submit_group("run-1", request)
        assert recovered["status"] == status and recovered["error"] == "original reason"
        assert service.runs["run-1"].groups["group-0"].future is None
    finally:
        service.close()


@pytest.mark.parametrize("running_record", [False, True])
def test_interrupted_group_restores_unknown_and_blocks_new_generation(tmp_path, running_record):
    request = group_request()
    directory = tmp_path / GROUP_DIR / "run-1/group-0"
    publish(directory / "request.json", request)
    if running_record:
        publish(directory / "running.json", {"group_id": "group-0", "request_digest": request_digest(request),
                                            "status": "running", "created_at": 1})
    for _ in range(2):
        service = RolloutService(run_root=tmp_path)
        try:
            service.register_run("run-1", MANIFEST)
            state = service.submit_group("run-1", request)
            assert state["status"] == "unknown" and "cleanup" in state["error"]
            assert service.runs["run-1"].groups["group-0"].future is None
            with pytest.raises(ServiceError) as error:
                service.submit_group("run-1", group_request(group_index=1), wait=False)
            assert error.value.code == "admission_closed"
            with pytest.raises(ServiceError) as error:
                service.cancel_run("run-1")
            assert error.value.code == "cleanup_required"
            with pytest.raises(CleanupRequiredError):
                CleanupAdmission(tmp_path / "native-rollouts").check()
        finally:
            service.close()
    assert read_json(directory / "unknown.json")["status"] == "unknown"
    assert read_json(directory / "request.json") == request


def test_run_directory_has_one_live_service_owner(tmp_path):
    first = RolloutService(run_root=tmp_path)
    try:
        with pytest.raises(ServiceError) as error:
            RolloutService(run_root=tmp_path)
        assert error.value.code == "run_directory_busy"
    finally:
        first.close()
    second = RolloutService(run_root=tmp_path)
    second.close()


def test_restart_rejects_changed_registration_and_corrupt_group(tmp_path):
    first = RolloutService(run_root=tmp_path)
    first.register_run("run-1", MANIFEST)
    first.close()
    second = RolloutService(run_root=tmp_path)
    try:
        with pytest.raises(ServiceError) as error:
            second.register_run("run-1", {**MANIFEST, "group_size": 3})
        assert error.value.code == "conflict"
        request = group_request()
        directory = tmp_path / GROUP_DIR / "run-1/group-0"
        publish(directory / "request.json", request)
        publish(directory / "failed.json", {"status": "failed", "group_id": "group-0", "request_digest": "wrong"})
        with pytest.raises(ServiceError) as error:
            second.register_run("run-1", MANIFEST)
        assert error.value.code == "invalid_persisted_group"
        assert not second.runs
    finally:
        second.close()


def test_complete_is_reported_only_after_result_is_durable(tmp_path, monkeypatch):
    from harborrl.rollout import service as module
    publishing, release = threading.Event(), threading.Event()
    original = module.publish
    def delayed_result(path, value):
        if path.name == "result.json":
            publishing.set()
            assert release.wait(5)
        return original(path, value)
    async def executor(_config, _runtime, request):
        return write_group_irs(tmp_path, request)
    service = RolloutService(run_root=tmp_path, executor=executor, allow_synthetic=True)
    try:
        ready(service, tmp_path)
        monkeypatch.setattr(module, "publish", delayed_result)
        service.submit_group("run-1", group_request(), wait=False)
        assert publishing.wait(5)
        state = service.group_status("run-1", "group-0")
        assert state["status"] == "running" and "result" not in state
        release.set()
        service.runs["run-1"].groups["group-0"].future.result(timeout=5)
        assert service.group_status("run-1", "group-0")["status"] == "complete"
    finally:
        release.set()
        service.close()


def test_executor_setup_failure_becomes_terminal_not_stuck_running(tmp_path, monkeypatch):
    from harborrl.rollout import service as module
    original = module.publish
    def failed_running(path, value):
        if path.name == "running.json":
            raise OSError("cannot persist running state")
        return original(path, value)
    async def forbidden(*_):
        raise AssertionError("setup failed before allocation")
    service = RolloutService(run_root=tmp_path, executor=forbidden)
    try:
        ready(service, tmp_path)
        monkeypatch.setattr(module, "publish", failed_running)
        result = service.submit_group("run-1", group_request())
        assert result["status"] == "failed" and "cannot persist" in result["error"]
    finally:
        service.close()


@pytest.mark.parametrize("bad_id", ["..", "../escape", "a/b", "a\\b"])
def test_storage_identities_are_one_directory_component(tmp_path, bad_id):
    request = group_request()
    request["group_id"] = bad_id
    with pytest.raises(ServiceError, match="directory component"):
        validate_group_request(request)
    service = RolloutService(run_root=tmp_path)
    try:
        with pytest.raises(ServiceError, match="directory component"):
            service.register_run(bad_id, {"run_id": bad_id})
    finally:
        service.close()


def test_running_group_is_cancelled_before_lock_is_released(tmp_path):
    # The lock must also prevent a second owner while asynchronous cleanup runs.
    started = asyncio.Event()
    async def executor(*_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(.01)
    service = RolloutService(run_root=tmp_path, executor=executor)
    try:
        ready(service, tmp_path)
        service.submit_group("run-1", group_request(), wait=False)
    finally:
        service.close()
    recovered = RolloutService(run_root=tmp_path)
    try:
        recovered.register_run("run-1", MANIFEST)
        assert recovered.group_status("run-1", "group-0")["status"] == "cancelled"
    finally:
        recovered.close()
