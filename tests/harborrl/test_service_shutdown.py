"""Service shutdown joins real async cleanup; no model, Docker or GPU allocation."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from test_rollout_service import FakeRegistry, group_request

from harborrl.rollout.service import RolloutService, ServiceError


def service_for(root, executor, runtime_close=lambda: None, **kwargs):
    service = RolloutService(run_root=root, executor=executor, poll_interval=.005, **kwargs)
    service.register_run("run", {"run_id": "run"})
    runtime = SimpleNamespace(registry=FakeRegistry(), close=runtime_close)
    service.attach_runtime("run", SimpleNamespace(drain_timeout=1), runtime)
    service.activate_policy("run", [{"endpoint": "cpu", "healthy": True, "weight_version": "1"}], "1")
    return service


def test_close_joins_runner_finally_before_runtime_and_loop(tmp_path):
    started, cleaned = threading.Event(), threading.Event()
    order = []
    children = []
    async def executor(*_):
        import sys
        child = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time;time.sleep(60)")
        children.append(child)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(.02)
            child.terminate()
            await child.wait()
            order.append("runner cleaned")
            cleaned.set()
    service = service_for(tmp_path, executor, lambda: order.append("runtime closed"), cleanup_timeout=2)
    try:
        service.submit_group("run", group_request(), wait=False)
        assert started.wait(5)
        assert service.drain_run("run")["state"] == "drained"
        assert not cleaned.is_set()  # generation drain is not a group join
        service.close()
        assert order == ["runner cleaned", "runtime closed"]
        assert cleaned.is_set() and children[0].returncode is not None
        assert service.group_status("run", "group-0")["status"] == "cancelled"
        assert service.loop.is_closed() and not service.thread.is_alive()
        service.close()  # idempotent; no scheduling on an already closed loop
    finally:
        service.close()


def test_cleanup_timeout_is_visible_and_retry_does_not_recancel_finally(tmp_path):
    started, cleaning, release, cleaned = (threading.Event() for _ in range(4))
    closed = []
    async def executor(*_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            while not release.is_set():
                await asyncio.sleep(.005)
            cleaned.set()
    service = service_for(tmp_path, executor, lambda: closed.append(True), cleanup_timeout=.03)
    try:
        service.submit_group("run", group_request(), wait=False)
        assert started.wait(5)
        with pytest.raises(ServiceError) as error:
            service.cancel_run("run")
        assert error.value.code == "cleanup_timeout"
        assert cleaning.is_set() and not cleaned.is_set()
        group = service.runs["run"].groups["group-0"]
        assert not group.future.done() and not group.task.done()
        with pytest.raises(ServiceError, match="still cleaning"):
            service.close()
        assert service.loop.is_running() and not closed
        assert (tmp_path / "native-rollouts/cleanup-required.json").is_file()
        with pytest.raises(ServiceError, match="closing"):
            service.register_run("other", {"run_id": "other"})
        release.set()
        service.cleanup_timeout = 2
        service.close()
        assert cleaned.is_set() and closed == [True]
    finally:
        release.set()
        service.cleanup_timeout = 2
        service.close()


def test_close_blocks_new_admission_while_cleanup_is_running(tmp_path):
    started, cleaning, release = (threading.Event() for _ in range(3))
    failures = []
    async def executor(*_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            while not release.is_set():
                await asyncio.sleep(.005)
    service = service_for(tmp_path, executor, cleanup_timeout=2)
    def close():
        try:
            service.close()
        except Exception as exc:  # noqa: BLE001 - report worker failures to the test thread
            failures.append(exc)
    try:
        service.submit_group("run", group_request(), wait=False)
        assert started.wait(5)
        closer = threading.Thread(target=close)
        closer.start()
        assert cleaning.wait(5)
        with pytest.raises(ServiceError, match="closing"):
            service.submit_group("run", group_request(group_index=1), wait=False)
        with pytest.raises(ServiceError, match="closing"):
            service.announce_serving("run", "http://cpu")
        release.set()
        closer.join(5)
        assert not closer.is_alive() and not failures
    finally:
        release.set()
        service.close()


def test_default_cleanup_budget_covers_runner_and_sandbox_deadlines(tmp_path):
    service = RolloutService(run_root=tmp_path)
    try:
        assert service._cleanup_budget() > 120
        service.register_run("run", {"run_id": "run"})
        service.attach_runtime("run", SimpleNamespace(sandbox={"cleanup_timeout_s": 300}), None)
        assert service._cleanup_budget() > 315
    finally:
        service.close()


def test_close_does_not_interrupt_slower_sibling_cleanup(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as runtime_module

    started = threading.Event()
    events, active = [], set()
    async def slot(_config, _runtime, _slots, index, _entry, _reward):
        active.add(index)
        if len(active) == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            if index == 1:
                await asyncio.sleep(.05)
            events.append(f"slot {index} cleaned")
    monkeypatch.setattr(runtime_module, "_run_slot", slot)
    async def executor(config, current, request):
        return await runtime_module.run_group(
            config, current, entry=request["task"], group_index=0, rollout_id=0,
            sampling_digest="sampling", policy_version="1", group_size=2)
    service = service_for(tmp_path, executor, lambda: events.append("runtime closed"), cleanup_timeout=2)
    service.runs["run"].config = SimpleNamespace(run_id="run", max_attempts=1, drain_timeout=1)
    service.runs["run"].runtime.profile = {}
    try:
        service.submit_group("run", group_request(), wait=False)
        assert started.wait(5)
        service.close()
        assert events == ["slot 0 cleaned", "slot 1 cleaned", "runtime closed"]
    finally:
        service.close()
