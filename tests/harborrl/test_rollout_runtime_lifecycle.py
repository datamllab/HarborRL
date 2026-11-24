"""CPU regressions for embedded loop ownership, cleanup admission and shutdown."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from harborrl.rollout import runtime


def cached_runtime_fixture(tmp_path, monkeypatch, *, sandbox=None):
    from harborrl.gateway import server, sglang_backend

    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]")
    profile = tmp_path / "profile.json"
    profile.write_text('{"model": "policy"}')
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2],
                                get_vocab=lambda: {"token": 1})
    tok, template = runtime.tokenizer_fingerprint(tokenizer)
    config = runtime.RolloutRuntimeConfig(
        "run", str(tmp_path), str(catalog), str(profile), "127.0.0.1", 0,
        "http://gateway", "/python", (), {}, 2, 128, 1024, 1, 2, tok, template,
        True, 1, "qwen3-8B", "http://router", sandbox=sandbox, runner_transport="local")
    monkeypatch.setattr(server, "make_server", lambda *a, **k: SimpleNamespace(
        serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None))
    monkeypatch.setattr(sglang_backend, "SGLangBackend", lambda *a, **k: SimpleNamespace())
    return config, tokenizer


@pytest.mark.parametrize("sandbox_enabled", [False, True])
def test_cached_embedded_runtime_survives_contended_rollout_rounds(tmp_path, monkeypatch,
                                                                  sandbox_enabled):
    runtime.close_runtime()
    sandbox = {"max_live_instances": 1,
               "bindings": [{"image": "image", "max_live_instances": 1}]} if sandbox_enabled else None
    config, tokenizer = cached_runtime_fixture(tmp_path, monkeypatch, sandbox=sandbox)
    seen, loops = [], []
    active, peak, completed = 0, 0, 0

    async def round_():
        current = await runtime.runtime(config, tokenizer=tokenizer)
        seen.append(current)
        loops.append(asyncio.get_running_loop())

        async def slot(index):
            nonlocal active, peak, completed
            admission = (current.sandbox_admission.lease(sandbox["bindings"][0], tmp_path / str(index))
                         if sandbox else nullcontext())
            async with admission, current.semaphore:
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.01)
                active -= 1
                completed += 1

        await asyncio.gather(slot(0), slot(1))

    try:
        for _ in range(3):
            runtime.run_embedded(round_())
        assert len({id(item) for item in seen}) == 1
        assert len({id(loop) for loop in loops}) == 1
        assert completed == 6 and peak == 1
        old_loop = loops[0]
        runtime.close_runtime()
        assert old_loop.is_closed()
        runtime.run_embedded(round_())
        assert seen[-1] is not seen[0] and loops[-1] is not old_loop
    finally:
        runtime.close_runtime()


def test_cached_runtime_rejects_foreign_loop_and_config(tmp_path, monkeypatch):
    runtime.close_runtime()
    config, tokenizer = cached_runtime_fixture(tmp_path, monkeypatch)
    try:
        asyncio.run(runtime.runtime(config, tokenizer=tokenizer))
        with pytest.raises(runtime.RuntimeLifecycleError, match="different event loop"):
            asyncio.run(runtime.runtime(config, tokenizer=tokenizer))
        with pytest.raises(runtime.RuntimeLifecycleError, match="configuration"):
            asyncio.run(runtime.runtime(replace(config, run_id="other"), tokenizer=tokenizer))
    finally:
        runtime.close_runtime()


def test_slime_hook_reuses_runtime_across_rollouts_and_closes_after_failure(tmp_path, monkeypatch):
    from harborrl.backends.slime_v032 import rollout

    runtime.close_runtime()
    config, tokenizer = cached_runtime_fixture(tmp_path, monkeypatch)
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=2)
    source = SimpleNamespace(get_samples=lambda _: [[object(), object()]])
    seen, loops, completed = [], [], []
    fail = False

    async def runner(args_, groups, policy_version):
        current = await runtime.runtime(config, tokenizer=tokenizer)
        seen.append(current)
        loops.append(asyncio.get_running_loop())

        async def slot(index):
            async with current.semaphore:
                await asyncio.sleep(.001)
                completed.append(index)

        await asyncio.gather(slot(0), slot(1))
        if fail:
            raise ValueError('rollout failed')
        return []

    monkeypatch.setattr(rollout, 'lock_policy_version', lambda _: '1')
    monkeypatch.setattr(rollout, 'export_training_batch', lambda *a, **k: object())
    monkeypatch.setattr(rollout, 'to_slime_samples', lambda *a, **k: [])
    monkeypatch.setattr(rollout, 'collect_metrics', lambda *a, **k: {})

    def generate(index):
        return rollout.generate_rollout(args, index, source, runner=runner,
                                        output_factory=SimpleNamespace)

    try:
        for index in range(3):
            assert generate(index).samples == []
        assert len({id(item) for item in seen}) == 1
        assert len({id(loop) for loop in loops}) == 1
        assert completed == [0, 1] * 3
        fail = True
        with pytest.raises(ValueError, match='rollout failed'):
            generate(3)
        assert loops[0].is_closed()
        assert runtime._RUNTIME is None and runtime._EMBEDDED_LOOP is None
        fail = False
        generate(4)
        assert seen[-1] is not seen[0] and loops[-1] is not loops[0]
    finally:
        runtime.close_runtime()


def test_embedded_shutdown_awaits_tasks_before_closing_gateway(monkeypatch):
    runtime.close_runtime()
    events = []

    async def pending():
        try:
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(0)
            events.append("task cleaned")

    async def launch():
        asyncio.create_task(pending())
        await asyncio.sleep(0)

    runtime.run_embedded(launch())
    loop = runtime._EMBEDDED_LOOP
    monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(close=lambda: events.append("gateway closed")))
    runtime.close_runtime()
    assert events == ["task cleaned", "gateway closed"]
    assert loop.is_closed()
    assert runtime._RUNTIME is None and runtime._EMBEDDED_LOOP is None


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_embedded_failure_waits_for_other_groups_and_releases_loop(monkeypatch, error_type):
    from harborrl.backends.slime_v032 import rollout

    runtime.close_runtime()
    events = []

    async def generate(args, group, params, *, rollout_id):
        if group == "failed":
            await asyncio.sleep(.01)
            raise error_type("original group error")
        try:
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(.01)
            events.append("sibling cleaned")

    monkeypatch.delenv("HARBORRL_ROLLOUT_SERVICE_URL", raising=False)
    monkeypatch.setattr(rollout, "generate_group", generate)
    # Python 3.10's asyncio.gather can discard CancelledError messages.
    match = "original group error" if error_type is ValueError else None
    with pytest.raises(error_type, match=match):
        runtime.run_embedded(rollout.run_native_groups(
            SimpleNamespace(rollout_max_response_len=128), ["failed", "pending"], None, 0))
    assert events == ["sibling cleaned"]
    assert runtime._EMBEDDED_LOOP is None and runtime._RUNTIME is None


@pytest.mark.parametrize("rollout_failed", [False, True])
def test_embedded_shutdown_error_releases_resources_and_preserves_cause(monkeypatch, caplog,
                                                                     rollout_failed):
    runtime.close_runtime()

    def fail_close():
        raise RuntimeError("gateway shutdown failed")

    loops = []

    async def round_():
        loops.append(asyncio.get_running_loop())
        monkeypatch.setattr(runtime, "_RUNTIME", SimpleNamespace(close=fail_close))
        if rollout_failed:
            raise ValueError("original rollout failure")

    try:
        if rollout_failed:
            with pytest.raises(ValueError, match="original rollout failure"):
                runtime.run_embedded(round_())
            assert "gateway shutdown failed" in caplog.text
        else:
            runtime.run_embedded(round_())
            with pytest.raises(RuntimeError, match="gateway shutdown failed"):
                runtime.close_runtime()
        assert loops[0].is_closed()
        assert runtime._RUNTIME is None and runtime._EMBEDDED_LOOP is None
        runtime.run_embedded(asyncio.sleep(0))
    finally:
        runtime.close_runtime()


def test_embedded_rejects_foreign_thread_but_allows_idle_shutdown():
    runtime.close_runtime()

    async def round_():
        return asyncio.get_running_loop()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            loop = pool.submit(runtime.run_embedded, round_()).result(timeout=5)
        with pytest.raises(runtime.RuntimeLifecycleError, match="owning thread"):
            runtime.run_embedded(round_())
        runtime.close_runtime()
        assert loop.is_closed()
    finally:
        runtime.close_runtime()


def test_embedded_rejects_nested_sync_calls_and_active_shutdown():
    runtime.close_runtime()

    async def round_():
        with pytest.raises(runtime.RuntimeLifecycleError, match="active event loop"):
            runtime.run_embedded(asyncio.sleep(0))
        with pytest.raises(runtime.RuntimeLifecycleError, match="between calls"):
            runtime.close_runtime()

    try:
        runtime.run_embedded(round_())
        assert not runtime._EMBEDDED_LOOP.is_closed()
    finally:
        runtime.close_runtime()


@pytest.mark.parametrize("marker_path", ["cleanup-required.json", "runner/cleanup-required.json"])
def test_runtime_restart_requires_cleanup_confirmation(tmp_path, monkeypatch, marker_path):
    config, tokenizer = cached_runtime_fixture(tmp_path, monkeypatch)
    marker = tmp_path / "native-rollouts/group/old-attempt" / marker_path
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    with pytest.raises(runtime.CleanupRequiredError, match="old-attempt"):
        runtime.RolloutRuntime(config, tokenizer)
    marker.unlink()
    restarted = runtime.RolloutRuntime(config, tokenizer)
    restarted.close()
