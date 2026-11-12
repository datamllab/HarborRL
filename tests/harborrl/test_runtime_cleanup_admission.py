"""Cleanup markers block retries, queued slots and restarted runtimes."""
import asyncio
from types import SimpleNamespace

import pytest

from harborrl.rollout import runtime


@pytest.mark.parametrize("marker_path", [
    "cleanup-required.json", "group/attempt/runner/cleanup-required.json",
])
def test_runtime_restart_checks_cleanup_before_starting_gateway(tmp_path, monkeypatch, marker_path):
    catalog, profile = tmp_path / "catalog.json", tmp_path / "profile.json"
    catalog.write_text("[]")
    profile.write_text('{}')
    marker = tmp_path / "native-rollouts" / marker_path
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{}')
    # Gateways start eagerly in __init__ here; the cleanup check runs before
    # any gateway construction, so construction itself is the forbidden seam.
    def forbidden(*_):
        raise AssertionError("gateway must not start before cleanup is confirmed")
    import harborrl.gateway.server as gateway_server
    monkeypatch.setattr(gateway_server, "make_server", forbidden)
    config = SimpleNamespace(run_root=tmp_path, catalog_path=catalog, profile_path=profile)
    with pytest.raises(runtime.CleanupRequiredError, match="cleanup unresolved"):
        runtime.RolloutRuntime(config, None)
    marker.unlink()
    runtime.CleanupAdmission(tmp_path / "native-rollouts").check()


@pytest.mark.parametrize("marker_path", ["cleanup-required.json", "runner/cleanup-required.json"])
@pytest.mark.parametrize("raises", [False, True])
def test_docker_cleanup_debt_blocks_retries_and_waiting_slots(tmp_path, monkeypatch, marker_path, raises):
    calls = []
    async def attempt(identity, **kwargs):
        calls.append(identity)
        await asyncio.sleep(.01)
        marker = kwargs["root"] / marker_path
        marker.parent.mkdir(parents=True)
        marker.write_text('{}')
        if raises:
            raise RuntimeError("Runner cleanup failed")
        return {}
    monkeypatch.setattr(runtime, "run_attempt", attempt)
    config = SimpleNamespace(run_id="run", group_size=2, max_attempts=3, workers=(),
        task_worker_overrides={}, runner_transport="local", sandbox=None,
        gateway_routing="modulo",
        gateway_url="http://gateway", runner_python="/python", max_output_tokens=128)
    shard = SimpleNamespace(url="http://gateway", registry=None)
    current = SimpleNamespace(profile={}, rollout_root=tmp_path,
                              semaphore=asyncio.Semaphore(1), registry=None,
                              shards=[shard])
    entry = {"id": "task", "path": "/task", "task_digest": "digest", "reward_profile": {}}
    with pytest.raises(runtime.CleanupRequiredError, match="automatic retry blocked"):
        asyncio.run(runtime.run_group(config, current, entry=entry,
            group_index=0, rollout_id=0, sampling_digest="x", policy_version="1"))
    assert len(calls) == 1
