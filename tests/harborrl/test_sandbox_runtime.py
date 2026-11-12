"""Shared Sandbox scheduling and service lifecycle contracts."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_deployment_config import deployment_mapping

from harborrl.backends.slime_v032.rollout_service_client import (
    RolloutServiceClient,
    ServiceRolloutError,
)
from harborrl.deployment.schema import validate
from harborrl.rollout.harbor_job.sandbox_config import SandboxAdmission
from harborrl.rollout.runtime import RolloutRuntimeConfig, run_slot


def test_sandbox_deployment_allows_local_or_remote_runners(tmp_path):
    cfg = deployment_mapping(tmp_path)
    cfg['rollout'] = {'provider': 'sandbox', 'workers': []}
    assert validate(cfg)['rollout']['workers'] == []
    cfg['rollout']['workers'] = deployment_mapping(tmp_path)['rollout']['workers']
    assert validate(cfg)['rollout']['workers']


def test_slow_group_is_posted_once_even_after_transport_timeout(monkeypatch):
    client = RolloutServiceClient('http://manager', 'run', 'token', poll_interval=0)
    calls = []
    def request(method, path, body=None):
        calls.append(method)
        if method == 'POST':
            raise ServiceRolloutError('timeout', code='transport')
        return {'status': 'running' if len(calls) == 2 else 'complete', 'result': {}}
    monkeypatch.setattr(client, '_request', request)
    assert client.submit_group({'group_id': 'group-0'})['status'] == 'complete'
    assert calls == ['POST', 'GET', 'GET']


def test_sandbox_cleanup_debt_prevents_retry(tmp_path, monkeypatch):
    from harborrl.rollout import runtime
    from harborrl.rollout.harbor_job import sandbox_config
    profile = tmp_path / 'profile.json'
    profile.write_text('{}')
    binding = {'image': 'image', 'max_live_instances': 1}
    sandbox = {'max_live_instances': 1, 'bindings': [binding]}
    config = RolloutRuntimeConfig('run', str(tmp_path), '', str(profile), '127.0.0.1', 1,
        'http://gateway', '/python', (), {}, 2, 128, 1024, 2, 2, 'tok', 'tpl', True, 1,
        'qwen3-8B', 'http://router', sandbox=sandbox, runner_transport='local')
    current = SimpleNamespace(
        rollout_root=tmp_path, semaphore=asyncio.Semaphore(2), registry=None, profile={},
        sandbox_admission=SandboxAdmission(sandbox),
        shards=(SimpleNamespace(name="gateway", url="http://gateway", registry=None),))
    calls = []
    async def attempt(identity, **kwargs):
        calls.append(kwargs)
        assert kwargs['runner_host'] is None
        assert kwargs['sandbox'] == sandbox
        kwargs['root'].mkdir(parents=True)
        (kwargs['root'] / 'cleanup-required.json').write_text('{}')
        raise RuntimeError('deletion unconfirmed')
    monkeypatch.setattr(runtime, 'run_attempt', attempt)
    monkeypatch.setattr(sandbox_config, 'task_binding', lambda *args: binding)
    entry = {'id': 'task', 'path': '/task', 'task_digest': 'digest', 'reward_profile': {}}
    with pytest.raises(RuntimeError, match='cleanup unresolved'):
        asyncio.run(run_slot(config, current, 0, 0, 0, entry, 'sampling', '1'))
    assert len(calls) == 1 and current.sandbox_admission.blocked
    restored = replace(config, workers=('worker',), runner_transport='ssh')
    assert json.loads(restored.environment()['HARBORRL_NATIVE_SANDBOX']) == sandbox


def test_group_failure_waits_for_sibling_cleanup(tmp_path, monkeypatch):
    from harborrl.rollout import runtime
    events = []
    async def attempt(identity, **kwargs):
        if identity.slot_id == '0':
            await asyncio.sleep(.01)
            raise RuntimeError('failed slot')
        try:
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(.01)
            events.append('cleanup complete')
    monkeypatch.setattr(runtime, 'run_attempt', attempt)
    config = SimpleNamespace(run_id='run', group_size=2, max_attempts=1, workers=(),
        task_worker_overrides={}, runner_transport='local', sandbox=None,
        gateway_routing='modulo',
        gateway_url='http://gateway', runner_python='/python', max_output_tokens=128)
    current = SimpleNamespace(
        profile={}, rollout_root=tmp_path, semaphore=asyncio.Semaphore(2), registry=None,
        shards=(SimpleNamespace(name="gateway", url="http://gateway", registry=None),))
    entry = {'id': 'task', 'path': '/task', 'task_digest': 'digest', 'reward_profile': {}}
    with pytest.raises(RuntimeError, match='exhausted its retry budget'):
        asyncio.run(runtime.run_group(config, current, entry=entry,
            group_index=0, rollout_id=0, sampling_digest='x', policy_version='1'))
    assert events == ['cleanup complete']
