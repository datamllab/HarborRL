import asyncio

import httpx
import pytest

pytest.importorskip('harbor.environments.opensandbox')

from harborrl.rollout.harbor_job.environments.opensandbox import (
    ManagedOpenSandboxEnvironment,
)


@pytest.mark.parametrize('delete_status,expected', [(204, 'deleted'), (500, 'delete_failed')])
def test_allocated_id_cleanup_requires_confirmed_absence(tmp_path, monkeypatch, delete_status, expected):
    calls = []

    async def handler(request):
        calls.append((request.method, request.url.path))
        assert request.url.path == '/v1/sandboxes/sbx-owned'
        if request.method == 'DELETE':
            return httpx.Response(delete_status)
        return httpx.Response(200 if len(calls) == 1 else 404)

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client_class(
        **kwargs, transport=httpx.MockTransport(handler)))
    env = object.__new__(ManagedOpenSandboxEnvironment)
    env.cfg = {'domain': 'https://platform.test', 'cleanup_timeout_s': 1}
    env.binding = {'image': 'owned'}
    env._api_key = 'test-only'
    env.evidence = tmp_path
    env.record = {'sandbox_id': 'sbx-owned', 'runtime_digest': 'test', 'events': []}
    asyncio.run(env._cleanup_allocated())
    assert env.record['state'] == expected
    assert (tmp_path / 'cleanup-required.json').exists() == (expected == 'delete_failed')
    assert calls[:2] == [('GET', '/v1/sandboxes/sbx-owned'), ('DELETE', '/v1/sandboxes/sbx-owned')]


def test_startup_failure_preserves_cause_types_without_sdk_secrets(tmp_path):
    import json

    env = object.__new__(ManagedOpenSandboxEnvironment)
    env.cfg = {'create_timeout_s': 600, 'request_timeout_s': 180}
    env.evidence = tmp_path
    env.record = {'sandbox_id': 'sbx-owned'}
    cause = httpx.ReadTimeout('private token and signed URL')
    outer = RuntimeError('secret response body')
    outer.__cause__ = cause
    env.startup_failed('create', outer)
    text = (tmp_path / 'sandbox-startup-failed.json').read_text()
    assert 'private token' not in text and 'secret response' not in text
    assert json.loads(text) == {'stage': 'create', 'sandbox_id': 'sbx-owned',
        'causes': [{'error_type': 'RuntimeError'}, {'error_type': 'ReadTimeout'}],
        'create_timeout_s': 600, 'request_timeout_s': 180}
