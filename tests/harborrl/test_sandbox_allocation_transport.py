import asyncio
import json

import httpx
import pytest

from harborrl.rollout.harbor_job.sandbox_transport import SandboxStartupTransport


def test_generic_transport_observes_allocation_without_retry():
    async def run():
        calls, allocations = [], []

        def handler(request):
            calls.append(request.method)
            return httpx.Response(200, json={'id': 'sbx-one'}) if request.method == 'POST' else httpx.Response(503)

        transport = SandboxStartupTransport('https://sandbox.example/v1', allocations.append,
                                           transport=httpx.MockTransport(handler))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post('https://sandbox.example/v1/sandboxes')
            result = await client.get('https://sandbox.example/v1/sandboxes/sbx-one/endpoints/44772')
            assert result.status_code == 503
        assert calls == ['POST', 'GET'] and allocations == ['sbx-one']

    asyncio.run(run())


@pytest.mark.parametrize('kind', ['401', 'timeout', 'cancelled'])
def test_request_evidence_keeps_only_safe_lifecycle_metadata(kind):
    async def run():
        records = []

        async def handler(request):
            if kind == 'timeout':
                raise httpx.ReadTimeout('private signed URL and token')
            if kind == 'cancelled':
                raise asyncio.CancelledError('private SDK details')
            return httpx.Response(401, text='private response body')

        transport = SandboxStartupTransport('https://sandbox.example/v1', lambda _: None,
            transport=httpx.MockTransport(handler), on_request=records.append)
        async with httpx.AsyncClient(transport=transport) as client:
            url = 'https://sandbox.example/v1/sandboxes/sbx-one/endpoints/44772?token=private'
            if kind == '401':
                assert (await client.get(url, headers={'Authorization': 'private'})).status_code == 401
            else:
                with pytest.raises(httpx.ReadTimeout if kind == 'timeout' else asyncio.CancelledError):
                    await client.get(url)
        assert len(records) == 1
        record = records[0]
        assert record.keys() == {'method', 'operation', 'http_status', 'error_type', 'elapsed_sec', 'time'}
        assert record['method'] == 'GET' and record['operation'] == 'endpoint'
        assert record['http_status'] == (401 if kind == '401' else None)
        assert record['error_type'] == {'401': None, 'timeout': 'ReadTimeout', 'cancelled': 'CancelledError'}[kind]
        assert record['elapsed_sec'] >= 0
        assert 'private' not in json.dumps(records)

    asyncio.run(run())
