"""Synthetic CPU Gateway/coordinator recovery, never real training evidence."""
import asyncio
import errno
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from test_native_contracts import identity
from test_native_gateway import Backend, payload, setup
from test_native_lifecycle import PROFILE
from test_sandbox_reliability import _reliability_runtime_config

from harborrl.gateway.messages import ProtocolError
from harborrl.gateway.server import Gateway, make_server
from harborrl.gateway.trace import TraceRegistry
from harborrl.rollout.harbor_job.coordinator import GroupSlots
from harborrl.rollout.harbor_job.reliability import (
    InfraBudget,
    InfraBudgetExhausted,
    ReliabilityLedger,
    RetryableRolloutFailure,
    RetryPolicy,
    summarize_reliability,
)
from harborrl.rollout.runtime import _run_slot
from harborrl.trajectories.native import RewardProfile, read_json


@pytest.mark.parametrize('error,retryable', [
    (OSError('transport failed'), True), (ConnectionResetError(), True),
    (TimeoutError(), True), (URLError(ConnectionRefusedError()), True),
    (HTTPError('http://engine', 503, 'unavailable', {}, None), True),
    (HTTPError('http://engine', 400, 'invalid', {}, None), False),
    (HTTPError('http://engine', 401, 'unauthorized', {}, None), False),
    (URLError('unknown url type'), False), (ProtocolError('malformed output'), False),
    (ValueError('policy mismatch'), False), (PermissionError(), False),
    (OSError(errno.ENOSPC, 'no space left'), False),
])
def test_gateway_records_trusted_transport_origin(tmp_path, error, retryable):
    registry, ident, token, gateway = setup(tmp_path)
    def fail(*args):
        raise error
    gateway.backend.generate = fail
    with pytest.raises(type(error)):
        gateway.generate(token, payload())
    assert not registry.transport_failure_only(ident.attempt_id)
    registry.close(ident.attempt_id)
    _, seal = registry.seal(ident.attempt_id)
    assert seal['errors'] == [type(error).__name__]
    assert registry.transport_failure_only(ident.attempt_id) is retryable
    files = list((registry.attempts[ident.attempt_id]['root'] / 'generated').glob('*.json'))
    assert len(files) == 1
    assert (read_json(files[0]).get('failure_kind') == 'upstream-transport') is retryable


def test_exception_name_alone_does_not_authorize_recovery(tmp_path):
    registry, ident, token, _ = setup(tmp_path)
    aid, rid, *_ = registry.begin(token)
    registry.finish(aid, rid, error='OSError')
    registry.close(aid)
    registry.seal(aid)
    assert not registry.transport_failure_only(ident.attempt_id)


WORKER = '''import json, sys, time
from pathlib import Path
from urllib import request, error
req = json.loads(sys.stdin.readline())
trial = 'trial-' + req['identity']['attempt_id']
print(json.dumps({'type': 'prepared', 'trial_id': trial}), flush=True)
sys.stdin.readline()
opener = request.build_opener(request.ProxyHandler({}))
messages = []
for index in range(2):
    for retry in range(2):
        data = json.dumps({'model': 'policy', 'max_tokens': 64,
                           'messages': [{'role': 'user', 'content': 'hello'}]}).encode()
        try:
            with opener.open(request.Request(req['gateway_url'] + '/v1/messages', data,
                             {'x-api-key': req['credential']}), timeout=5) as response:
                messages.append(json.load(response))
            time.sleep(.05)  # Let the synthetic HTTP handler acknowledge delivery.
            break
        except error.HTTPError:
            if retry:
                raise
root = Path(req['output']) / 'harbor' / trial
session = root / 'agent/sessions/projects/session.jsonl'
session.parent.mkdir(parents=True)
session.write_text(''.join(json.dumps({'type': 'assistant', 'sessionId': 's',
    'version': '2.1.0', 'message': message}) + chr(10) for message in messages))
(root / 'verifier').mkdir()
(root / 'verifier/reward.json').write_text('{"reward": 0}')
(root / 'result.json').write_text(json.dumps({'id': trial, 'finished_at': 'test',
    'verifier_environment_mode': 'shared', 'verifier_result': {'rewards': {'reward': 0}}}))
print(json.dumps({'type': 'terminal', 'trial_id': trial, 'trial_dir': str(root),
                  'result_path': str(root / 'result.json')}), flush=True)
'''

@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX immutable trace publication')
@pytest.mark.parametrize('fault', ['once', 'persistent', 'protocol', 'policy', 'tokenizer', 'consumption'])
def test_real_coordinator_retries_only_proven_transport_and_records_terminal(tmp_path, monkeypatch, fault):
    from harborrl.trajectories import native

    worker = tmp_path / 'synthetic-worker.py'
    worker.write_text(WORKER)
    spawn = asyncio.create_subprocess_exec
    async def fake_spawn(*args, **kwargs):
        return await spawn(sys.executable, '-S', str(worker), **kwargs)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_spawn)
    # Permit synthetic evidence in this CPU fixture only; keep every IR check.
    require_ready = native.require_ready
    monkeypatch.setattr(native, 'require_ready', lambda ir: require_ready(ir, allow_synthetic=True))
    registry = TraceRegistry(tmp_path / 'traces')
    seen, generated = [], {}
    class FaultyBackend(Backend):
        def generate(self, converted, ident, response_id):
            aid = ident.attempt_id
            if aid not in seen:
                seen.append(aid)
                if fault == 'persistent' or len(seen) == 1:
                    raise ProtocolError('malformed output') if fault == 'protocol' else OSError('transport failed')
            generated[aid] = generated.get(aid, 0) + 1
            turn = super().generate(converted, ident, response_id)
            if fault == 'policy':
                turn['policy_version'] = 'wrong'
            if fault == 'tokenizer' and generated[aid] == 2:
                turn['tokenizer_digest'] = 'different'
            return turn
    if fault == 'consumption':
        monkeypatch.setattr(registry, 'consume', lambda *args, **kwargs: None)
    server = make_server(Gateway(registry, FaultyBackend(), model='policy'))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = _reliability_runtime_config(tmp_path)
    config.gateway_url = f'http://127.0.0.1:{server.server_port}'
    ledger = ReliabilityLedger(tmp_path / 'ledger', policy=RetryPolicy(backoff_seconds=(0,)))
    runtime = SimpleNamespace(rollout_root=tmp_path / 'attempts', semaphore=asyncio.Semaphore(1),
                              profile=PROFILE, registry=registry, reliability=ledger,
                              shards=(SimpleNamespace(url=config.gateway_url, registry=registry),))
    slots = GroupSlots([identity('0'), identity('1')], max_attempts=2)
    call = _run_slot(config, runtime, slots, 0, {'id': 'task', 'path': str(tmp_path)}, RewardProfile())
    try:
        if fault == 'once':
            result = asyncio.run(call)
            assert result['identity']['attempt_id'] == seen[1] != seen[0]
            assert result['evaluation']['training_reward'] == 0
            assert len(slots.completed) == 1
            require_ready(result, allow_synthetic=True)
        else:
            with pytest.raises(RetryableRolloutFailure if fault == 'persistent' else ValueError):
                asyncio.run(call)
            assert not slots.completed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    summary = summarize_reliability(ledger.root)
    expected = 2 if fault in {'once', 'persistent'} else 1
    assert len(seen) == summary['attempts'] == summary['terminal_attempts'] == expected
    assert summary['completed_attempts'] == (1 if fault == 'once' else 0)
    assert summary['retry_decisions'] == (1 if fault == 'once' else 2 if fault == 'persistent' else 0)
    assert all(row['status'] != 'running' for row in slots.history['0'])
    assert all(state['closed'] and state['sealed'] for state in registry.attempts.values())
    if fault in {'once', 'persistent', 'tokenizer'}:
        failed = read_json(runtime.rollout_root / 'group' / seen[0] / 'trajectory.v2.json')
        assert failed['readiness'] == 'EVAL_ONLY'
        assert failed['trace_seal']['errors']
        with pytest.raises(ValueError):
            require_ready(failed, allow_synthetic=True)


def test_rejection_of_wrong_identity_records_failure_and_no_retry(tmp_path, monkeypatch):
    from test_native_contracts import make_ir

    from harborrl.rollout import runtime as module
    from harborrl.trajectories import native

    require_ready = native.require_ready
    monkeypatch.setattr(native, 'require_ready', lambda ir: require_ready(ir, allow_synthetic=True))
    async def wrong_identity(*args, **kwargs):
        return make_ir(tmp_path, slot='1')
    monkeypatch.setattr(module, 'run_attempt', wrong_identity)
    ledger = ReliabilityLedger(tmp_path / 'ledger')
    runtime = SimpleNamespace(rollout_root=tmp_path, semaphore=asyncio.Semaphore(1),
                              profile={}, registry=None, reliability=ledger,
                              shards=(SimpleNamespace(url='http://gateway', registry=None),))
    slots = GroupSlots([identity('0'), identity('1')])
    with pytest.raises(ValueError, match='active attempt'):
        asyncio.run(_run_slot(_reliability_runtime_config(tmp_path), runtime, slots, 0,
                             {'id': 'task', 'path': '/task'}, RewardProfile()))
    summary = summarize_reliability(ledger.root)
    assert summary['attempts'] == summary['terminal_attempts'] == summary['failed_attempts'] == 1
    assert summary['retry_decisions'] == 0
    assert slots.history['0'][0]['status'] == 'invalid'


def test_failure_budget_exhaustion_still_records_cleanup(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as module

    ledger = ReliabilityLedger(tmp_path / 'ledger', budget=InfraBudget(2, 1, 1, 1, 3600))
    ledger.budget.remaining['storage_faults'] = 0
    runtime = SimpleNamespace(rollout_root=tmp_path, semaphore=asyncio.Semaphore(1),
                              profile={}, registry=None, reliability=ledger,
                              shards=(SimpleNamespace(url='http://gateway', registry=None),))
    async def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, 'no space left')
    cleaned = []
    monkeypatch.setattr(module, 'run_attempt', fail)
    monkeypatch.setattr(module, '_record_cleanup', lambda ledger, ident, root: cleaned.append(ident.attempt_id))
    slots = GroupSlots([identity('0'), identity('1')])
    with pytest.raises(InfraBudgetExhausted):
        asyncio.run(_run_slot(_reliability_runtime_config(tmp_path), runtime, slots, 0,
                             {'id': 'task', 'path': '/task'}, RewardProfile()))
    assert cleaned == ['attempt-0']
    assert summarize_reliability(ledger.root)['failed_attempts'] == 1
