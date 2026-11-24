import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from harborrl.rollout.harbor_job.reconcile import (
    cleanup_debt,
    cleanup_resolved,
    close_owned_processes,
    owned_processes,
    process_info,
    processes_verified_absent,
    reconcile_instance,
)
from harborrl.rollout.harbor_job.sandbox_config import SandboxAdmission
from harborrl.trajectories.native import digest, publish


def config():
    return {'domain': 'https://sandbox.test', 'cleanup_timeout_s': .08, 'request_timeout_s': .03}


def process_proof():
    return {'verified_absent': True, 'observations': [{'signal': None, 'processes': []}]}


def run_reconciliation(handler, *, lifecycle=None, proof=None):
    cfg = config()
    lifecycle = lifecycle or {'runtime_digest': digest(cfg), 'sandbox_id': 'owned', 'state': 'deleted'}
    return asyncio.run(reconcile_instance(cfg, lifecycle, params={'project': 'test'},
        credential='test-secret', process_proof=proof if proof is not None else process_proof(),
        transport=httpx.MockTransport(handler), retry_interval=.001))


def test_delete_acknowledgement_is_not_independent_absence():
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(204 if request.method == 'DELETE' else 200)
    result = run_reconciliation(handler)
    assert result['status'] == 'unresolved'
    assert 'DELETE' in calls and not result.get('get404')
    assert 'test-secret' not in json.dumps(result)


def test_cleanup_retries_delete_and_complete_inventory_with_owned_id_only():
    calls, gets, inventories = [], 0, 0
    def handler(request):
        nonlocal gets, inventories
        assert request.url.params['project'] == 'test'
        calls.append((request.method, request.url.path))
        if request.url.path.endswith('/owned'):
            if request.method == 'DELETE':
                return httpx.Response(503 if gets == 1 else 204)
            gets += 1
            return httpx.Response(200 if gets <= 2 else 404)
        inventories += 1
        if inventories == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={'items': [{'id': 'someone-else'}],
            'pagination': {'hasNextPage': False, 'totalItems': 1}})
    result = run_reconciliation(handler)
    assert result['status'] == 'verified_absent'
    assert result['inventory_ids'] == ['someone-else']
    assert result['inventory_complete'] and result['get404']
    assert all(path.endswith('/owned') for method, path in calls if method == 'DELETE')


@pytest.mark.parametrize('kind', ['stale', 'missing-pagination', 'lying-total', 'repeated-page'])
def test_stale_or_incomplete_inventory_never_releases_capacity(kind):
    def handler(request):
        if request.url.path.endswith('/owned'):
            return httpx.Response(404)
        data = {'items': [{'id': 'owned'}], 'pagination': {'hasNextPage': False, 'totalItems': 1}}
        if kind == 'missing-pagination':
            data.pop('pagination')
        elif kind == 'lying-total':
            data['pagination']['totalItems'] = 5
        elif kind == 'repeated-page':
            data['pagination'].update(hasNextPage=True, totalItems=2)
        return httpx.Response(200, json=data)
    assert run_reconciliation(handler)['status'] == 'unresolved'


def test_complete_paginated_inventory_is_required():
    pages = []
    def handler(request):
        if request.url.path.endswith('/owned'):
            return httpx.Response(404)
        page = int(request.url.params['page'])
        pages.append(page)
        return httpx.Response(200, json={'items': [{'id': f'other-{page}'}],
            'pagination': {'hasNextPage': page == 1, 'totalItems': 2}})
    result = run_reconciliation(handler)
    assert result['status'] == 'verified_absent' and pages == [1, 2]
    assert result['inventory_ids'] == ['other-1', 'other-2']


def test_unknown_allocation_and_surviving_processes_fail_closed():
    def handler(request):
        pytest.fail('must not query or delete unowned allocations')
    cfg = config()
    result = run_reconciliation(handler, lifecycle={
        'runtime_digest': digest(cfg), 'state': 'creation_unknown', 'sandbox_id': None})
    assert result['status'] == 'unresolved'
    result = run_reconciliation(handler, proof={'verified_absent': False})
    assert result['status'] == 'unresolved'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process/session observer')
def test_observer_closes_owned_runner_and_child_without_killing_other_processes():
    code = "import os,subprocess,time; child=subprocess.Popen(['sleep','60']); print(child.pid,flush=True); time.sleep(60)"
    process = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
        env={**os.environ, 'HARBORRL_ATTEMPT_ID': 'cleanup-test'}, stdout=subprocess.PIPE, text=True)
    child = int(process.stdout.readline())
    try:
        owner = {**process_info(process.pid), 'host': socket.gethostname(), 'attempt_id': 'cleanup-test'}
        assert {process.pid, child} <= {row['pid'] for row in owned_processes(owner)}
        result = asyncio.run(close_owned_processes(owner, deadline=__import__('time').monotonic() + 5))
        assert result['verified_absent']
        assert os.getpid() not in {row['pid'] for obs in result['observations'] for row in obs['processes']}
    finally:
        if process.poll() is None:
            os.killpg(process.pid, __import__('signal').SIGKILL)
        process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process/session observer')
def test_observer_finds_escaped_tagged_child_and_preserves_unrelated_session():
    code = ("import subprocess,time; child=subprocess.Popen(['sleep','60'],start_new_session=True); "
            "print(child.pid,flush=True); time.sleep(60)")
    runner = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
        env={**os.environ, 'HARBORRL_ATTEMPT_ID': 'escaped-cleanup-test'},
        stdout=subprocess.PIPE, text=True)
    other = subprocess.Popen(['sleep', '60'], start_new_session=True,
        env={**os.environ, 'HARBORRL_ATTEMPT_ID': 'another-attempt'})
    child = int(runner.stdout.readline())
    try:
        owner = {**process_info(runner.pid), 'host': socket.gethostname(),
                 'attempt_id': 'escaped-cleanup-test'}
        assert process_info(child)['session'] != owner['session']
        assert {runner.pid, child} == {row['pid'] for row in owned_processes(owner)}
        proof = asyncio.run(close_owned_processes(owner, deadline=__import__('time').monotonic() + 5))
        assert processes_verified_absent(proof) and other.poll() is None
    finally:
        for pid in (child, runner.pid, other.pid):
            try:
                os.kill(pid, __import__('signal').SIGKILL)
            except ProcessLookupError:
                pass
        runner.wait(timeout=5)
        other.wait(timeout=5)


@pytest.mark.skipif(sys.platform == 'win32', reason='fsynced immutable receipts require POSIX directory fd')
def test_admission_retains_debt_until_independent_receipt_and_preserves_marker(tmp_path):
    binding = {'image': 'test', 'max_live_instances': 1}
    admission = SandboxAdmission({'max_live_instances': 1, 'bindings': [binding]})
    publish(tmp_path / 'cleanup-required.json', {'reason': 'old delete failure'})
    async def scenario():
        async with admission.lease(binding, tmp_path):
            pass
        assert admission.blocked and admission.total.locked()
        with pytest.raises(RuntimeError, match='blocked'):
            async with admission.lease(binding, tmp_path / 'new'):
                pytest.fail('new create admitted with debt')
    asyncio.run(scenario())
    cfg = config()
    lifecycle = {'state': 'delete_failed', 'runtime_digest': digest(cfg), 'sandbox_id': 'owned'}
    owner = {'attempt_id': 'a'}
    publish(tmp_path / 'runner/sandbox-lifecycle.json', lifecycle)
    publish(tmp_path / 'runner-process.json', owner)
    publish(tmp_path / 'cleanup-reconciliation.json', {'status': 'verified_absent',
        'schema': 'harborrl-cleanup-v1', 'runtime_digest': digest(cfg), 'sandbox_id': 'owned',
        'get404': True, 'inventory_complete': True, 'inventory_ids': [],
        'reconciled_lifecycle': 'deleted', 'elapsed_sec': .1,
        'observations': [{'operation': 'get', 'http_status': 404},
                         {'operation': 'inventory', 'page': 1, 'ids': [], 'total_items': 0,
                          'has_next_page': False, 'http_status': 200}],
        'lifecycle_sha256': digest(lifecycle), 'owner_sha256': digest(owner),
        'process_proof': process_proof()})
    admission.reconcile(tmp_path)
    assert not admission.blocked and not admission.total.locked()
    assert (tmp_path / 'cleanup-required.json').is_file() and not cleanup_debt(tmp_path)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
def test_protected_preexisting_process_is_excluded_but_possible_escaped_child_fails_closed(monkeypatch):
    from harborrl.rollout.harbor_job import reconcile

    owner = {'host': socket.gethostname(), 'pid': 71, 'session': 71,
             'start_ticks': 100, 'attempt_id': 'test'}
    monkeypatch.setattr(Path, 'iterdir', lambda _: iter([Path('/proc/72')]))
    def unreadable(_):
        raise PermissionError('protected process')
    monkeypatch.setattr(Path, 'read_bytes', unreadable)
    monkeypatch.setattr(reconcile, 'observer_ancestors', lambda _: {})
    def info(pid):
        return {'pid': pid, 'session': pid, 'start_ticks': 100 if pid == 71 else 99, 'state': 'S'}
    monkeypatch.setattr(reconcile, 'process_info', info)
    assert owned_processes(owner) == []
    monkeypatch.setattr(reconcile, 'process_info', lambda pid: {
        'pid': pid, 'session': pid, 'start_ticks': 100, 'state': 'S'})
    with pytest.raises(PermissionError):
        owned_processes(owner)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
def test_protected_observer_ancestor_is_excluded_by_pid_and_start_ticks(monkeypatch):
    from harborrl.rollout.harbor_job import reconcile

    owner = {'host': socket.gethostname(), 'pid': 71, 'session': 71,
             'start_ticks': 100, 'attempt_id': 'test'}
    monkeypatch.setattr(Path, 'iterdir', lambda _: iter([Path('/proc/72')]))
    monkeypatch.setattr(reconcile, 'observer_ancestors', lambda _: {72: 101})
    monkeypatch.setattr(reconcile, 'process_info', lambda pid: {
        'pid': pid, 'session': pid, 'start_ticks': 100 if pid == 71 else 101, 'state': 'S'})
    def unreadable(_):
        raise PermissionError('protected observer ancestor')
    monkeypatch.setattr(Path, 'read_bytes', unreadable)
    assert owned_processes(owner) == []
    # The same PID with a new start tick cannot inherit that exclusion.
    monkeypatch.setattr(reconcile, 'observer_ancestors', lambda _: {72: 99})
    with pytest.raises(PermissionError):
        owned_processes(owner)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
@pytest.mark.parametrize('fault', ['runner-ancestor', 'changed-parent'])
def test_untrusted_observer_ancestry_fails_closed(monkeypatch, fault):
    from harborrl.rollout.harbor_job import reconcile

    observer = os.getpid()
    calls = 0
    def info(pid):
        nonlocal calls
        calls += 1
        if pid == observer:
            return {'pid': observer, 'ppid': 71 if calls == 1 or fault != 'changed-parent' else 72,
                    'session': observer, 'start_ticks': 200}
        return {'pid': 71, 'ppid': 1, 'session': 71, 'start_ticks': 100}
    monkeypatch.setattr(reconcile, 'process_info', info)
    with pytest.raises(RuntimeError, match='observer'):
        reconcile.observer_ancestors({'pid': 71, 'session': 71, 'start_ticks': 100, 'attempt_id': 'test'})


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
@pytest.mark.parametrize('kind', ['same-session', 'inherited-tag'])
def test_reparented_observer_cannot_certify_its_own_runner_absent(monkeypatch, kind):
    from harborrl.rollout.harbor_job import reconcile

    owner = {'pid': 71, 'session': 71, 'start_ticks': 100, 'attempt_id': 'owned-attempt'}
    observer = os.getpid()
    monkeypatch.setattr(reconcile, 'process_info', lambda _: {
        'pid': observer, 'ppid': 0, 'session': 71 if kind == 'same-session' else observer,
        'start_ticks': 101})
    monkeypatch.setenv('HARBORRL_ATTEMPT_ID', 'owned-attempt' if kind == 'inherited-tag' else 'other')
    with pytest.raises(RuntimeError, match='independent observer required'):
        reconcile.observer_ancestors(owner)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
def test_transient_unreadable_process_requires_a_new_complete_scan(monkeypatch):
    from harborrl.rollout.harbor_job import reconcile

    scans = 0
    def scan(_):
        nonlocal scans
        scans += 1
        if scans == 1:
            raise PermissionError(13, 'process changed during observation')
        return []
    monkeypatch.setattr(reconcile, 'owned_processes', scan)
    result = asyncio.run(close_owned_processes({}, deadline=__import__('time').monotonic() + 1))
    assert scans == 2 and result['verified_absent']
    assert result['observations'][0]['error_type'] == 'PermissionError'
    assert result['observations'][0]['processes'] is None
    assert result['observations'][-1]['processes'] == []
    assert processes_verified_absent(result)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
def test_persistent_unreadable_process_exhausts_deadline_without_signalling(monkeypatch):
    from harborrl.rollout.harbor_job import reconcile

    def scan(_):
        raise PermissionError(13, 'process remains protected')
    monkeypatch.setattr(reconcile, 'owned_processes', scan)
    monkeypatch.setattr(os, 'kill', lambda *_: pytest.fail('unverified process must not be killed'))
    result = asyncio.run(close_owned_processes({}, deadline=__import__('time').monotonic() + .02))
    assert not result['verified_absent'] and len(result['observations']) >= 2
    assert all(row['processes'] is None for row in result['observations'])
    assert not processes_verified_absent({**result, 'verified_absent': True})


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
def test_pid_reuse_is_terminal_and_retains_an_unresolved_process_receipt(monkeypatch):
    from harborrl.rollout.harbor_job import reconcile

    def scan(_):
        raise RuntimeError('runner PID was reused; process ownership uncertain')
    monkeypatch.setattr(reconcile, 'owned_processes', scan)
    result = asyncio.run(close_owned_processes({}, deadline=__import__('time').monotonic() + 5))
    assert not result['verified_absent'] and len(result['observations']) == 1
    assert result['observations'][0]['error_type'] == 'RuntimeError'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process observer')
def test_signal_permission_error_retains_debt_and_is_not_an_empty_scan(monkeypatch):
    from harborrl.rollout.harbor_job import reconcile

    process = {'pid': 123, 'session': 123, 'start_ticks': 100}
    monkeypatch.setattr(reconcile, 'owned_processes', lambda _: [process])
    monkeypatch.setattr(reconcile, 'process_info', lambda _: process)
    async def no_wait(_):
        pass
    monkeypatch.setattr(asyncio, 'sleep', no_wait)
    def denied(*_):
        raise PermissionError(1, 'signal forbidden')
    monkeypatch.setattr(os, 'kill', denied)
    result = asyncio.run(close_owned_processes({}, deadline=__import__('time').monotonic() + 5))
    assert not result['verified_absent']
    assert result['observations'][-1]['error_type'] == 'PermissionError'
    assert result['observations'][-1]['processes'] is None
    assert not processes_verified_absent({**result, 'verified_absent': True})


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX fsynced evidence')
def test_waiting_admission_wakes_and_fails_when_cleanup_retains_all_capacity(tmp_path):
    binding = {'image': 'test', 'max_live_instances': 1}
    admission = SandboxAdmission({'max_live_instances': 1, 'bindings': [binding]})
    async def scenario():
        async with admission.lease(binding, tmp_path / 'first'):
            async def waiting():
                async with admission.lease(binding, tmp_path / 'second'):
                    pytest.fail('queued create bypassed cleanup debt')
            task = asyncio.create_task(waiting())
            await asyncio.sleep(.01)
            assert not task.done()
            publish(tmp_path / 'first/cleanup-required.json', {'reason': 'injected delete failure'})
        with pytest.raises(RuntimeError, match='blocked'):
            await asyncio.wait_for(task, .5)
        assert admission.total.locked() and admission.bindings['test'].locked()
    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX fsynced evidence')
def test_cleanup_status_and_sdk_receipt_alone_do_not_resolve_debt(tmp_path):
    lifecycle, owner = {'state': 'deleted', 'sandbox_id': 'owned'}, {'attempt_id': 'a'}
    publish(tmp_path / 'runner/sandbox-lifecycle.json', lifecycle)
    publish(tmp_path / 'runner-process.json', owner)
    publish(tmp_path / 'cleanup-reconciliation.json', {'status': 'verified_absent',
        'lifecycle_sha256': digest(lifecycle), 'owner_sha256': digest(owner),
        'process_proof': {'verified_absent': True}})
    assert not cleanup_resolved(tmp_path)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX fsynced evidence')
@pytest.mark.parametrize('fault', ['missing-process-scan', 'surviving-process', 'missing-page',
                                 'inventory-total', 'get404-after-inventory'])
def test_malformed_independent_proof_does_not_release_capacity(tmp_path, fault):
    cfg = config()
    lifecycle = {'runtime_digest': digest(cfg), 'sandbox_id': 'owned', 'state': 'deleted'}
    def handler(request):
        if request.url.path.endswith('/owned'):
            return httpx.Response(404)
        return httpx.Response(200, json={'items': [],
                              'pagination': {'hasNextPage': False, 'totalItems': 0}})
    receipt = run_reconciliation(handler, lifecycle=lifecycle)
    owner = {'attempt_id': 'a'}
    receipt['owner_sha256'] = digest(owner)
    if fault == 'missing-process-scan':
        receipt['process_proof'].pop('observations')
    elif fault == 'surviving-process':
        receipt['process_proof']['observations'][-1]['processes'] = [
            {'pid': 123, 'session': 123, 'start_ticks': 100}]
    elif fault == 'missing-page':
        receipt['observations'][-1]['page'] = 2
    elif fault == 'inventory-total':
        receipt['observations'][-1]['total_items'] = 1
    else:
        receipt['observations'].reverse()
    publish(tmp_path / 'runner/sandbox-lifecycle.json', lifecycle)
    publish(tmp_path / 'runner-process.json', owner)
    publish(tmp_path / 'cleanup-reconciliation.json', receipt)
    assert cleanup_debt(tmp_path) and not cleanup_resolved(tmp_path)
