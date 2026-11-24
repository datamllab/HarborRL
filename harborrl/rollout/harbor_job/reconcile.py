"""Independent, bounded Sandbox cleanup; never trust an SDK delete receipt.

The CLI runs on the runner host after the trial process exits. Only the ID
in this attempt's locked lifecycle can be deleted. History is retained and
closed by a separate immutable receipt.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import re
import secrets
import signal
import socket
import sys
import time
from pathlib import Path

import httpx

from harborrl.trajectories.native import digest, publish, read_json


def process_info(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = text[text.rindex(')') + 2:].split()
    return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]), 'pgrp': int(fields[2]),
            'session': int(fields[3]), 'start_ticks': int(fields[19])}


def register_runner(root, identity):
    """Give the independent observer a session and inherited ownership tag."""
    if os.getsid(0) != os.getpid():
        os.setsid()
    os.environ['HARBORRL_ATTEMPT_ID'] = identity['attempt_id']
    publish(Path(root) / 'runner-process.json', {
        **process_info(os.getpid()), 'host': socket.gethostname(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'attempt_id': identity['attempt_id']})


def observer_ancestors(owner):
    """Prove observer ancestry rather than ignoring protected process names."""
    ancestors, child = {}, process_info(os.getpid())
    if (child['session'] == owner['session'] or child['pid'] == owner['pid']
            or os.environ.get('HARBORRL_ATTEMPT_ID') == owner['attempt_id']):
        raise RuntimeError('process observer belongs to the runner; independent observer required')
    while child['ppid']:
        parent = process_info(child['ppid'])
        current = process_info(child['pid'])
        if (current['ppid'] != child['ppid'] or current['start_ticks'] != child['start_ticks']
                or parent['start_ticks'] > child['start_ticks'] or parent['pid'] in ancestors):
            raise RuntimeError('observer ancestry changed; process ownership uncertain')
        if parent['session'] == owner['session'] or parent['pid'] == owner['pid']:
            raise RuntimeError('process observer is a runner descendant; independent observer required')
        ancestors[parent['pid']] = parent['start_ticks']
        child = parent
    return ancestors


def owned_processes(owner):
    if owner['host'] != socket.gethostname():
        raise RuntimeError('process observer is on a different runner host')
    if owner.get('boot_id') and owner['boot_id'] != Path(
            '/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('runner host restarted; process ownership uncertain')
    try:
        leader = process_info(owner['pid'])
    except FileNotFoundError:
        leader = None
    if leader is not None and leader['start_ticks'] != owner['start_ticks']:
        raise RuntimeError('runner PID was reused; process ownership uncertain')
    ancestors = observer_ancestors(owner)
    tag = ('HARBORRL_ATTEMPT_ID=' + owner['attempt_id']).encode()
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            info = process_info(int(path.name))
            # Zombies cannot execute or hold a live agent connection.
            if info['state'] == 'Z':
                continue
            if ancestors.get(info['pid']) == info['start_ticks']:
                continue
            member = info['session'] == owner['session']
            # A process older than the runner cannot be its descendant. Do not
            # require access to protected platform services that predate it.
            # Equal ticks remain in scope: children can fork within one tick.
            if not member and info['start_ticks'] < owner['start_ticks']:
                continue
            tagged = not member and tag in (path / 'environ').read_bytes().split(b'\0')
            if member or tagged:
                found.append(info)
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Unreadable processes within the possible descendant lifetime fail
        # closed, including those that escaped the original session.
    return found


async def close_owned_processes(owner, *, deadline):
    observations = []

    async def scan(sig):
        while True:
            try:
                processes = owned_processes(owner)
            except (OSError, RuntimeError) as exc:
                observations.append({'signal': int(sig) if sig else None, 'processes': None,
                                     'error_type': type(exc).__name__,
                                     'errno': getattr(exc, 'errno', None)})
                # /proc can become unreadable during exec/exit. Retry a full
                # scan, never interpret an unreadable process as absent. PID
                # reuse, host changes and persistent read errors retain debt.
                remaining = deadline - time.monotonic()
                if isinstance(exc, RuntimeError) or remaining <= 0:
                    return None
                await asyncio.sleep(min(.1, remaining))
            else:
                observations.append({'signal': int(sig) if sig else None, 'processes': processes})
                return processes

    for sig in (None, signal.SIGTERM, signal.SIGKILL):
        processes = await scan(sig)
        if processes is None:
            return {'verified_absent': False, 'observations': observations}
        if not processes:
            return {'verified_absent': True, 'observations': observations}
        if time.monotonic() >= deadline:
            return {'verified_absent': False, 'observations': observations}
        if sig is not None:
            for process in processes:
                try:
                    current = process_info(process['pid'])
                    if current['start_ticks'] != process['start_ticks']:
                        raise RuntimeError('owned process PID changed during cleanup')
                    os.kill(process['pid'], sig)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                except (OSError, RuntimeError) as exc:
                    observations.append({'signal': int(sig), 'processes': None,
                                         'error_type': type(exc).__name__,
                                         'errno': getattr(exc, 'errno', None)})
                    return {'verified_absent': False, 'observations': observations}
        await asyncio.sleep(min(1.0, max(0, deadline - time.monotonic())))
    survivors = await scan(None)
    return {'verified_absent': survivors == [], 'observations': observations}


def processes_verified_absent(proof):
    if not isinstance(proof, dict) or proof.get('verified_absent') is not True:
        return False
    rows = proof.get('observations')
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if (not isinstance(row, dict) or 'signal' not in row
                or row['signal'] not in (None, int(signal.SIGTERM), int(signal.SIGKILL))):
            return False
        if (row.get('processes', 'missing') is None
                and isinstance(row.get('error_type'), str) and row['error_type']):
            continue
        if not isinstance(row.get('processes'), list) or 'error_type' in row:
            return False
        for process in row['processes']:
            if not isinstance(process, dict) or any(
                    type(process.get(key)) is not int or process[key] < minimum
                    for key, minimum in (('pid', 1), ('session', 1), ('start_ticks', 0))):
                return False
    return rows[-1]['processes'] == []


def inventory_verified_absent(receipt):
    sid, ids, rows = receipt.get('sandbox_id'), receipt.get('inventory_ids'), receipt['observations']
    if (not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids) or sid in ids):
        return False
    get_index = next((index for index in reversed(range(len(rows)))
                      if rows[index].get('operation') == 'get'
                      and rows[index].get('http_status') == 404), None)
    if get_index is None:
        return False
    pages = rows[get_index + 1:]
    observed, total = set(), None
    if not pages:
        return False
    for index, row in enumerate(pages, 1):
        page_ids, count = row.get('ids'), row.get('total_items')
        if (row.get('operation') != 'inventory' or row.get('http_status') != 200
                or type(row.get('page')) is not int or row['page'] != index
                or type(count) is not int or count < 0
                or type(row.get('has_next_page')) is not bool
                or row['has_next_page'] != (index < len(pages))
                or not isinstance(page_ids, list)
                or any(not isinstance(value, str) or not value for value in page_ids)
                or len(set(page_ids)) != len(page_ids) or observed.intersection(page_ids)
                or (index < len(pages) and not page_ids)
                or (total is not None and total != count)):
            return False
        total = count
        observed.update(page_ids)
    return len(observed) == total and observed == set(ids)


async def inventory(client, base, params, observations, *, max_pages=100):
    """Require complete, consistent pagination, never an empty first page alone."""
    ids, total = set(), None
    for page in range(1, max_pages + 1):
        response = await client.get(base + '/sandboxes',
                                    params={**params, 'page': page, 'pageSize': 100})
        row = {'operation': 'inventory', 'page': page, 'http_status': response.status_code}
        observations.append(row)
        if response.status_code != 200:
            return None
        body = response.json()
        items, pagination = body.get('items'), body.get('pagination')
        if not isinstance(items, list) or not isinstance(pagination, dict):
            raise TypeError('incomplete inventory schema')
        more, count = pagination.get('hasNextPage'), pagination.get('totalItems')
        if type(more) is not bool or type(count) is not int or count < 0:
            raise ValueError('invalid inventory pagination')
        if total is not None and count != total:
            return None  # Inventory changed while paging; restart a full observation.
        total = count
        page_ids = [item.get('id') for item in items if isinstance(item, dict)]
        if (len(page_ids) != len(items) or any(not isinstance(sid, str) or not sid for sid in page_ids)
                or len(set(page_ids)) != len(page_ids) or ids.intersection(page_ids)):
            raise ValueError('invalid or repeated inventory entries')
        ids.update(page_ids)
        row.update(ids=sorted(page_ids), has_next_page=more, total_items=total)
        if not more:
            if len(ids) != total:
                raise ValueError('inventory total does not match complete pages')
            return sorted(ids)
        if not items:
            raise ValueError('empty intermediate inventory page')
    raise ValueError('inventory page budget exhausted')


async def reconcile_instance(cfg, lifecycle, *, params, credential, process_proof,
                             transport=None, retry_interval=2.0):
    started = time.monotonic()
    deadline = started + cfg['cleanup_timeout_s']
    sid = lifecycle.get('sandbox_id')
    receipt = {'schema': 'harborrl-cleanup-v1', 'runtime_digest': digest(cfg),
               'sandbox_id': sid, 'status': 'unresolved', 'observations': [],
               'process_proof': process_proof, 'lifecycle_sha256': digest(lifecycle)}
    observations = receipt['observations']
    if lifecycle.get('runtime_digest') != digest(cfg):
        receipt['reason'] = 'runtime lock mismatch'
    elif not processes_verified_absent(process_proof):
        receipt['reason'] = 'owned processes remain or ownership unverified'
    elif not sid and lifecycle.get('state') in {'pending', 'waiting', 'reserved'}:
        receipt.update(status='not_allocated', reason='creation never started')
    elif not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sid):
        receipt['reason'] = 'unknown allocation; automatic deletion forbidden'
    else:
        base = cfg['domain'].rstrip('/') + '/v1'
        try:
            async with asyncio.timeout(cfg['cleanup_timeout_s']):
                async with httpx.AsyncClient(
                        timeout=min(cfg['request_timeout_s'], cfg['cleanup_timeout_s']),
                        headers={'OPEN-SANDBOX-API-KEY': credential}, transport=transport) as client:
                    while True:
                        try:
                            response = await client.get(base + '/sandboxes/' + sid, params=params)
                            observations.append({'operation': 'get', 'http_status': response.status_code})
                            if response.status_code == 404:
                                ids = await inventory(client, base, params, observations)
                                if ids is not None and sid not in ids:
                                    receipt.update(status='verified_absent', inventory_ids=ids,
                                                   inventory_complete=True, get404=True,
                                                   reconciled_lifecycle='deleted')
                                    break
                            elif response.status_code == 200:
                                deleted = await client.delete(base + '/sandboxes/' + sid, params=params)
                                observations.append({'operation': 'delete', 'http_status': deleted.status_code})
                            elif response.status_code in {401, 403}:
                                receipt['reason'] = 'cleanup authorization rejected'
                                break
                        except (httpx.HTTPError, ValueError, TypeError) as exc:
                            observations.append({'operation': 'observe', 'error_type': type(exc).__name__})
                        await asyncio.sleep(min(retry_interval, max(0, deadline - time.monotonic())))
        except TimeoutError:
            receipt['reason'] = 'independent cleanup deadline exceeded'
    receipt['elapsed_sec'] = time.monotonic() - started
    return receipt


def cleanup_resolved(root):
    root = Path(root)
    receipt_path = root / 'cleanup-reconciliation.json'
    if not receipt_path.is_file():
        return False
    try:
        receipt = read_json(receipt_path)
        lifecycle = read_json(root / 'runner/sandbox-lifecycle.json')
        owner = read_json(root / 'runner-process.json')
        if not all(isinstance(item, dict) for item in (receipt, lifecycle, owner)):
            return False
        proof, observations = receipt.get('process_proof'), receipt.get('observations')
        if (not isinstance(proof, dict) or not isinstance(observations, list)
                or not all(isinstance(item, dict) for item in observations)):
            return False
        common = (receipt.get('schema') == 'harborrl-cleanup-v1'
                  and receipt.get('runtime_digest') == lifecycle.get('runtime_digest')
                  and receipt.get('lifecycle_sha256') == digest(lifecycle)
                  and receipt.get('owner_sha256') == digest(owner)
                  and processes_verified_absent(proof)
                  and type(receipt.get('elapsed_sec')) in {float, int}
                  and math.isfinite(receipt['elapsed_sec'])
                  and receipt['elapsed_sec'] >= 0)
        if not common:
            return False
        if receipt.get('status') == 'not_allocated':
            return (not lifecycle.get('sandbox_id')
                    and lifecycle.get('state') in {'pending', 'waiting', 'reserved'})
        return (receipt.get('status') == 'verified_absent'
                and isinstance(lifecycle.get('sandbox_id'), str)
                and receipt.get('sandbox_id') == lifecycle['sandbox_id']
                and receipt.get('get404') is True
                and receipt.get('inventory_complete') is True
                and receipt.get('reconciled_lifecycle') == 'deleted'
                and inventory_verified_absent(receipt))
    except (OSError, ValueError, TypeError):
        return False


def cleanup_debt(root):
    root = Path(root)
    return (any((root / path).exists() for path in
                ('cleanup-required.json', 'runner/cleanup-required.json',
                 'runner/sandbox-lifecycle.json', 'runner-process.json'))
            and not cleanup_resolved(root))

async def reconcile_attempt(cfg, root):
    from harborrl.rollout.harbor_job.sandbox_config import (
        environment_class,
        read_credential,
    )

    root = Path(root)
    if cleanup_resolved(root):
        return read_json(root / 'cleanup-reconciliation.json')
    lifecycle = read_json(root / 'runner/sandbox-lifecycle.json')
    owner = read_json(root / 'runner-process.json')
    proof = await close_owned_processes(owner, deadline=time.monotonic() + cfg['cleanup_timeout_s'])
    cls = environment_class(cfg)
    receipt = await reconcile_instance(cfg, lifecycle, params=cls.cleanup_api_params(cfg),
                                       credential=read_credential(cfg), process_proof=proof)
    receipt['owner_sha256'] = digest(owner)
    publish(root / 'cleanup-reconciliations' / f'{secrets.token_hex(16)}.json', receipt)
    if receipt['status'] in {'verified_absent', 'not_allocated'}:
        publish(root / 'cleanup-reconciliation.json', receipt)
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()
    cfg = __import__('json').loads(sys.stdin.readline())
    receipt = asyncio.run(reconcile_attempt(cfg, args.root))
    if receipt['status'] not in {'verified_absent', 'not_allocated'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
