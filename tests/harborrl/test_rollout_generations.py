import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from harborrl.rollout.generations import (
    ACTIVE_GENERATION,
    GenerationJournal,
    GenerationPolicy,
    generation_batch_id,
    reserve_generation_attempt,
)
from harborrl.rollout.harbor_job.reliability import (
    InfraBudgetExhausted,
    RetryableRolloutFailure,
)
from harborrl.trajectories.native import Identity, read_json

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='POSIX fsynced immutable manifests')


def result(rollout, generation, version='1'):
    return [[{'identity': {'batch_id': generation_batch_id(rollout, generation),
                          'policy_version': version, 'trajectory_id': f't-{generation}-{group}'}}]
            for group in range(2)]


def test_entire_rollout_reruns_with_fixed_inputs_and_no_partial_selection(tmp_path):
    journal = GenerationJournal(tmp_path)
    calls, discarded, partial = [], [], []
    async def run(generation):
        calls.append(dict(ACTIVE_GENERATION.get()))
        if generation == 0:
            partial.append(result(2, 0)[0])
            raise RetryableRolloutFailure('sibling exhausted attempts')
        assert not partial
        return result(2, generation)
    def discard():
        discarded.append(True)
        partial.clear()
    output = asyncio.run(journal.execute(2, {'tasks': ['a', 'b']}, '1', run, discard=discard))
    assert output == result(2, 1) and len(discarded) == 1
    assert len(calls) == 2 and calls[0]['input_sha256'] == calls[1]['input_sha256']
    root = tmp_path / 'rollout-generations/rollout-2'
    assert read_json(root / 'generation-0/superseded.json')['export_allowed'] is False
    assert read_json(root / 'selected.json')['generation'] == 1
    assert ACTIVE_GENERATION.get() is None


def test_rerun_budget_is_seed_scoped_across_batches_and_restarts(tmp_path):
    policy = GenerationPolicy(max_reruns_per_seed=1)
    async def run(generation):
        if generation == 0:
            raise RetryableRolloutFailure('failure')
        return result(0, generation)
    asyncio.run(GenerationJournal(tmp_path, policy).execute(0, {}, '1', run))
    async def failed(generation):
        raise RetryableRolloutFailure('failure')
    with pytest.raises(InfraBudgetExhausted):
        asyncio.run(GenerationJournal(tmp_path, policy).execute(1, {}, '1', failed))
    assert not (tmp_path / 'rollout-generations/rollout-1/selected.json').exists()
    with pytest.raises(ValueError, match='already exists'):
        asyncio.run(GenerationJournal(tmp_path, policy).execute(0, {}, '1', run))


@pytest.mark.parametrize('fault', ['protocol', 'cancel', 'stale-policy', 'old-generation'])
def test_protocol_cancel_and_cross_generation_errors_never_trigger_rerun(tmp_path, fault):
    calls = []
    async def run(generation):
        calls.append(generation)
        if fault == 'protocol':
            raise ValueError('task lock mismatch')
        if fault == 'cancel':
            raise asyncio.CancelledError
        return result(0, generation + (fault == 'old-generation'),
                      version='2' if fault == 'stale-policy' else '1')
    with pytest.raises(asyncio.CancelledError if fault == 'cancel' else ValueError):
        asyncio.run(GenerationJournal(tmp_path).execute(0, {}, '1', run))
    assert calls == [0]
    assert not (tmp_path / 'rollout-generations/rollout-0/selected.json').exists()
    assert (tmp_path / 'rollout-generations/rollout-0/generation-0/failed.json').exists()


def test_concurrent_reservations_and_restart_cannot_exceed_rollout_budget(tmp_path):
    owner = Identity('run', 'rollout-0-generation-0', 'group', '0', 'attempt', 'trajectory',
                     'd' * 64, '1', 'h' * 64, 'r' * 64, 's' * 64)
    def reserve(index):
        try:
            reserve_generation_attempt(tmp_path, replace(owner, attempt_id=f'attempt-{index}'),
                                       {'max_attempts': 3, 'max_failed_groups': 2})
            return True
        except RetryableRolloutFailure:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(16)))
    assert sum(outcomes) == 3
    assert reserve(17) is False


@pytest.mark.parametrize('fault', ['empty', 'duplicate', 'mutated-input', 'export-audit'])
def test_invalid_generation_is_journaled_and_never_selected(tmp_path, fault):
    inputs, discarded = {}, []
    async def run(generation):
        output = result(0, generation)
        if fault == 'empty':
            return []
        if fault == 'duplicate':
            output[1] = output[0]
        if fault == 'mutated-input':
            inputs['sampling'] = 'changed'
        return output
    def audit(_):
        if fault == 'export-audit':
            raise ValueError('training export audit rejected')
    with pytest.raises(ValueError):
        asyncio.run(GenerationJournal(tmp_path).execute(0, inputs, '1', run,
            validate=audit, discard=lambda: discarded.append(True)))
    assert discarded == [True]
    assert not (tmp_path / 'rollout-generations/rollout-0/selected.json').exists()


def test_generation_limits_cannot_increase_on_a_new_batch_or_worker_restart(tmp_path):
    GenerationJournal(tmp_path)
    with pytest.raises(ValueError, match='conflicting immutable record'):
        GenerationJournal(tmp_path, GenerationPolicy(max_reruns_per_seed=3))
    owner = Identity('run', 'rollout-0-generation-0', 'group', '0', 'attempt', 'trajectory',
                     'd' * 64, '1', 'h' * 64, 'r' * 64, 's' * 64)
    reserve_generation_attempt(tmp_path, owner, {'max_attempts': 1, 'max_failed_groups': 2})
    with pytest.raises(ValueError, match='conflicting immutable record'):
        reserve_generation_attempt(tmp_path, replace(owner, attempt_id='new'),
                                   {'max_attempts': 2, 'max_failed_groups': 2})
