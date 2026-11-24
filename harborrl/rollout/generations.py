"""Bounded whole-batch recovery with immutable, reward-independent inputs."""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from pathlib import Path

from harborrl.rollout.harbor_job.reliability import (
    InfraBudgetExhausted,
    RetryableRolloutFailure,
)
from harborrl.trajectories.native import digest, publish, read_json

ACTIVE_GENERATION = contextvars.ContextVar('harborrl_rollout_generation', default=None)


@dataclass(frozen=True)
class GenerationPolicy:
    max_reruns_per_seed: int = 2
    max_attempts_per_rollout: int = 32
    max_failed_groups_per_rollout: int = 8

    def __post_init__(self):
        for key, minimum in (('max_reruns_per_seed', 0), ('max_attempts_per_rollout', 1),
                             ('max_failed_groups_per_rollout', 1)):
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')


def generation_batch_id(rollout_id, generation):
    return f'rollout-{rollout_id}-generation-{generation}'


def generation_group_id(rollout_id, generation, group_index):
    return f'{generation_batch_id(rollout_id, generation)}-group-{group_index}'


def reserve_generation_attempt(run_root, identity, context):
    """A shared, restart-stable bound for embedded and service group workers."""
    import fcntl

    root = Path(run_root) / 'generation-budgets' / identity.batch_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish(root / 'limits.json', {'max_attempts': context['max_attempts'],
                                       'max_failed_groups': context['max_failed_groups']})
        if (len(list((root / 'attempts').glob('*.json'))) >= context['max_attempts']
                or len(list((root / 'failed-groups').glob('*.json'))) > context['max_failed_groups']):
            raise RetryableRolloutFailure('rollout infrastructure budget exhausted')
        publish(root / 'attempts' / f'{identity.attempt_id}.json', identity.to_dict())


def record_failed_group(run_root, identity):
    root = Path(run_root) / 'generation-budgets' / identity.batch_id / 'failed-groups'
    publish(root / f'{identity.group_id}.json', {'group_id': identity.group_id})


class GenerationJournal:
    def __init__(self, run_root, policy=None):
        self.root = Path(run_root) / 'rollout-generations'
        self.policy = policy or GenerationPolicy()
        self.root.mkdir(parents=True, exist_ok=True)
        publish(self.root / 'policy.json', self.policy.__dict__)

    def reserve_rerun(self, rollout_id, generation):
        # Persist at seed scope across batches and process restarts. A conflict
        # never silently buys another retry or resumes a superseded protocol.
        import fcntl

        with (self.root / 'budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            used = len(list((self.root / 'rerun-budget').glob('*.json')))
            if used >= self.policy.max_reruns_per_seed:
                raise InfraBudgetExhausted('rollout generation rerun budget exhausted')
            publish(self.root / 'rerun-budget' / f'{used:04d}.json',
                    {'rollout_id': rollout_id, 'generation': generation})

    async def execute(self, rollout_id, inputs, policy_version, run, *, discard=None, validate=None):
        fixed = {'rollout_id': rollout_id, 'inputs': inputs, 'policy_version': policy_version,
                 'policy': self.policy.__dict__}
        fingerprint = digest(fixed)
        batch_root = self.root / f'rollout-{rollout_id}'
        # Existing attempts must be inspected under a fresh run; never replay a
        # frozen generation implicitly after a process restart.
        try:
            batch_root.mkdir()
        except FileExistsError as exc:
            raise ValueError('rollout generation already exists; use a fresh run protocol') from exc
        publish(batch_root / 'inputs.json', fixed)
        generation = 0
        while True:
            directory = batch_root / f'generation-{generation}'
            context = {'rollout_id': rollout_id, 'generation': generation,
                       'policy_version': policy_version, 'input_sha256': fingerprint,
                       'max_attempts': self.policy.max_attempts_per_rollout,
                       'max_failed_groups': self.policy.max_failed_groups_per_rollout}
            publish(directory / 'reserved.json', context)
            token = ACTIVE_GENERATION.set(context)
            try:
                result = await run(generation)
                if digest(fixed) != fingerprint or digest(read_json(batch_root / 'inputs.json')) != fingerprint:
                    raise ValueError('frozen rollout inputs changed during recovery')
                ids = self.validate_result(result, inputs, rollout_id, generation, policy_version)
                if validate is not None:
                    validate(result)
            except RetryableRolloutFailure as exc:
                publish(directory / 'failed.json', {'error_type': type(exc).__name__,
                                                    'input_sha256': fingerprint})
                if discard is not None:
                    discard()
                try:
                    self.reserve_rerun(rollout_id, generation + 1)
                except InfraBudgetExhausted:
                    publish(batch_root / 'failed-seed.json', {'generation': generation,
                                                            'reason': 'rerun budget exhausted'})
                    raise
                publish(directory / 'superseded.json', {'next_generation': generation + 1,
                                                        'export_allowed': False})
                generation += 1
            except BaseException as exc:
                if discard is not None:
                    discard()
                publish(directory / 'failed.json', {'error_type': type(exc).__name__,
                                                    'input_sha256': fingerprint})
                raise
            else:
                publish(batch_root / 'selected.json', {'generation': generation,
                    'input_sha256': fingerprint, 'identities': ids, 'result_sha256': digest(result)})
                return result
            finally:
                ACTIVE_GENERATION.reset(token)

    @staticmethod
    def validate_result(result, inputs, rollout_id, generation, policy_version):
        if not isinstance(result, list) or not result or any(
                not isinstance(group, list) or not group for group in result):
            raise ValueError('rollout result is empty or incomplete')
        ids = [[ir['identity'] for ir in group] for group in result]
        if any(identity['policy_version'] != policy_version or
               identity['batch_id'] != generation_batch_id(rollout_id, generation)
               for group in ids for identity in group):
            raise ValueError('rollout result escaped its generation or policy lock')
        trajectory_ids = [identity['trajectory_id'] for group in ids for identity in group]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError('duplicate trajectory in rollout generation')
        groups = inputs.get('groups')
        if groups is not None:
            if len(groups) != len(ids) or len({g['group_index'] for g in groups}) != len(groups):
                raise ValueError('rollout group membership changed')
            for expected, actual in zip(groups, ids, strict=True):
                if len(expected['slots']) != len(actual) or len(expected['tasks']) != len(actual):
                    raise ValueError('rollout slot membership changed')
                for index, identity in enumerate(actual):
                    if (identity['group_id'] != generation_group_id(
                            rollout_id, generation, expected['group_index'])
                            or identity['slot_id'] != expected['slots'][index]
                            or identity['task_digest'] != expected['tasks'][index]['task_digest']):
                        raise ValueError('rollout task or slot order changed')
        return ids
