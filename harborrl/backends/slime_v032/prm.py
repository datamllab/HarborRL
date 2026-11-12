"""Process reward transport through Slime's standard per-sample rewards field."""
from __future__ import annotations

import math
import os

from harborrl.math import prm
from harborrl.trajectories.native import digest, finite


def reward_payload(unit, group, sample_index):
    span = unit.turns[0]
    scores = span.process_rewards
    return {
        'kind': prm.ADAPTER, 'identity': scores.identity,
        'tokens_digest': digest(list(span.tokens)), 'sample_index': sample_index,
        'rollout_id': unit.rollout_id, 'trajectory_id': unit.trajectory_id,
        'group_key': group.group_key, 'group_size': group.group_size,
        'group_totals': {member.trajectory_id: member.reward for member in group.trajectories},
        'step_ends': list(scores.step_ends), 'probabilities': list(scores.probabilities),
        'token_returns': list(scores.token_returns), 'baseline': group.reward_mean,
    }


def compute_advantages_and_returns(args, rollout_data):
    import torch

    frozen = prm.load(os.environ['HARBORRL_PROCESS_REWARD'])
    fields = ('rewards', 'tokens', 'response_lengths', 'sample_indices', 'rollout_ids', 'loss_masks',
              'rollout_log_probs', 'rollout_mask_sums')
    if (any(key not in rollout_data for key in fields) or not rollout_data['rewards']
            or len({len(rollout_data[key]) for key in fields}) != 1
            or rollout_data.get('teacher_log_probs') is not None):
        raise ValueError('PRM requires complete aligned sample fields without teacher scores')
    advantages, returns, summaries = [], [], []
    group_records = {}
    for index, payload in enumerate(rollout_data['rewards']):
        if not isinstance(payload, dict) or payload.get('kind') != prm.ADAPTER:
            raise ValueError('PRM requires structured process rewards before advantage computation')
        tokens = rollout_data['tokens'][index]
        tokens = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        length = rollout_data['response_lengths'][index]
        old = rollout_data['rollout_log_probs'][index]
        mask = rollout_data['loss_masks'][index]
        totals = payload.get('group_totals', {})
        if (payload.get('identity') != frozen['identity'] or payload.get('tokens_digest') != digest(tokens)
                or payload.get('sample_index') != rollout_data['sample_indices'][index]
                or payload.get('rollout_id') != rollout_data['rollout_ids'][index]
                or payload.get('group_size') != args.n_samples_per_prompt
                or not isinstance(totals, dict) or len(totals) != args.n_samples_per_prompt
                or payload.get('trajectory_id') not in totals or any(not finite(value) for value in totals.values())):
            raise ValueError('PRM identity, sample binding or complete-group baseline changed')
        values = prm.token_returns(payload.get('step_ends', []), payload.get('probabilities', []), length)
        baseline = math.fsum(totals.values()) / len(totals)
        if (payload.get('token_returns') != values or payload.get('baseline') != baseline
                or not math.isclose(totals[payload['trajectory_id']], values[0], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError('PRM reward-to-go or complete-group baseline changed')
        key = payload.get('group_key')
        if not isinstance(key, str) or not key:
            raise ValueError('PRM group key must be nonempty')
        record = (payload['rollout_id'], totals)
        if key in group_records and group_records[key] != record:
            raise ValueError('PRM local group members disagree on their complete group')
        group_records[key] = record
        if (old.ndim != 1 or mask.shape != old.shape or old.numel() != length
                or not torch.isfinite(old).all() or not torch.all(mask == 1)
                or rollout_data['rollout_mask_sums'][index] != length):
            raise ValueError('PRM requires every exact response token and a matching mask')
        target = torch.tensor(values, dtype=torch.float32, device=old.device)
        returns.append(target)
        advantages.append((target - baseline).detach())
        summaries.append(values[0] - baseline)
    rollout_data['advantages'] = advantages
    rollout_data['returns'] = returns
    # Slime's subsequent metric logger expects scalar rewards, not receipts.
    rollout_data['rewards'] = summaries
