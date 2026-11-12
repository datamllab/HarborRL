"""Native IR v2 to backend-neutral TrainingBatch export (pure Python)."""
from __future__ import annotations

import math

from harborrl.export.contract import (
    REDUCTION,
    SCHEMA_VERSION,
    TokenSpan,
    TrainingBatch,
    TrainingGroup,
    TrajectoryTrainingUnit,
)
from harborrl.export.validate import validate_batch
from harborrl.trajectories.native import Identity, finite, require_ready


def group_statistics(rewards, *, epsilon=1e-6):
    """Population statistics and centered GRPO advantages without std scaling."""
    values = list(rewards)
    if len(values) < 2:
        raise ValueError("a complete group of at least two trajectories is required")
    if any(not finite(value) for value in values):
        raise ValueError("nonfinite group statistics")
    if not finite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
    if not finite(mean) or not finite(std):
        raise ValueError("nonfinite group statistics")
    advantages = [value - mean for value in values]
    return mean, std, advantages


def _rollout_ids(ir_groups, rollout_ids, rollout_id_base):
    if rollout_ids is not None:
        ids = list(rollout_ids)
        if len(ids) != len(ir_groups):
            raise ValueError("rollout ids must align with native groups")
    else:
        if type(rollout_id_base) is not int or rollout_id_base < 0:
            raise ValueError("rollout_id_base must be a nonnegative integer")
        ids = [rollout_id_base + index for index in range(len(ir_groups))]
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("rollout ids must be nonnegative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("each native group needs a distinct rollout id")
    return ids


def export_training_batch(
    ir_groups,
    *,
    group_size=None,
    rollout_ids=None,
    rollout_id_base=0,
    epsilon=1e-6,
    allow_synthetic=False,
) -> TrainingBatch:
    """Validate complete Native IR groups and export immutable training units."""
    groups = [list(group) for group in ir_groups]
    if not groups:
        raise ValueError("native training batch requires at least one group")
    ids = _rollout_ids(groups, rollout_ids, rollout_id_base)
    exported_groups = []
    policy_versions = set()
    for trajectories, rollout_id in zip(groups, ids):
        if group_size is None:
            expected_size = len(trajectories)
        else:
            expected_size = group_size
            if type(expected_size) is not int:
                raise ValueError("group_size must be an integer")
        if expected_size < 2 or len(trajectories) != expected_size:
            raise ValueError("a complete group of at least two trajectories is required")
        for ir in trajectories:
            require_ready(ir, allow_synthetic=allow_synthetic)
        identities = [Identity(**ir["identity"]) for ir in trajectories]
        if len({identity.group_key for identity in identities}) != 1:
            raise ValueError("mixed group, policy, task or profiles")
        for key in ("slot_id", "attempt_id", "trajectory_id"):
            if len({getattr(identity, key) for identity in identities}) != expected_size:
                raise ValueError(f"duplicate {key}")
        rewards = [ir["evaluation"]["training_reward"] for ir in trajectories]
        mean, std, advantages = group_statistics(rewards, epsilon=epsilon)
        units = []
        for ir, identity, reward, advantage in zip(trajectories, identities, rewards, advantages):
            turns = []
            token_count = 0
            for turn_index, turn in enumerate(ir["turns"]):
                prompt = turn["input_ids"]
                response = turn["output_ids"]
                turns.append(TokenSpan(
                    turn_index=turn_index,
                    tokens=prompt + response,
                    prompt_length=len(prompt),
                    response_length=len(response),
                    loss_mask=[False] * len(prompt) + [True] * len(response),
                    old_logprobs=turn["logprobs"],
                    policy_version=identity.policy_version,
                ))
                token_count += len(response)
            units.append(TrajectoryTrainingUnit(
                trajectory_id=identity.trajectory_id,
                rollout_id=rollout_id,
                group_key=identity.group_key,
                reward=float(reward),
                advantage=float(advantage),
                token_count=token_count,
                turns=tuple(turns),
            ))
            policy_versions.add(identity.policy_version)
        exported_groups.append(TrainingGroup(
            group_key=identities[0].group_key,
            group_size=expected_size,
            reward_mean=mean,
            reward_std=std,
            trajectories=tuple(units),
        ))
    batch = TrainingBatch(
        schema_version=SCHEMA_VERSION,
        groups=tuple(exported_groups),
        reduction=REDUCTION,
        policy_versions=frozenset(policy_versions),
    )
    validate_batch(batch, epsilon=epsilon)
    return batch
