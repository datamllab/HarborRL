"""Fail-closed validation for backend-neutral training batches."""
from __future__ import annotations

import math

from harborrl.export.contract import (
    DAPO_REDUCTION,
    OPD_REDUCTION,
    REDUCTION,
    SCHEMA_VERSION,
    TokenSpan,
    TrainingBatch,
    TrainingGroup,
    TrajectoryTrainingUnit,
)
from harborrl.trajectories.native import finite


def _nonempty_str(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _plain_int(value, label):
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")


def token_weight(group_size, token_count):
    """Fixed trajectory-weighted reduction: 1 / (group_size * token_count)."""
    _plain_int(group_size, "group_size")
    _plain_int(token_count, "token_count")
    if group_size < 2 or token_count < 1:
        raise ValueError("group_size must be >= 2 and token_count >= 1")
    return 1.0 / (group_size * token_count)


def validate_token_span(span: TokenSpan):
    _plain_int(span.turn_index, "turn_index")
    if span.turn_index < 0:
        raise ValueError("turn_index must be nonnegative")
    _plain_int(span.prompt_length, "prompt_length")
    _plain_int(span.response_length, "response_length")
    if span.prompt_length < 1 or span.response_length < 1:
        raise ValueError("a token span requires prompt and response tokens")
    if len(span.tokens) != span.prompt_length + span.response_length:
        raise ValueError("token length differs from prompt plus response length")
    if any(type(token) is not int or token < 0 for token in span.tokens):
        raise ValueError("tokens must be nonnegative integers")
    if len(span.loss_mask) != len(span.tokens):
        raise ValueError("loss mask length differs from token length")
    if any(type(flag) is not bool for flag in span.loss_mask):
        raise ValueError("loss mask entries must be booleans")
    if any(flag for flag in span.loss_mask[:span.prompt_length]):
        raise ValueError("prompt tokens must not enter the policy loss")
    if not all(span.loss_mask[span.prompt_length:]):
        raise ValueError("native response tokens must all be trainable")
    if len(span.old_logprobs) != span.response_length:
        raise ValueError("old logprob length differs from response length")
    if any(not finite(value) or value > 0 for value in span.old_logprobs):
        raise ValueError("old logprobs must be finite and nonpositive")
    _nonempty_str(span.policy_version, "policy_version")
    if span.process_rewards is not None:
        from harborrl.export.contract import ProcessRewards
        from harborrl.math.prm import token_returns
        scores = span.process_rewards
        if not isinstance(scores, ProcessRewards):
            raise ValueError('process_rewards must be a ProcessRewards contract')
        _nonempty_str(scores.identity, 'process reward identity')
        expected = token_returns(scores.step_ends, scores.probabilities, span.response_length)
        if tuple(expected) != scores.token_returns:
            raise ValueError('PRM token returns differ from step reward-to-go')


def validate_trajectory(unit: TrajectoryTrainingUnit):
    _nonempty_str(unit.trajectory_id, "trajectory_id")
    _nonempty_str(unit.group_key, "group_key")
    _plain_int(unit.rollout_id, "rollout_id")
    if unit.rollout_id < 0:
        raise ValueError("rollout_id must be nonnegative")
    if not finite(unit.reward) or not finite(unit.advantage):
        raise ValueError("trajectory reward and advantage must be finite")
    _plain_int(unit.token_count, "token_count")
    if not unit.turns:
        raise ValueError("a trajectory requires at least one turn")
    if [span.turn_index for span in unit.turns] != list(range(len(unit.turns))):
        raise ValueError("trajectory turn indices must be dense and ordered")
    versions = set()
    for span in unit.turns:
        validate_token_span(span)
        versions.add(span.policy_version)
    if len(versions) != 1:
        raise ValueError("a trajectory mixes policy versions")
    if unit.token_count != sum(span.response_length for span in unit.turns):
        raise ValueError("trajectory token count differs from its turns")


def validate_group(group: TrainingGroup, *, epsilon=1e-6, algorithm="grpo"):
    if algorithm not in ('grpo', 'opd', 'dapo'):
        raise ValueError('unsupported group algorithm')
    _nonempty_str(group.group_key, "group_key")
    _plain_int(group.group_size, "group_size")
    if group.group_size < 2 or len(group.trajectories) != group.group_size:
        raise ValueError("a complete group of at least two trajectories is required")
    if not all(unit.group_key == group.group_key for unit in group.trajectories):
        raise ValueError("mixed group, policy, task or profiles")
    trajectory_ids = [unit.trajectory_id for unit in group.trajectories]
    if len(set(trajectory_ids)) != group.group_size:
        raise ValueError("duplicate trajectory_id")
    rollout_ids = [unit.rollout_id for unit in group.trajectories]
    if len(set(rollout_ids)) != 1:
        raise ValueError("one native group must share exactly one rollout id")
    if not finite(group.reward_mean) or not finite(group.reward_std) or group.reward_std < 0:
        raise ValueError("nonfinite or negative group statistics")
    rewards = [unit.reward for unit in group.trajectories]
    mean = math.fsum(rewards) / group.group_size
    std = math.sqrt(math.fsum((reward - mean) ** 2 for reward in rewards) / group.group_size)
    if not math.isclose(mean, group.reward_mean, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("group reward mean differs from its trajectories")
    if not math.isclose(std, group.reward_std, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("group reward std differs from its trajectories")
    if algorithm == "grpo" and group.reward_std == 0 and any(unit.advantage != 0.0 for unit in group.trajectories):
        raise ValueError("zero-std groups must produce zero advantages")
    for unit in group.trajectories:
        validate_trajectory(unit)
        if algorithm in ('grpo', 'dapo'):
            expected = unit.reward - mean
            if algorithm == 'dapo':
                from harborrl.math.dapo import advantage
                if std == 0:
                    raise ValueError('DAPO requires a nonzero reward std')
                expected = advantage(unit.reward, mean, std, group.group_size)
            if not math.isclose(unit.advantage, expected, rel_tol=1e-9, abs_tol=1e-9):
                if algorithm == 'dapo':
                    raise ValueError('DAPO advantage must use complete-group sample std plus 1e-6')
                raise ValueError("GRPO advantage must be centered without std normalization")
            if any(span.teacher_logprobs is not None for span in unit.turns):
                raise ValueError("Outcome policy gradients must not consume teacher scores")
        else:
            if unit.advantage != 0.0:
                raise ValueError("OPD requires token scores, not a scalar reward advantage")
            for span in unit.turns:
                values = span.teacher_logprobs
                if (values is None or len(values) != span.response_length
                        or any(not finite(value) or value > 0 for value in values)):
                    raise ValueError("OPD requires aligned finite teacher token logprobs")
        if algorithm == 'dapo':
            continue
        weight_sum = math.fsum(
            token_weight(group.group_size, unit.token_count) for span in unit.turns for _ in range(span.response_length)
        )
        if not math.isclose(weight_sum, 1.0 / group.group_size, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("trajectory weights must sum to 1 / group_size")


def validate_batch(batch: TrainingBatch, *, epsilon=1e-6):
    if batch.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported training batch schema: {batch.schema_version}")
    reductions = {"grpo": REDUCTION, "opd": OPD_REDUCTION, "dapo": DAPO_REDUCTION}
    if batch.algorithm not in reductions or batch.reduction != reductions[batch.algorithm]:
        raise ValueError(f"unsupported training reduction: {batch.reduction}")
    if batch.algorithm == "opd":
        _nonempty_str(batch.teacher_identity, "teacher_identity")
    elif batch.teacher_identity is not None:
        raise ValueError("Only OPD may declare a teacher")
    if batch.process_reward_identity is not None:
        _nonempty_str(batch.process_reward_identity, 'process_reward_identity')
        if batch.algorithm != 'grpo':
            raise ValueError('Process rewards currently require GRPO')
    if not batch.groups:
        raise ValueError("native training batch requires at least one group")
    if len(batch.policy_versions) != 1:
        raise ValueError("incomplete native turns or mixed policy batch")
    group_keys = [group.group_key for group in batch.groups]
    rollout_ids = [group.trajectories[0].rollout_id for group in batch.groups]
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("duplicate native group")
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("each native group needs a distinct rollout id")
    for group in batch.groups:
        validate_group(group, epsilon=epsilon, algorithm=batch.algorithm)
        for unit in group.trajectories:
            scores = [span.process_rewards for span in unit.turns]
            if batch.process_reward_identity is None:
                if any(value is not None for value in scores):
                    raise ValueError('PRM evidence requires a batch process_reward_identity')
            elif (len(scores) != 1 or scores[0] is None
                  or scores[0].identity != batch.process_reward_identity
                  or not math.isclose(unit.reward, scores[0].token_returns[0], rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError('PRM batch requires one aligned process reward source and matching total reward')
        versions = {span.policy_version for unit in group.trajectories for span in unit.turns}
        if versions != batch.policy_versions:
            raise ValueError("training batch mixes policy versions")


def batch_weight_sum(batch: TrainingBatch):
    """Sum of trajectory weights; each complete group contributes exactly one."""
    if batch.algorithm == 'dapo':
        validate_batch(batch)
        return 1.0
    return math.fsum(
        token_weight(group.group_size, unit.token_count)
        for group in batch.groups
        for unit in group.trajectories
        for span in unit.turns
        for _ in range(span.response_length)
    )
