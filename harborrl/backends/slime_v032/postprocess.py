"""Defensive actor-side validation for the Slime v0.3.2 adapter."""
from __future__ import annotations

import math
import os

REQUIRED_FIELDS = (
    "tokens",
    "response_lengths",
    "rewards",
    "raw_reward",
    "truncated",
    "sample_indices",
    "rollout_ids",
    "loss_masks",
    "rollout_log_probs",
    "rollout_mask_sums",
)


def _finite_scalar(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _integer_scalar(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _validate_rollout_data(args, rollout_id, rollout_data):
    """Slime ``--rollout-data-postprocess-path`` hook; fail closed on drift."""
    missing = [field for field in REQUIRED_FIELDS if field not in rollout_data]
    if missing:
        raise ValueError(f"native rollout data is missing standard fields: {missing}")
    count = len(rollout_data["tokens"])
    fields = (
        "response_lengths", "rewards", "raw_reward", "truncated", "sample_indices",
        "rollout_ids", "loss_masks", "rollout_log_probs", "rollout_mask_sums",
    )
    if any(len(rollout_data[field]) != count for field in fields):
        raise ValueError("native rollout data fields have mismatched sample counts")
    if count < 1:
        raise ValueError("native rollout data is empty")
    teacher = rollout_data.get('teacher_log_probs')
    algorithm = os.environ.get('HARBORRL_MATH_ALGORITHM', 'grpo')
    if algorithm == 'opd':
        if teacher is None or len(teacher) != count:
            raise ValueError('OPD requires teacher scores for every sample')
    elif teacher is not None:
        raise ValueError('GRPO must not consume teacher scores')
    group_size = int(getattr(args, "n_samples_per_prompt", 0))
    if group_size < 2:
        raise ValueError("native training requires a complete group")
    if os.environ.get('HARBORRL_PROCESS_REWARD'):
        import torch
        if algorithm != 'grpo' or teacher is not None:
            raise ValueError('PRM requires GRPO without teacher scores')
        for field in ('advantages', 'returns'):
            values = rollout_data.get(field)
            if values is None or len(values) != count:
                raise ValueError('PRM requires computed token advantages and returns')
            for index, value in enumerate(values):
                if (value.ndim != 1 or value.numel() != rollout_data['response_lengths'][index]
                        or not torch.isfinite(value).all() or value.requires_grad):
                    raise ValueError('PRM advantages and returns must be aligned finite detached token tensors')
    group_weights = {}
    for index in range(count):
        group_id = _integer_scalar(rollout_data["rollout_ids"][index])
        if group_id is None or group_id < 0:
            raise ValueError("invalid native rollout id")
        response_length = _integer_scalar(rollout_data["response_lengths"][index])
        mask = rollout_data["loss_masks"][index]
        logprobs = rollout_data["rollout_log_probs"][index]
        mask_sum = _integer_scalar(rollout_data["rollout_mask_sums"][index])
        truncated = _integer_scalar(rollout_data["truncated"][index])
        if response_length is None or response_length < 1:
            raise ValueError("invalid native response length")
        if len(mask) != response_length or len(logprobs) != response_length:
            raise ValueError("native response field lengths differ")
        if any(flag not in (0, 1) for flag in mask):
            raise ValueError('Native response masks must be binary')
        if algorithm == 'dapo' and (not all(mask) or mask_sum != response_length):
            raise ValueError('DAPO requires all response tokens and exact per-sample token counts')
        if teacher is not None and (len(teacher[index]) != response_length or any(
                not _finite_scalar(value) or value > 0 for value in teacher[index])):
            raise ValueError('OPD teacher token scores are invalid or misaligned')
        if any(not _finite_scalar(value) for value in logprobs):
            raise ValueError("nonfinite native rollout logprob")
        if (not _finite_scalar(rollout_data["rewards"][index])
                or not _finite_scalar(rollout_data["raw_reward"][index])):
            raise ValueError("nonfinite native reward")
        if mask_sum is None or mask_sum < 1 or mask_sum < sum(1 for flag in mask if flag):
            raise ValueError("invalid native rollout mask sum")
        if truncated is None or truncated not in (0, 1):
            raise ValueError("invalid native truncated flag")
        group_weights.setdefault(group_id, []).append(
            sum(1 for flag in mask if flag) / (group_size * mask_sum)
        )
    # Actor shards may contain partial groups; each group has its own unit budget.
    for group_id, weights in group_weights.items():
        local_weight_sum = math.fsum(weights)
        if not math.isfinite(local_weight_sum) or local_weight_sum > 1.0 + 1e-9:
            raise ValueError(
                f"native local weights exceed the one-per-group budget for rollout id "
                f"{group_id}: {local_weight_sum}"
            )


def rollout_data_postprocess(args, rollout_id, rollout_data):
    from harborrl.consistency.config import environment
    if environment()["mode"] == "off":
        return _validate_rollout_data(args, rollout_id, rollout_data)
    from harborrl.consistency.slime import check_training_batch
    return check_training_batch(args, rollout_id, rollout_data, _validate_rollout_data)
