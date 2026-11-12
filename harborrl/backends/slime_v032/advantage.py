"""Slime v0.3.2 custom advantage hook for Native GRPO."""
from __future__ import annotations

from harborrl.trajectories.native import finite


def _sequence_length(value):
    try:
        return len(value)
    except TypeError as exc:
        raise ValueError("advantage inputs must be sequences or tensors") from exc


def _response_values(prototype, value):
    if hasattr(prototype, "detach"):
        import torch
        return torch.full(
            (len(prototype),),
            float(value),
            dtype=torch.float32,
            device=prototype.device,
        )
    return [float(value)] * len(prototype)


def compute_advantages_and_returns(args, rollout_data):
    """Expand the converter's centered trajectory advantage to response tokens.

    The converter already writes the complete-group advantage into ``rewards``;
    regrouping here would be unsound after Slime's DP split.
    """
    rewards = rollout_data.get("rewards")
    loss_masks = rollout_data.get("loss_masks")
    logprobs = rollout_data.get("rollout_log_probs")
    if rewards is None or loss_masks is None or logprobs is None:
        raise ValueError("native advantage requires rewards, loss_masks, and rollout_log_probs")
    if not (len(rewards) == len(loss_masks) == len(logprobs)):
        raise ValueError("native advantage inputs have mismatched sample counts")
    advantages = []
    returns = []
    for index, reward in enumerate(rewards):
        if not finite(reward):
            raise ValueError("nonfinite native advantage")
        mask_length = _sequence_length(loss_masks[index])
        logprob_length = _sequence_length(logprobs[index])
        if mask_length < 1 or mask_length != logprob_length:
            raise ValueError("response mask and logprob lengths differ")
        values = _response_values(loss_masks[index], reward)
        advantages.append(values)
        returns.append(values)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
