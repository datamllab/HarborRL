"""Slime v0.3.2 custom clipped PG loss for Native trajectory weights."""
from __future__ import annotations

import math

from harborrl.export.validate import token_weight
from harborrl.trajectories.native import finite

# The pinned Slime scheduler counts distinct rollout ids, one per group.
# Its outer microbatch/DP scaling therefore averages these group-weighted
# sums over groups, with no additional group-size correction.
LOSS_SCALE_CORRECTION = 1.0


def _validate_loss_inputs(new_logprobs, old_logprobs, advantages, weights, epsilon):
    if not (len(new_logprobs) == len(old_logprobs) == len(advantages) == len(weights)):
        raise ValueError("native loss inputs must have identical lengths")
    if not 0 < epsilon < float("inf"):
        raise ValueError("native clipping epsilon must be finite and positive")
    for values in (new_logprobs, old_logprobs, advantages, weights):
        if any(not finite(value) for value in values):
            raise ValueError("nonfinite native loss input")
    if any(weight < 0 for weight in weights):
        raise ValueError("negative native token weight")


def clipped_pg_loss_pure(new_logprobs, old_logprobs, advantages, weights, epsilon=0.2):
    """CPU reference for the global sum of weighted clipped PG losses."""
    _validate_loss_inputs(new_logprobs, old_logprobs, advantages, weights, epsilon)
    total = 0.0
    for new_logprob, old_logprob, advantage, weight in zip(
        new_logprobs, old_logprobs, advantages, weights
    ):
        ratio = math.exp(new_logprob - old_logprob)
        if not math.isfinite(ratio):
            raise ValueError("nonfinite native policy ratio")
        unclipped = -advantage * ratio
        clipped = -advantage * min(max(ratio, 1 - epsilon), 1 + epsilon)
        total += max(unclipped, clipped) * weight
    return total


def clipped_loss(new_logprobs, old_logprobs, advantages, weights, epsilon=0.2, loss_masks=None, epsilon_high=None):
    """Global sum of weighted PPO losses; usable on CPU for gradient contracts."""
    import torch
    if not (new_logprobs.shape == old_logprobs.shape == advantages.shape == weights.shape):
        raise ValueError("native token tensors must have identical shapes")
    if loss_masks is not None and loss_masks.shape != weights.shape:
        raise ValueError("native loss mask must match token weights")
    if not 0 < epsilon < float("inf"):
        raise ValueError("native clipping epsilon must be finite and positive")
    epsilon_high = epsilon if epsilon_high is None else epsilon_high
    if not 0 < epsilon_high < float('inf'):
        raise ValueError('native upper clipping epsilon must be finite and positive')
    if any(not torch.isfinite(tensor).all() for tensor in (new_logprobs, old_logprobs, advantages, weights)) or (weights < 0).any():
        raise ValueError("nonfinite native loss input or negative weight")
    ratio = (new_logprobs - old_logprobs).exp()
    loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - epsilon, 1 + epsilon_high))
    if loss_masks is not None:
        weights = weights * loss_masks.to(device=weights.device, dtype=weights.dtype)
    return (loss * weights).sum()


def loss_function(args, batch, logits, unused_reducer):
    """Slime ``--custom-loss-function-path`` hook using only standard fields."""
    import torch
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    _, output = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )
    group_size = int(args.n_samples_per_prompt)
    if group_size < 2:
        raise ValueError("native training requires a complete group")
    weights = torch.cat([
        torch.full(
            (int(response_length),),
            token_weight(group_size, int(mask_sum)),
            dtype=torch.float32,
            device=logits.device,
        )
        for response_length, mask_sum in zip(
            batch["response_lengths"], batch["rollout_mask_sums"], strict=True
        )
    ])
    old_logprobs, advantages, loss_masks = (
        torch.cat(batch[key])
        for key in ("rollout_log_probs", "advantages", "loss_masks")
    )
    loss = clipped_loss(
        torch.cat(output["log_probs"]),
        old_logprobs,
        advantages,
        weights,
        args.eps_clip,
        loss_masks,
    )
    weight_sum = (weights * loss_masks.to(device=weights.device, dtype=weights.dtype)).sum()
    metrics = {
        "native_pg_loss": loss.detach(),
        "native_weight_sum": weight_sum.detach(),
        "native_token_count": loss_masks.sum().detach(),
        "native_group_count": weight_sum.detach().round(),
    }
    from harborrl.consistency.config import environment
    if environment()["mode"] != "off":
        p0 = torch.cat(batch["log_probs"]).detach()
        current = torch.cat(output["log_probs"]).detach()
        metrics["consistency_backend_abs_sum"] = ((p0 - old_logprobs).abs() * loss_masks).sum()
        metrics["consistency_update_abs_sum"] = ((current - p0).abs() * loss_masks).sum()
        metrics["consistency_valid_tokens"] = loss_masks.sum().detach()
    return loss * LOSS_SCALE_CORRECTION, metrics

