"""Sampled-token OPD hooks for the shared Slime/Megatron optimizer loop."""
from __future__ import annotations

from harborrl.algorithms.opd import weighted_sampled_opd_loss
from harborrl.export.validate import token_weight


def compute_advantages_and_returns(args, rollout_data):
    import torch

    teacher = rollout_data.get('teacher_log_probs')
    behavior = rollout_data.get('rollout_log_probs')
    masks = rollout_data.get('loss_masks')
    if teacher is None or behavior is None or masks is None or not len(teacher):
        raise ValueError('OPD requires teacher and behavior token logprobs and masks')
    if not len(teacher) == len(behavior) == len(masks):
        raise ValueError('OPD sample counts differ')
    advantages = []
    for target, old, mask in zip(teacher, behavior, masks, strict=True):
        if target.shape != old.shape or old.shape != mask.shape or old.ndim != 1:
            raise ValueError('OPD teacher, behavior and mask token positions differ')
        if not torch.isfinite(target).all() or not torch.isfinite(old).all():
            raise ValueError('Nonfinite OPD token logprob')
        advantages.append((target - old).detach())
    rollout_data['advantages'] = advantages
    rollout_data['returns'] = advantages


def loss_function(args, batch, logits, unused_reducer):
    import torch
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    _, output = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
        total_lengths=batch['total_lengths'], response_lengths=batch['response_lengths'],
        with_entropy=False,
    )
    weights = torch.cat([
        torch.full((int(length),), token_weight(int(args.n_samples_per_prompt), int(count)),
                   dtype=torch.float32, device=logits.device)
        for length, count in zip(batch['response_lengths'], batch['rollout_mask_sums'], strict=True)
    ])
    current = torch.cat(output['log_probs'])
    behavior = torch.cat(batch['rollout_log_probs'])
    teacher = torch.cat(batch['teacher_log_probs'])
    mask = torch.cat(batch['loss_masks'])
    loss = weighted_sampled_opd_loss(current, behavior, teacher, weights, mask)
    valid_weights = weights * mask.to(weights.dtype)
    return loss, {
        'opd_surrogate_loss': loss.detach(),
        'opd_sampled_reverse_kl': ((behavior - teacher).detach() * valid_weights).sum(),
        'opd_teacher_behavior_abs': ((teacher - behavior).detach().abs() * valid_weights).sum(),
        'opd_weight_sum': valid_weights.sum().detach(),
        'opd_token_count': mask.sum().detach(),
    }
