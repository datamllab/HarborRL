"""DAPO token sums; Megatron supplies the global valid-token denominator."""
from harborrl.backends.slime_v032.loss import clipped_loss


def loss_function(args, batch, logits, unused_reducer):
    import torch
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    if not args.calculate_per_token_loss or args.context_parallel_size != 1:
        raise ValueError('DAPO requires global per-token normalization and CP=1')
    _, output = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
        total_lengths=batch['total_lengths'], response_lengths=batch['response_lengths'],
        with_entropy=False)
    current = torch.cat(output['log_probs'])
    old, advantages, masks = (torch.cat(batch[key]).detach() for key in (
        'rollout_log_probs', 'advantages', 'loss_masks'))
    # The pinned Slime denominator clamps each sample's token count to >=1.
    # This DAPO variant retains every response token, including truncations.
    if not (masks == 1).all():
        raise ValueError('DAPO requires all response tokens, including truncations')
    loss = clipped_loss(current, old, advantages, torch.ones_like(current),
                        args.eps_clip, masks, epsilon_high=args.eps_clip_high)
    ratio = (current.detach() - old).exp()
    clipped = ((advantages > 0) & (ratio > 1 + args.eps_clip_high)) | (
        (advantages < 0) & (ratio < 1 - args.eps_clip))
    return loss, {'dapo_pg_loss': loss.detach(), 'dapo_token_count': masks.sum(),
                  'dapo_clipped_tokens': clipped.sum()}
