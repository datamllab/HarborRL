"""Frozen pre-update reference check, performed before any backward call."""
from pathlib import Path

from .report import assess, distributed_gate


def check_reference(model, rows, config, tokenizer):
    import torch
    import torch.distributed as dist

    from harborrl.distill.models import response_logprobs

    settings = config['consistency']
    pairs = []
    try:
        with torch.no_grad():
            for row in rows:
                p0 = response_logprobs(model, [row['prompt_ids']], [row['response_ids']], tokenizer.pad_token_id)[0]
                pairs.append((p0.detach().float().cpu().tolist(), row['behavior_logprobs'], row['loss_mask']))
        part = assess(pairs, settings, metadata={'backend': 'transformers', 'rank': dist.get_rank() if dist.is_initialized() else 0})
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        part = {'statistics': None, 'error': str(exc), 'metadata': {'backend': 'transformers'}}
    step = rows[0]['sample_id'].split('-')[0] if rows else 'empty'
    return distributed_gate(part, settings, Path(config['output']['directory']) / 'consistency' / f'batch-{step}.json',
                            phase='pre-update-backend-gap')
