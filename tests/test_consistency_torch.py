"""Real optimizer and distributed rejection tests; optional Torch dependency."""
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from harborrl.consistency.config import validate
from harborrl.consistency.report import assess, distributed_gate
from harborrl.distill.config import DEFAULTS
from harborrl.distill.models import load_student, parameter_digest, response_logprobs
from harborrl.distill.runner import optimize


def test_gate_rejects_before_optimizer_or_gradient(tmp_path):
    from copy import deepcopy
    config = deepcopy(DEFAULTS)
    config['consistency'] = validate({'mode': 'enforce', 'max_abs_delta': .001})
    config['output']['directory'] = str(tmp_path)
    config['method'] = 'reinforce'
    config['training']['outcome_weight'] = 1.
    model, tokenizer = load_student(config, torch.device('cpu'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = parameter_digest(model)
    row = {'sample_id': '000000-000000', 'prompt_ids': [1, 4], 'response_ids': [5, 6],
           'behavior_logprobs': [-.01, -.01], 'loss_mask': [1, 1], 'verification': {'reward': 1.}}
    with pytest.raises(ValueError, match='rejected'):
        optimize(model, optimizer, [row], config, tokenizer, 1)
    assert parameter_digest(model) == before
    assert all(p.grad is None for p in model.parameters())
    assert not optimizer.state
    row['sample_id'] = '000001-000000'
    row['behavior_logprobs'] = response_logprobs(model, [[1, 4]], [[5, 6]], 0)[0].detach().tolist()
    result = optimize(model, optimizer, [row], config, tokenizer, 1)
    assert result['gradient_norm'] > 0
    assert parameter_digest(model) != before


def rank_failure(rank, rendezvous, root):
    from datetime import timedelta

    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        settings = validate({'mode': 'enforce', 'max_abs_delta': .01})
        part = assess([([-1.], [-1. if rank == 0 else float('nan')], [1])], settings)
        with pytest.raises(ValueError, match='rejected'):
            distributed_gate(part, settings, Path(root) / 'failure.json', phase='injected-rank-failure')
        (Path(root) / f'rank{rank}.ok').write_text('rejected')
    finally:
        dist.destroy_process_group()


def test_one_rank_error_rejects_all_ranks(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(rank_failure, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / 'rank0.ok').exists()
    assert (tmp_path / 'rank1.ok').exists()
