"""DAPO contracts for complete-group sampling and global token gradients."""
import json
import math
import os
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from test_math_rollout import samples

from harborrl.backends.slime_v032 import math_rollout
from harborrl.backends.slime_v032.postprocess import rollout_data_postprocess
from harborrl.config import ROOT, launch_plan, load_config
from harborrl.export.contract import DAPO_REDUCTION
from harborrl.export.validate import validate_batch
from harborrl.math import dapo
from harborrl.math.data import verify
from harborrl.platform.evidence import effective_config

SETTINGS = {'clip_low': .2, 'clip_high': .28, 'overlong_buffer_tokens': 2,
            'overlong_penalty': 1., 'max_refill_batches': 3}


def dapo_samples(monkeypatch, lengths=(1, 3)):
    monkeypatch.setenv('HARBORRL_MATH_ALGORITHM', 'dapo')
    monkeypatch.setenv('HARBORRL_DAPO', json.dumps(SETTINGS))
    rows = samples()
    for sample, length in zip(rows, lengths, strict=True):
        sample.tokens = [1, 2] + [3] * length
        sample.response_length = length
        sample.loss_mask = [1] * length
        sample.rollout_log_probs = [-.5] * length
        sample.rollout_id = 0
        evidence = sample.metadata['math_evidence']
        evidence.update(algorithm='dapo', tokens=sample.tokens, response_length=length,
                        logprobs=sample.rollout_log_probs,
                        dapo=dapo.reward(evidence['verification']['reward'], length, 4, SETTINGS))
    return rows


def test_dapo_shaping_advantages_and_backend_contract(monkeypatch):
    rows = dapo_samples(monkeypatch)
    args = SimpleNamespace(n_samples_per_prompt=2, rollout_max_response_len=4)
    batch = math_rollout.build_batch(args, rows)
    assert batch.reduction == DAPO_REDUCTION
    group = batch.groups[0]
    assert [unit.reward for unit in group.trajectories] == [1., -1.5]
    assert group.reward_std == 1.25
    expected = 1.25 / (math.sqrt(3.125) + 1e-6)
    assert [unit.advantage for unit in group.trajectories] == pytest.approx([expected, -expected])
    data = math_rollout.convert_samples_to_train_data(args, rows)
    assert data['raw_reward'] == [1., -1.]
    assert data['rollout_mask_sums'] == [1, 3]
    rollout_data_postprocess(args, 0, data)
    data['loss_masks'][1][-1] = 0
    with pytest.raises(ValueError, match='all response tokens'):
        rollout_data_postprocess(args, 0, data)
    wrong = replace(group.trajectories[0], advantage=1.)
    with pytest.raises(ValueError, match='sample std'):
        validate_batch(replace(batch, groups=(replace(group, trajectories=(wrong, group.trajectories[1])),)))
    rows[0].metadata['math_evidence']['dapo']['reward'] = 2.
    with pytest.raises(ValueError, match='DAPO reward'):
        math_rollout.build_batch(args, rows)


@pytest.mark.parametrize('length,expected', [(1, 0.), (2, 0.), (3, -.5), (4, -1.)])
def test_soft_overlong_boundary(length, expected):
    assert dapo.reward(1., length, 4, SETTINGS) == {'reward': 1. + expected, 'overlong_penalty': expected}


@pytest.mark.parametrize('outcome', [-1., 0., 1.])
def test_dapo_accepts_every_verifier_outcome(outcome):
    assert dapo.reward(outcome, 1, 4, SETTINGS) == {'reward': outcome, 'overlong_penalty': 0.}
    with pytest.raises(ValueError, match='verifier outcome'):
        dapo.reward(.5, 1, 4, SETTINGS)


def test_dapo_config_and_effective_semantics(tmp_path):
    config = yaml.safe_load((ROOT / 'examples/math/train.yaml').read_text(encoding='utf-8'))
    config['tasks']['path'] = str(ROOT / 'examples/math/tasks.jsonl')
    config['algorithm'] = {'name': 'dapo'}
    path = tmp_path / 'train.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    with pytest.raises(ValueError, match='dapo section'):
        load_config(path)
    config['dapo'] = SETTINGS
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    plan = launch_plan(load_config(path))
    assert json.loads(plan['environment']['HARBORRL_DAPO']) == SETTINGS
    semantics = effective_config(plan, plan['environment'], {}, {}, [])['update_semantics']
    assert semantics['std_normalization'] is True
    assert semantics['reduction'] == DAPO_REDUCTION
    for override in ('dapo.clip_low=1', 'dapo.clip_high=0.1', 'dapo.max_refill_batches=0',
                     'dapo.overlong_penalty=2', 'dapo.overlong_buffer_tokens=0', 'algorithm.name=grpo'):
        with pytest.raises(ValueError):
            load_config(path, [override])


@pytest.mark.parametrize('mode', ['success', 'exhausted', 'version', 'duplicate'])
@pytest.mark.skipif(os.name == 'nt', reason='production immutable evidence requires POSIX directory fsync')
def test_dynamic_sampling_complete_groups_and_failure_evidence(tmp_path, monkeypatch, mode):
    prototype = dapo_samples(monkeypatch, (1, 1))
    monkeypatch.setenv('RUN_DIR', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'slime.rollout.base_types', SimpleNamespace(RolloutFnTrainOutput=SimpleNamespace))
    args = SimpleNamespace(n_samples_per_prompt=2, rollout_batch_size=2, rollout_max_response_len=4,
                           rollout_temperature=1., rollout_top_p=1., rollout_top_k=-1)

    class Source:
        offset = 0

        def get_samples(self, count):
            groups = []
            for index in range(self.offset, self.offset + count):
                rows = deepcopy(prototype)
                for sample in rows:
                    sample.group_index = index
                groups.append(rows)
            self.offset += count
            return groups

    async def generate(args, rollout_id, groups):
        result = []
        for group in groups:
            for index, sample in enumerate(group):
                group_id = sample.group_index
                if group_id == 0 or mode == 'exhausted':
                    sample.response = '999999'
                task = sample.metadata['math_task']
                outcome = verify(sample.response, task['reference'])
                key = f'group-{group_id if mode != "duplicate" else 0}'
                evidence = sample.metadata['math_evidence']
                evidence.update(response=sample.response, verification=outcome, group_key=key,
                                rollout_id=group_id, trajectory_id=f'{key}:{index}',
                                policy_version='2' if mode == 'version' and group_id > 1 else '1',
                                dapo=dapo.reward(outcome['reward'], 1, 4, SETTINGS))
                sample.rollout_id = group_id
                sample.index = group_id * 2 + index
                result.append(sample)
        return result

    monkeypatch.setattr(math_rollout, '_generate', generate)
    source = Source()
    if mode == 'success':
        output = math_rollout.generate_rollout(args, 0, source)
        assert source.offset == 3
        assert len(output.samples) == 4
        assert {sample.group_index for sample in output.samples} == {1, 2}
        assert output.metrics['dapo_rejected_groups'] == 1.
        assert output.metrics['dapo_generation_batches'] == 2.
        selection = json.loads((tmp_path / 'math/000000/sampling.json').read_text())
        assert selection['status'] == 'complete'
        assert len(selection['accepted']) == 2 and len(selection['rejected']) == 1
    else:
        with pytest.raises(ValueError, match={'exhausted': 'refill exhausted', 'version': 'policy changed',
                                               'duplicate': 'duplicate'}[mode]):
            math_rollout.generate_rollout(args, 0, source)
        if mode == 'exhausted':
            selection = json.loads((tmp_path / 'math/000000/sampling.json').read_text())
            assert selection['status'] == 'exhausted'
            assert len(selection['rejected']) == 6
        assert (tmp_path / 'math/000000/sampling-evidence-000.json').is_file()


def test_dapo_loss_partition_invariance_and_asymmetric_gradient(monkeypatch):
    torch = pytest.importorskip('torch')
    from harborrl.backends.slime_v032.dapo import loss_function

    ratios = torch.tensor([.7, .85, 1.22, 1.3, .7, .85, 1.22, 1.3], dtype=torch.float64)
    current = (ratios.log() - 1).requires_grad_()
    old = torch.full((8,), -1., dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([1.] * 4 + [-1.] * 4, dtype=torch.float64, requires_grad=True)
    args = SimpleNamespace(calculate_per_token_loss=True, context_parallel_size=1,
                           eps_clip=.2, eps_clip_high=.28)

    def loss(indices):
        selected = current[indices]
        batch = {'unconcat_tokens': [], 'total_lengths': [len(indices) + 1],
                 'response_lengths': [len(indices)], 'rollout_log_probs': [old[indices]],
                 'advantages': [advantages[indices]], 'loss_masks': [torch.ones(len(indices))]}
        monkeypatch.setitem(sys.modules, 'slime.backends.megatron_utils.loss', SimpleNamespace(
            get_log_probs_and_entropy=lambda *a, **k: (None, {'log_probs': [selected]})))
        return loss_function(args, batch, selected, None)[0]

    full = loss(list(range(8))) / 8
    # Uneven DP shards and microbatches must sum before the global token division.
    partitioned = (loss([0, 3, 5]) + loss([2]) + loss([1, 4, 6, 7])) / 8
    assert partitioned.item() == pytest.approx(full.item())
    expected = torch.maximum(-advantages.detach() * ratios,
                             -advantages.detach() * ratios.clamp(.8, 1.28)).mean()
    assert full.item() == pytest.approx(expected.item())
    partitioned.backward()
    assert current.grad.tolist() == pytest.approx([-.7 / 8, -.85 / 8, -1.22 / 8, 0.,
                                                  0., .85 / 8, 1.22 / 8, 1.3 / 8])
    assert old.grad is None and advantages.grad is None
