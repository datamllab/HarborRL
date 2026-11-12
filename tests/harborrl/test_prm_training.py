"""PRM identity, boundaries, complete-group baselines and gradient contracts."""
import json
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import pytest
import yaml
from test_math_rollout import samples

from harborrl.backends.slime_v032 import math_rollout
from harborrl.config import ROOT, launch_plan, load_config
from harborrl.export.validate import validate_batch
from harborrl.math import prm
from harborrl.math.data import verify
from harborrl.platform.evidence import effective_config, run_environment
from harborrl.trajectories.native import digest


class Tokenizer:
    pieces: ClassVar = {1: '2', 2: '\n\n', 3: '#### 2', 4: ' and next', 5: '\n\nNext', 6: '#### 999999', 99: ''}

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(self.pieces[token] for token in ids)


@pytest.fixture
def frozen(tmp_path):
    model = tmp_path / 'prm'
    model.mkdir()
    config = {'architectures': ['Qwen2ForProcessRewardModel'], 'auto_map': {
        'AutoConfig': 'configuration_qwen2_rm.Qwen2RMConfig',
        'AutoModel': 'modeling_qwen2_rm.Qwen2ForProcessRewardModel'}}
    (model / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'configuration_qwen2_rm.py',
                 'modeling_qwen2_rm.py', 'model.safetensors'):
        (model / name).write_text('{}', encoding='utf-8')
    return prm.freeze({'endpoint': 'http://127.0.0.1:30091', 'checkpoint': str(model), 'revision': 'a' * 40})


def receipt(frozen, query, ids, probabilities):
    def transport(url, payload):
        assert url.endswith('/score')
        return {'identity': frozen['identity'], 'request_digest': digest(payload),
                'scorer_input_digest': digest([7, 8, 9]), 'probabilities': probabilities}
    return prm.score_response(frozen, query, ids, Tokenizer(), transport=transport)


def setup_samples(frozen, tmp_path, monkeypatch):
    manifest = tmp_path / 'process_reward.json'
    manifest.write_text(json.dumps(frozen), encoding='utf-8')
    monkeypatch.setenv('HARBORRL_PROCESS_REWARD', str(manifest))
    monkeypatch.setenv('HARBORRL_MATH_ALGORITHM', 'grpo')
    monkeypatch.setitem(sys.modules, 'slime.utils.processing_utils', SimpleNamespace(
        load_tokenizer=lambda *args, **kwargs: Tokenizer()))
    rows = samples()
    for sample, scores in zip(rows, ([.9, .8], [.2, .4]), strict=True):
        ids = [1, 2, 3 if sample.index == 0 else 6, 99]
        sample.tokens = [42, 43] + ids
        sample.response = Tokenizer().decode(ids)
        sample.response_length = len(ids)
        sample.loss_mask = [1] * len(ids)
        sample.rollout_log_probs = [-.5] * len(ids)
        sample.rollout_id = 0
        evidence = sample.metadata['math_evidence']
        evidence.update(tokens=list(sample.tokens), response=sample.response, response_length=len(ids),
                        verification=verify(sample.response, sample.metadata['math_task']['reference']),
                        logprobs=list(sample.rollout_log_probs),
                        process_reward=receipt(frozen, sample.metadata['math_task']['prompt'], ids, scores))
    return rows, SimpleNamespace(n_samples_per_prompt=2, hf_checkpoint='student')


def test_identity_pins_executable_source_and_weights(frozen):
    config = {key: frozen[key] for key in prm.FIELDS}
    assert prm.freeze({**config, 'endpoint': 'http://127.0.0.1:39999'})['identity'] == frozen['identity']
    from pathlib import Path
    for name in ('modeling_qwen2_rm.py', 'model.safetensors', 'tokenizer.json'):
        path = Path(frozen['checkpoint']) / name
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        assert prm.freeze(config)['identity'] != frozen['identity']
        path.write_bytes(original)


@pytest.mark.parametrize('mutation', ['identity', 'revision', 'frozen', 'is_generation'])
def test_live_scorer_identity(frozen, mutation):
    info = {key: frozen[key] for key in ('identity', 'revision', 'adapter', 'checkpoint')}
    info.update(frozen=True, is_generation=False)
    prm.check(frozen, transport=lambda url: info)
    info[mutation] = not info[mutation] if type(info[mutation]) is bool else 'changed'
    with pytest.raises(ValueError):
        prm.check(frozen, transport=lambda url: info)


@pytest.mark.parametrize('ids,ends,steps', [
    ([1, 2, 3, 99], [2, 4], ['2\n\n', '#### 2']),
    ([1, 5, 3, 99], [2, 4], ['2\n\nNext', '#### 2']),
    ([2, 1, 2, 99], [4], ['\n\n2\n\n']),
    ([1, 2, 99], [3], ['2\n\n']),
])
def test_steps_keep_boundary_tokens_and_trailing_specials(ids, ends, steps):
    assert prm.split_steps(ids, Tokenizer()) == (ends, steps)
    assert ''.join(steps) == Tokenizer().decode(ids)


def test_partial_unicode_prefix_waits_for_complete_character():
    class ByteTokenizer:
        def decode(self, ids, **kwargs):
            return {1: 'x\n\n\ufffd', 2: 'x\n\n\u6570', 3: 'x\n\n\u6570y'}[len(ids)]
    assert prm.split_steps([1, 2, 3], ByteTokenizer()) == ([2, 3], ['x\n\n\u6570', 'y'])
    with pytest.raises(ValueError, match='no visible reasoning'):
        prm.split_steps([99], Tokenizer())


@pytest.mark.parametrize('ends,values', [([0, 2], [.5, .5]), ([2, 2], [.5, .5]),
                                       ([1], [.5]), ([2], [float('nan')]), ([2], [1.1])])
def test_invalid_token_partition_and_probability(ends, values):
    with pytest.raises(ValueError):
        prm.token_returns(ends, values, 2)


@pytest.mark.parametrize('mutation', ['identity', 'request_digest', 'response_tokens_digest', 'step_ends',
                                     'probabilities', 'token_returns', 'scorer_input_digest'])
def test_scoring_receipt_binds_request_tokens_and_returns(frozen, mutation):
    scores = receipt(frozen, 'question', [1, 2, 3, 99], [.9, .2])
    scores[mutation] = {'step_ends': [1, 4], 'probabilities': [.9, .3], 'token_returns': [9.] * 4}.get(mutation, 'changed')
    with pytest.raises(ValueError):
        prm.validate_receipt(frozen, scores, 'question', [1, 2, 3, 99], Tokenizer())


def test_prm_neutral_contract_and_scalar_outcome_separation(frozen, tmp_path, monkeypatch):
    rows, args = setup_samples(frozen, tmp_path, monkeypatch)
    batch = math_rollout.build_batch(args, rows)
    assert batch.algorithm == 'grpo'
    assert batch.process_reward_identity == frozen['identity']
    group = batch.groups[0]
    assert [unit.reward for unit in group.trajectories] == pytest.approx([1.4, -.8])
    assert [unit.advantage for unit in group.trajectories] == pytest.approx([1.1, -1.1])
    assert group.trajectories[0].turns[0].process_rewards.token_returns == pytest.approx([1.4, 1.4, .6, .6])
    data = math_rollout.convert_samples_to_train_data(args, rows)
    assert data['raw_reward'] == [1., -1.]
    assert data['rewards'][0]['baseline'] == pytest.approx(.3)
    assert 'teacher_log_probs' not in data
    for changed in (replace(batch, process_reward_identity=None), replace(batch, process_reward_identity='changed')):
        with pytest.raises(ValueError, match='PRM'):
            validate_batch(changed)
    corrupted = replace(group.trajectories[0].turns[0].process_rewards, token_returns=(0.,) * 4)
    unit = replace(group.trajectories[0], turns=(replace(group.trajectories[0].turns[0], process_rewards=corrupted),))
    with pytest.raises(ValueError, match='reward-to-go'):
        validate_batch(replace(batch, groups=(replace(group, trajectories=(unit, group.trajectories[1])),)))


def test_prm_config_uses_same_training_chain_and_explicit_semantics(tmp_path):
    config = yaml.safe_load((ROOT / 'examples/math/train.yaml').read_text(encoding='utf-8'))
    config['tasks']['path'] = str(ROOT / 'examples/math/tasks.jsonl')
    config['process_reward'] = {'endpoint': 'http://127.0.0.1:30091', 'checkpoint': '/models/prm', 'revision': 'a' * 40}
    path = tmp_path / 'train.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    plan = launch_plan(load_config(path))
    environment = run_environment(plan, tmp_path, path, path)
    assert environment['HARBORRL_PROCESS_REWARD'] == str(tmp_path / 'config' / 'process_reward.json')
    semantics = effective_config(plan, environment, {}, {}, [])['update_semantics']
    assert semantics['outcome_reward_mixing'] is False
    assert semantics['std_normalization'] is False
    assert 'reward_to_go' in semantics['advantage']
    assert plan['training_command'][-1].endswith('slime_train.sh')
    for override in ('algorithm.name=opd', 'algorithm.name=dapo', 'process_reward.revision=main'):
        with pytest.raises(ValueError):
            load_config(path, [override])


def test_dp_partial_group_retains_full_group_baseline_and_validates_transport(frozen, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from harborrl.backends.slime_v032.postprocess import rollout_data_postprocess
    from harborrl.backends.slime_v032.prm import compute_advantages_and_returns
    rows, args = setup_samples(frozen, tmp_path, monkeypatch)
    original = math_rollout.convert_samples_to_train_data(args, rows)

    def shard(index):
        data = {key: [deepcopy(values[index])] for key, values in original.items()}
        for key in ('tokens', 'rollout_log_probs', 'loss_masks'):
            data[key] = [torch.tensor(data[key][0])]
        return data

    data = shard(0)
    compute_advantages_and_returns(args, data)
    assert data['advantages'][0].tolist() == pytest.approx([1.1, 1.1, .3, .3])
    assert data['returns'][0].tolist() == pytest.approx([1.4, 1.4, .6, .6])
    assert data['rewards'] == pytest.approx([1.1])
    rollout_data_postprocess(args, 0, data)
    for key, value in (('identity', 'changed'), ('tokens_digest', 'changed'), ('baseline', 9.),
                       ('sample_index', 5), ('rollout_id', 5), ('group_totals', {'trajectory-0': 1.4})):
        data = shard(0)
        data['rewards'][0][key] = value
        with pytest.raises(ValueError):
            compute_advantages_and_returns(args, data)


def test_token_process_advantages_reach_ragged_clipped_policy_gradient(frozen, tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from harborrl.backends.slime_v032.loss import loss_function
    from harborrl.backends.slime_v032.prm import compute_advantages_and_returns
    rows, args = setup_samples(frozen, tmp_path, monkeypatch)
    args.eps_clip = .2
    data = math_rollout.convert_samples_to_train_data(args, rows)
    for key in ('tokens', 'rollout_log_probs', 'loss_masks'):
        data[key] = [torch.tensor(values) for values in data[key]]
    compute_advantages_and_returns(args, data)
    data.update(unconcat_tokens=[], total_lengths=[6, 6])
    ratios = torch.tensor([.7, 1., 1.3, 1., .7, 1., 1.3, 1.], dtype=torch.float64)
    current = (ratios.log() - .5).requires_grad_()
    monkeypatch.setitem(sys.modules, 'slime.backends.megatron_utils.loss', SimpleNamespace(
        get_log_probs_and_entropy=lambda *a, **k: (None, {'log_probs': list(current.split(4))})))
    loss, _ = loss_function(args, data, current, None)
    advantages = torch.tensor([1.1, 1.1, .3, .3, -1.1, -1.1, -.5, -.5])
    expected = torch.maximum(-advantages * ratios, -advantages * ratios.clamp(.8, 1.2)).mean()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert current.grad.tolist() == pytest.approx([-.7 * 1.1 / 8, -1.1 / 8, 0., -.3 / 8,
                                                  0., 1.1 / 8, 1.3 * .5 / 8, .5 / 8])
    assert all(not tensor.requires_grad for tensor in data['advantages'] + data['returns'])


def test_scorer_uses_frozen_marker_logits_and_float32_probabilities(frozen):
    torch = pytest.importorskip('torch')
    from harborrl.math.prm_server import Scorer

    class Model:
        training = False
        device = torch.device('cpu')
        config = SimpleNamespace(max_position_embeddings=5)

        def parameters(self):
            return [torch.nn.Parameter(torch.zeros(1), requires_grad=False)]

        def __call__(self, *, input_ids, use_cache):
            assert not torch.is_grad_enabled() and use_cache is False
            assert input_ids.tolist() == [[7, 99, 8, 99, 9]]
            return SimpleNamespace(logits=torch.tensor(
                [[[0., 0.], [.25, 1.5], [0., 0.], [2., -.5], [0., 0.]]], dtype=torch.bfloat16))

    class ScorerTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'system', 'content': prm.SYSTEM},
                                {'role': 'user', 'content': 'question'},
                                {'role': 'assistant', 'content': 'first<extra_0>second<extra_0>'}]
            assert kwargs == {'tokenize': False, 'add_generation_prompt': False}
            return 'rendered'

        def encode(self, text, **kwargs):
            assert text == 'rendered' and kwargs == {'add_special_tokens': False}
            return [7, 99, 8, 99, 9]

    scorer = Scorer.__new__(Scorer)
    scorer.frozen, scorer.model = frozen, Model()
    scorer.tokenizer, scorer.marker_id = ScorerTokenizer(), [99]
    payload = prm.request_payload(frozen['identity'], 'question', ['first', 'second'])
    result = scorer.score(payload)
    expected = torch.tensor([[.25, 1.5], [2., -.5]], dtype=torch.float32).softmax(-1)[:, 1].tolist()
    assert result['probabilities'] == pytest.approx(expected, abs=1e-7)
    assert result['request_digest'] == digest(payload)
    assert result['scorer_input_digest'] == digest([7, 99, 8, 99, 9])
    scorer.model.training = True
    with pytest.raises(ValueError, match='frozen'):
        scorer.score(payload)
    scorer.model.training = False
    scorer.model.config.max_position_embeddings = 4
    with pytest.raises(ValueError, match='context exceeded'):
        scorer.score(payload)
    scorer.model.config.max_position_embeddings = 5
    scorer.marker_id = [98]
    with pytest.raises(ValueError, match='marker positions'):
        scorer.score(payload)
    for changed in ({**payload, 'identity': 'changed'}, {**payload, 'extra': True},
                    {**payload, 'steps': ['injected<extra_0>']}):
        with pytest.raises(ValueError):
            scorer.score(changed)
