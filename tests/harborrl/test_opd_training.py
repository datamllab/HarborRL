"""OPD contracts at the frozen-teacher and shared production-loss boundaries."""
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from test_math_rollout import samples

from harborrl.backends.slime_v032.math_rollout import (
    build_batch,
    convert_samples_to_train_data,
)
from harborrl.backends.slime_v032.postprocess import rollout_data_postprocess
from harborrl.config import ROOT, launch_plan, load_config
from harborrl.export.validate import validate_batch
from harborrl.math import teacher
from harborrl.platform.evidence import effective_config, run_environment
from harborrl.trajectories.native import digest


@pytest.fixture
def frozen(tmp_path):
    checkpoint = tmp_path / 'teacher'
    checkpoint.mkdir()
    for name, data in {'config.json': '{}', 'model.safetensors': 'weight fixture',
                       'tokenizer.json': '{}', 'tokenizer_config.json': '{}'}.items():
        (checkpoint / name).write_text(data, encoding='utf-8')
    return teacher.freeze({'endpoint': 'http://127.0.0.1:30090', 'checkpoint': str(checkpoint),
                           'weight_version': 'teacher-v1'}, checkpoint)


def test_teacher_fingerprint_and_tokenizer_compatibility(frozen, tmp_path):
    root = tmp_path / 'teacher'
    assert teacher.freeze({key: frozen[key] for key in ('endpoint', 'checkpoint', 'weight_version')}, root) == frozen
    (root / 'model.safetensors').write_text('different weights', encoding='utf-8')
    changed = teacher.freeze({key: frozen[key] for key in ('endpoint', 'checkpoint', 'weight_version')}, root)
    assert changed['identity'] != frozen['identity']
    student = tmp_path / 'student'
    student.mkdir()
    (student / 'tokenizer.json').write_text('{"different":true}', encoding='utf-8')
    (student / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='tokenizer files'):
        teacher.freeze({key: frozen[key] for key in ('endpoint', 'checkpoint', 'weight_version')}, student)


@pytest.mark.parametrize('mutation', ['weights', 'tokenizer', 'version'])
def test_teacher_checks_live_identity(frozen, mutation):
    info = {'model_path': frozen['checkpoint'], 'tokenizer_path': frozen['checkpoint'],
            'weight_version': frozen['weight_version'], 'is_generation': True}
    assert teacher.check(frozen, transport=lambda url: info)['identity'] == frozen['identity']
    field = {'weights': 'model_path', 'tokenizer': 'tokenizer_path', 'version': 'weight_version'}[mutation]
    info[field] += '-changed'
    with pytest.raises(ValueError, match='changed'):
        teacher.check(frozen, transport=lambda url: info)


@pytest.mark.parametrize('mutation', [None, 'version', 'token', 'length', 'nan', 'positive'])
def test_teacher_scores_exact_input_positions(frozen, mutation):
    def transport(url, payload):
        assert url.endswith('/generate')
        assert payload['input_ids'] == [11, 12, 21, 22]
        assert payload['sampling_params']['max_new_tokens'] == 0
        assert payload['logprob_start_len'] == 0
        rows = [[None, 11], [-0.1, 12], [-0.2, 21], [-0.3, 22]]
        version = frozen['weight_version']
        if mutation == 'version':
            version = 'changed'
        elif mutation == 'token':
            rows[2][1] = 99
        elif mutation == 'length':
            rows.pop()
        elif mutation == 'nan':
            rows[2][0] = float('nan')
        elif mutation == 'positive':
            rows[2][0] = 0.1
        return {'meta_info': {'weight_version': version, 'input_token_logprobs': rows}}

    if mutation:
        with pytest.raises(ValueError):
            teacher.score_response(frozen, [11, 12], [21, 22], transport=transport)
    else:
        result = teacher.score_response(frozen, [11, 12], [21, 22], transport=transport)
        assert result['logprobs'] == [-0.2, -0.3]
        assert result['tokens_digest'] == digest([11, 12, 21, 22])


def opd_samples(frozen, tmp_path, monkeypatch):
    path = tmp_path / 'teacher.json'
    path.write_text(json.dumps(frozen), encoding='utf-8')
    monkeypatch.setenv('HARBORRL_MATH_ALGORITHM', 'opd')
    monkeypatch.setenv('HARBORRL_OPD_TEACHER', str(path))
    rows = samples()
    for index, sample in enumerate(rows):
        sample.teacher_log_probs = [-0.2 - index]
        sample.metadata['math_evidence'].update(algorithm='opd', teacher={
            'identity': frozen['identity'], 'weight_version': frozen['weight_version'],
            'logprobs': list(sample.teacher_log_probs), 'tokens_digest': digest(sample.tokens),
        })
    return rows


def test_opd_batch_preserves_token_scores_and_excludes_outcome_rewards(frozen, tmp_path, monkeypatch):
    rows = opd_samples(frozen, tmp_path, monkeypatch)
    args = SimpleNamespace(n_samples_per_prompt=2)
    batch = build_batch(args, rows)
    assert batch.algorithm == 'opd'
    assert batch.teacher_identity == frozen['identity']
    assert [unit.advantage for unit in batch.groups[0].trajectories] == [0., 0.]
    data = convert_samples_to_train_data(args, rows)
    assert data['raw_reward'] == [1., -1.]
    assert data['rewards'] == [0., 0.]
    assert data['teacher_log_probs'] == [[-0.2], [-1.2]]
    rollout_data_postprocess(args, 0, data)
    del data['teacher_log_probs']
    with pytest.raises(ValueError, match='every sample'):
        rollout_data_postprocess(args, 0, data)
    with pytest.raises(ValueError, match='teacher_identity'):
        validate_batch(replace(batch, teacher_identity=None))
    with pytest.raises(ValueError, match='reduction'):
        validate_batch(replace(batch, algorithm='grpo'))


@pytest.mark.parametrize('mutation', ['identity', 'version', 'tokens', 'score', 'length', 'nan', 'positive'])
def test_opd_rejects_teacher_evidence_drift(frozen, tmp_path, monkeypatch, mutation):
    rows = opd_samples(frozen, tmp_path, monkeypatch)
    scores = rows[0].metadata['math_evidence']['teacher']
    if mutation in ('identity', 'version', 'tokens'):
        key = {'identity': 'identity', 'version': 'weight_version', 'tokens': 'tokens_digest'}[mutation]
        scores[key] = 'changed'
    elif mutation == 'score':
        rows[0].teacher_log_probs = [-3.]
    else:
        values = [] if mutation == 'length' else [float('nan') if mutation == 'nan' else 0.1]
        rows[0].teacher_log_probs = values
        scores['logprobs'] = values
    with pytest.raises(ValueError):
        build_batch(SimpleNamespace(n_samples_per_prompt=2), rows)


def test_opd_config_and_audit_are_explicit(tmp_path):
    config = yaml.safe_load((ROOT / 'examples/math/train.yaml').read_text(encoding='utf-8'))
    config['tasks']['path'] = str(ROOT / 'examples/math/tasks.jsonl')
    config['algorithm'] = {'name': 'opd'}
    path = tmp_path / 'train.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    with pytest.raises(ValueError, match='teacher'):
        load_config(path)
    config['teacher'] = {'endpoint': 'http://127.0.0.1:30090', 'checkpoint': '/models/teacher',
                         'weight_version': 'teacher-v1'}
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    plan = launch_plan(load_config(path))
    assert plan['math']['algorithm']['id'] == 'opd'
    environment = run_environment(plan, tmp_path, path, path)
    assert environment['HARBORRL_OPD_TEACHER'] == str(tmp_path / 'config' / 'teacher.json')
    audit = effective_config(plan, environment, {}, {}, [])
    assert audit['update_semantics']['reward_mixing'] is False
    assert audit['update_semantics']['clipping'] is False
    with pytest.raises(ValueError, match='teacher'):
        load_config(path, ['algorithm.name=grpo'])


def test_production_opd_loss_matches_ragged_trajectory_mean_and_detaches(monkeypatch):
    torch = pytest.importorskip('torch')
    from harborrl.backends.slime_v032.opd import (
        compute_advantages_and_returns,
        loss_function,
    )

    current = torch.tensor([-0.5, -1.0, -0.7, -1.5, -0.2], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-0.6, -0.9, -0.8, -1.4, -0.3], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([-0.2, -1.1, -0.4, -1.0, -0.5], dtype=torch.float64, requires_grad=True)
    batch = {'unconcat_tokens': [], 'total_lengths': [5, 4], 'response_lengths': [3, 2],
             'rollout_mask_sums': [2, 2], 'loss_masks': [torch.tensor([1, 0, 1]), torch.tensor([1, 1])],
             'rollout_log_probs': list(old.split([3, 2])), 'teacher_log_probs': list(target.split([3, 2])),
             'rewards': [1000., -1000.]}
    monkeypatch.setitem(sys.modules, 'slime.backends.megatron_utils.loss', SimpleNamespace(
        get_log_probs_and_entropy=lambda *args, **kwargs: (None, {'log_probs': list(current.split([3, 2]))})))
    compute_advantages_and_returns(None, batch)
    assert not any(value.requires_grad for value in batch['advantages'])
    loss, metrics = loss_function(SimpleNamespace(n_samples_per_prompt=2), batch, current, None)
    # Two trajectories with two valid tokens each: each active token weighs 1/4.
    expected = -(current[[0, 2, 3, 4]] * (target - old).detach()[[0, 2, 3, 4]]).mean()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert current.grad.tolist() == pytest.approx([-0.1, 0., -0.1, -0.1, 0.05])
    assert old.grad is None and target.grad is None
    assert metrics['opd_weight_sum'].item() == pytest.approx(1.)
    assert metrics['opd_token_count'].item() == 4
