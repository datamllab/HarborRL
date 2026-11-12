"""Math evidence validation at the production backend boundary."""
import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from harborrl.backends.slime_v032 import math_rollout
from harborrl.backends.slime_v032.math_rollout import (
    build_batch,
    convert_samples_to_train_data,
)
from harborrl.config import ROOT, launch_plan, load_config
from harborrl.math.data import load_tasks, verify
from harborrl.trajectories.native import digest


def samples():
    task = load_tasks(ROOT / 'examples/math/tasks.jsonl')[0]
    output = []
    for index, response in enumerate([task['reference'], '999999']):
        evidence = {
            'tokens': [1, 2, 3], 'logprobs': [-0.5], 'response': response,
            'response_length': 1, 'group_key': 'group', 'policy_version': '1',
            'trajectory_id': f'trajectory-{index}', 'rollout_id': 0,
            'task_digest': digest(task), 'verification': verify(response, task['reference']),
            'truncated': False,
        }
        output.append(SimpleNamespace(
            tokens=[1, 2, 3], rollout_log_probs=[-0.5], response=response, response_length=1,
            loss_mask=[1], index=index,
            metadata={'math_task': task, 'task_digest': digest(task), 'math_evidence': evidence}))
    return output


def test_math_neutral_batch_and_standard_fields():
    args = SimpleNamespace(n_samples_per_prompt=2)
    batch = build_batch(args, samples())
    assert batch.policy_versions == {'1'}
    assert [unit.advantage for unit in batch.groups[0].trajectories] == [1., -1.]
    data = convert_samples_to_train_data(args, samples())
    assert data['rewards'] == [1., -1.]
    assert data['loss_masks'] == [[1], [1]]
    assert data['rollout_mask_sums'] == [1, 1]


@pytest.mark.parametrize('mutation', ['split', 'version', 'incomplete', 'tokens', 'reward'])
def test_math_rejects_invalid_evidence(mutation):
    rows = deepcopy(samples())
    if mutation == 'split':
        rows[0].metadata['math_task']['split'] = 'test'
    elif mutation == 'version':
        rows[0].metadata['math_evidence']['policy_version'] = '2'
    elif mutation == 'incomplete':
        rows.pop()
    elif mutation == 'tokens':
        rows[0].tokens = [2, 2, 3]
    else:
        rows[0].metadata['math_evidence']['verification']['reward'] = 9.
    with pytest.raises(ValueError):
        build_batch(SimpleNamespace(n_samples_per_prompt=2), rows)


def test_math_config_uses_shared_launcher_without_harbor():
    config = load_config(ROOT / 'examples/math/train.yaml')
    assert 'harbor' not in config
    plan = launch_plan(config)
    assert plan['training_command'] == ['bash', str(ROOT / 'harborrl/platform/slime_train.sh')]
    assert plan['environment']['HARBORRL_EXECUTION_BACKEND'] == 'text_math'
    with pytest.raises(ValueError, match='max_attempts'):
        load_config(ROOT / 'examples/math/train.yaml', ['sampling.max_attempts=2'])


def test_math_checkpoint_export_and_resume(tmp_path):
    from harborrl.platform.evidence import run_environment
    config = load_config(ROOT / 'examples/math/train.yaml', [f'training.resume={tmp_path.as_posix()}'])
    plan = launch_plan(config)
    assert plan['environment']['RESUME_LOAD'] == str(tmp_path.resolve())
    environment = run_environment(plan, tmp_path, tmp_path / 'tasks.json', tmp_path / 'prompts.jsonl')
    assert environment['SAVE_HF'] == str(tmp_path / 'hf' / 'rollout-{rollout_id}')
    with pytest.raises(ValueError, match='boolean'):
        load_config(ROOT / 'examples/math/train.yaml', ['training.export_hf=1'])


def test_math_seed_is_explicit_and_validated():
    path = ROOT / 'examples/math/train.yaml'
    assert 'HARBORRL_MATH_SEED' not in launch_plan(load_config(path))['environment']
    for seed in (0, 42, 2147483647):
        plan = launch_plan(load_config(path, [f'training.seed={seed}']))
        assert plan['environment']['HARBORRL_MATH_SEED'] == str(seed)
    for value in ('true', '-1', '2147483648', '1.5'):
        with pytest.raises(ValueError, match='training.seed'):
            load_config(path, [f'training.seed={value}'])


def test_math_batch_uses_task_verifier():
    rows = deepcopy(samples())
    for sample in rows:
        task = sample.metadata['math_task']
        task['verifier'] = 'gsm8k_numeric_v1'
        sample.metadata['task_digest'] = digest(task)
        evidence = sample.metadata['math_evidence']
        evidence['task_digest'] = digest(task)
        evidence['verification'] = verify(sample.response, task['reference'], task['verifier'])
    batch = build_batch(SimpleNamespace(n_samples_per_prompt=2), rows)
    assert [unit.advantage for unit in batch.groups[0].trajectories] == [0.5, -0.5]


@pytest.mark.parametrize('seed', [None, 0, 2147483647])
def test_generation_seed_tracks_persistent_sample_index(monkeypatch, seed):
    requests = []
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [10],
                                decode=lambda *_args, **_kwargs: '2')
    monkeypatch.setitem(sys.modules, 'slime.utils.processing_utils', SimpleNamespace(
        load_tokenizer=lambda *_args, **_kwargs: tokenizer))
    monkeypatch.delenv('HARBORRL_MATH_ALGORITHM', raising=False)
    monkeypatch.delenv('HARBORRL_PROCESS_REWARD', raising=False)
    monkeypatch.delenv('HARBORRL_MATH_SEED', raising=False)
    if seed is not None:
        monkeypatch.setenv('HARBORRL_MATH_SEED', str(seed))
    monkeypatch.setattr(math_rollout, '_pool', lambda _endpoint: ([], '0'))

    def request(_url, payload):
        requests.append(deepcopy(payload))
        return {'text': '2', 'meta_info': {'weight_version': '0',
                'output_token_logprobs': [[-0.5, 20]], 'finish_reason': {'type': 'stop'}}}

    monkeypatch.setattr(math_rollout, '_request', request)
    task = load_tasks(ROOT / 'examples/math/tasks.jsonl')[0]
    indices = [37, 38]
    group = [SimpleNamespace(index=index, prompt=task['prompt'],
             metadata={'math_task': task, 'task_digest': digest(task)},
             Status=SimpleNamespace(COMPLETED='completed', TRUNCATED='truncated')) for index in indices]
    args = SimpleNamespace(hf_checkpoint='unused', sglang_router_ip='localhost', sglang_router_port=1,
                           sglang_server_concurrency=1, rollout_max_response_len=10,
                           rollout_max_context_len=20, rollout_batch_size=1, rollout_seed=seed)
    generated = asyncio.run(math_rollout._generate(args, 8, [group]))
    assert len(generated) == 2
    assert [sample.metadata['math_evidence']['serving_input'] for sample in generated] == requests
    if seed is None:
        assert all('sampling_seed' not in payload['sampling_params'] for payload in requests)
    else:
        assert [payload['sampling_params']['sampling_seed'] for payload in requests] == [
            (seed + index) % 2**31 for index in indices]
