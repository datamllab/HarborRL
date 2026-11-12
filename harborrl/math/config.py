"""Plain math input to the shared Slime training launcher."""
import json
import re
import sys
from pathlib import Path

import yaml

from harborrl.backends.slime_v032.config import CONFIG_ENV
from harborrl.backends.slime_v032.versions import BACKEND_CONTRACT
from harborrl.config import ROOT
from harborrl.config.native import FIELDS
from harborrl.consistency.config import validate as validate_consistency
from harborrl.math.data import load_tasks
from harborrl.trajectories.native import finite

MATH_FIELDS = {key: FIELDS[key] for key in (
    'execution', 'model', 'training', 'sampling', 'deployment', 'output')}
MATH_FIELDS = {**MATH_FIELDS, 'tasks': {'path'}, 'consistency': FIELDS['consistency']}


def load_config(config, path, overrides=None, task_ids=None):
    if task_ids is not None:
        raise ValueError('text_math uses the locked train split, not Harbor task selectors')
    config['consistency'] = validate_consistency(config.get('consistency'))
    if not isinstance(config.get('training'), dict):
        raise TypeError('training must be a mapping')
    config['training'].setdefault('backend_contract', BACKEND_CONTRACT)
    config['training'].setdefault('resume', None)
    config['training'].setdefault('export_hf', False)
    config['training'].setdefault('seed', None)
    config.setdefault('algorithm', {'name': 'grpo'})
    fields = {**MATH_FIELDS, 'training': MATH_FIELDS['training'] | {'resume', 'export_hf', 'seed'},
              'algorithm': {'name'}}
    if 'teacher' in config:
        fields['teacher'] = {'endpoint', 'checkpoint', 'weight_version'}
    if 'dapo' in config:
        from harborrl.math.dapo import FIELDS as DAPO_FIELDS
        fields['dapo'] = DAPO_FIELDS
    if 'process_reward' in config:
        from harborrl.math.prm import FIELDS as PRM_FIELDS
        fields['process_reward'] = PRM_FIELDS
    if set(config) != set(fields) | {'schema_version'}:
        raise ValueError('text_math requires exactly the documented schema 2 sections')
    for section, keys in fields.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f'{section} requires exactly {sorted(keys)}')
    for override in overrides or ():
        key, sep, value = override.partition('=')
        section, _, field = key.partition('.')
        if not sep or field not in fields.get(section, ()):
            raise ValueError(f'unknown text_math override: {key}')
        config[section][field] = yaml.safe_load(value)
    config['consistency'] = validate_consistency(config['consistency'])
    algorithm = config['algorithm']['name']
    if algorithm not in ('grpo', 'opd', 'dapo'):
        raise ValueError('text_math algorithm.name must be grpo, opd or dapo')
    if (algorithm == 'opd') != ('teacher' in config):
        raise ValueError('Only OPD requires a teacher section')
    if (algorithm == 'dapo') != ('dapo' in config):
        raise ValueError('Only DAPO requires a dapo section')
    if 'process_reward' in config:
        from harborrl.math.prm import validate_config
        if algorithm != 'grpo':
            raise ValueError('process_reward currently requires algorithm.name: grpo')
        validate_config(config['process_reward'])
        checkpoint = Path(config['process_reward']['checkpoint']).expanduser()
        config['process_reward']['checkpoint'] = str((Path(path).parent / checkpoint).resolve())
    if algorithm == 'opd':
        from harborrl.math.teacher import validate_config
        validate_config(config['teacher'])
        checkpoint = Path(config['teacher']['checkpoint']).expanduser()
        config['teacher']['checkpoint'] = str((Path(path).parent / checkpoint).resolve())
    if type(config['training']['export_hf']) is not bool:
        raise ValueError('training.export_hf must be a boolean')
    seed = config['training']['seed']
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
        raise ValueError('training.seed must be null or an integer in 0..2147483647')
    if config['execution']['backend'] != 'text_math':
        raise ValueError('execution.backend must be text_math')
    if config['training']['backend_contract'] != BACKEND_CONTRACT:
        raise ValueError('text_math requires the pinned Slime backend')
    for section, field in [('tasks', 'path'), ('model', 'checkpoint'),
                           ('model', 'reference'), ('output', 'root'), ('training', 'resume')]:
        value = config[section][field]
        if section == 'training' and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{section}.{field} must be a path')
        config[section][field] = str((Path(path).parent / Path(value).expanduser()).resolve())
    for section, keys in [('sampling', FIELDS['sampling']),
                          ('deployment', FIELDS['deployment'] - {'layout'}),
                          ('training', {'num_rollout', 'save_interval'})]:
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] <= 0:
                raise ValueError(f'{section}.{key} must be a positive integer')
    rate = config['training']['learning_rate']
    if not finite(rate) or rate <= 0:
        raise ValueError('training.learning_rate must be positive and finite')
    sampling, deployment = config['sampling'], config['deployment']
    if algorithm == 'dapo':
        from harborrl.math.dapo import validate
        validate(config['dapo'], sampling['max_tokens'])
    if sampling['group_size'] < 2 or sampling['max_tokens'] >= sampling['max_context']:
        raise ValueError('text_math requires complete groups and max_tokens below max_context')
    if sampling['max_attempts'] != 1:
        raise ValueError('text_math requires max_attempts: 1; DAPO refills use max_refill_batches')
    if deployment['layout'] not in ('split', 'colocate'):
        raise ValueError('layout must be split or colocate')
    if deployment['layout'] == 'split' and deployment['actor_gpus'] + deployment['rollout_gpus'] > deployment['num_gpus']:
        raise ValueError('GPU budget exceeded')
    if deployment['layout'] == 'colocate' and not deployment['actor_gpus'] == deployment['rollout_gpus'] == deployment['num_gpus']:
        raise ValueError('colocate GPU counts must match')
    if (deployment['actor_gpus'] % deployment['actor_tensor_parallel_size']
            or deployment['rollout_gpus'] % deployment['rollout_gpus_per_engine']):
        raise ValueError('GPU counts must be divisible by tensor parallel sizes')
    if not isinstance(config['model']['args_file'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', config['model']['args_file']):
        raise ValueError('model.args_file must name a Slime model preset')
    if config['consistency']['mode'] != 'off':
        raise ValueError('text_math currently requires consistency.mode: off; Harbor sidecars are not math evidence')
    rows = load_tasks(config['tasks']['path'])
    if not any(row['split'] == 'train' for row in rows):
        raise ValueError('text_math requires a nonempty train split')
    return config


def launch_plan(config):
    environment = {
        name: config[section][field]
        for name, (section, field) in CONFIG_ENV.items()
        if section in config and field in config[section]
    }
    environment.update({
        'TRAIN_PYTHON': sys.executable,
        'HARBORRL_EXECUTION_BACKEND': 'text_math',
        'HARBORRL_MATH_ALGORITHM': config['algorithm']['name'],
        'HARBORRL_NATIVE_SLIME_CONTRACT': 'slime-v032',
        'HARBORRL_CONSISTENCY': json.dumps(config['consistency'], sort_keys=True),
        'HARBORRL_CONSISTENCY_MODE': 'off',
        'SLIME_RAY_PLACEMENT_GPU_PROBE': '1',
        'HARBORRL_SKIP_GLOBAL_CLEANUP': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:2048',
    })
    if config['training']['resume']:
        environment['RESUME_LOAD'] = config['training']['resume']
    if config['training']['seed'] is not None:
        environment['HARBORRL_MATH_SEED'] = config['training']['seed']
    if config['algorithm']['name'] == 'dapo':
        environment['HARBORRL_DAPO'] = json.dumps(config['dapo'], sort_keys=True)
    return {
        'config': config,
        'math': {'algorithm': {'id': config['algorithm']['name']}, 'harness': {'id': 'text_math'}},
        'environment': {key: str(value) for key, value in environment.items()},
        'training_command': ['bash', str(ROOT / 'harborrl/platform/slime_train.sh')],
        'command': [sys.executable, '-m', 'harborrl.platform.native_train'],
        'required_services': ['GPU Ray / SGLang'],
        'layout': {'mode': config['deployment']['layout'], 'offload': config['deployment']['layout'] == 'colocate'},
    }


def materialize(plan, root):
    from slime.utils.processing_utils import load_tokenizer

    from harborrl.trajectories.native import digest, publish

    config = plan['config']
    rows = load_tasks(config['tasks']['path'])
    tokenizer = load_tokenizer(config['model']['checkpoint'], trust_remote_code=True)
    normalized = root / 'config'
    normalized.mkdir(parents=True)
    if config['algorithm']['name'] == 'opd':
        from harborrl.math.teacher import check, freeze
        teacher = freeze(config['teacher'], config['model']['checkpoint'])
        check(teacher)
        publish(normalized / 'teacher.json', teacher)
    if 'process_reward' in config:
        from harborrl.math import prm
        scorer = prm.freeze(config['process_reward'])
        prm.check(scorer)
        publish(normalized / 'process_reward.json', scorer)
    catalog = normalized / 'math_tasks.json'
    publish(catalog, {'tasks': rows, 'digest': digest(rows)})
    prompts = normalized / 'math_train.jsonl'
    output = []
    for row in rows:
        if row['split'] != 'train':
            continue
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['prompt']}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        output.append({'task': prompt, 'metadata': {'math_task': row, 'task_digest': digest(row)}})
    prompts.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in output), encoding='utf-8')
    return catalog, prompts
