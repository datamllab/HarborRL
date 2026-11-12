"""Strict native schema 2; local paths resolve relative to the config file."""
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from harborrl.consistency.config import DEFAULTS as CONSISTENCY_DEFAULTS
from harborrl.consistency.config import validate as validate_consistency

FIELDS = {
    'consistency': set(CONSISTENCY_DEFAULTS),
    'execution': {'backend'}, 'tasks': {'catalog'}, 'harness': {'name', 'profile'},
    'harbor': {'python', 'version', 'workers', 'runner_transport', 'environment'},
    'gateway': {'host', 'port', 'advertised_url', 'tokenizer_digest', 'template_digest', 'audited_raw_logprobs'},
    'model': {'checkpoint', 'reference', 'args_file'},
    'training': {'num_rollout', 'learning_rate', 'save_interval', 'backend_contract', 'seed', 'export_hf'},
    'sampling': {'group_size', 'groups_per_batch', 'max_attempts', 'max_tokens', 'max_context'},
    'deployment': {'layout', 'num_gpus', 'actor_gpus', 'rollout_gpus', 'actor_tensor_parallel_size', 'rollout_gpus_per_engine'},
    'output': {'root'},
}
OPTIONAL = {'harbor': {'runner_transport', 'environment'}, 'training': {'seed', 'export_hf'}}


class SelectedConfig(dict):
    def __init__(self, config, task_ids):
        super().__init__(config)
        self.task_ids = tuple(task_ids)


def load_config(path, overrides=None, task_ids=None):
    """Load one schema-2 YAML file and apply strict ``section.key=value`` overrides."""
    path = Path(path).resolve()
    try:
        config = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f'configuration YAML: {exc}') from exc
    if not isinstance(config, dict):
        raise TypeError('configuration must be a mapping')
    if config.get('schema_version') != 2:
        raise ValueError('native configuration requires schema_version: 2')
    if config.get('execution', {}).get('backend') == 'text_math':
        from harborrl.math.config import load_config as load_math_config
        return load_math_config(config, path, overrides, task_ids)
    validate(config, path)
    for override in overrides or ():
        key, separator, value = override.partition('=')
        section_name, _, field = key.partition('.')
        section = config.get(section_name)
        if (not separator or field not in FIELDS.get(section_name, ())
                or not isinstance(section, dict)):
            raise ValueError(f'unknown native override: {key}')
        try:
            section[field] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f'{key}: invalid YAML override') from exc
    config = validate(config, path)
    validate_files(config)
    if task_ids is None:
        return config
    if (not isinstance(task_ids, (list, tuple)) or not task_ids
            or any(not isinstance(task_id, str) or not task_id.strip()
                   for task_id in task_ids)):
        raise ValueError('task selectors must be nonempty strings')
    return SelectedConfig(config, task_ids)


def validate(config, path):
    if not isinstance(config, dict):
        raise TypeError('configuration must be a mapping')
    config['consistency'] = validate_consistency(config.get('consistency'))
    if set(config) != set(FIELDS) | {'schema_version'} or type(config['schema_version']) is not int or config['schema_version'] != 2:
        raise ValueError('native schema requires exactly the documented schema 2 sections')
    if not isinstance(config['training'], dict):
        raise TypeError('training must be a mapping')
    config['training'].setdefault(
        'backend_contract', 'slime-v0.3.2-native-v1')
    config['training'].setdefault('export_hf', False)
    for section, fields in FIELDS.items():
        actual = set(config[section]) if isinstance(config[section], dict) else set()
        if not isinstance(config[section], dict) or not fields - OPTIONAL.get(section, set()) <= actual <= fields:
            raise ValueError(f'{section} requires exactly {sorted(fields)}')
    for section, key in [('tasks','catalog'), ('harness','profile'), ('harbor','python'),
                         ('model','checkpoint'), ('model','reference'), ('output','root')]:
        value = config[section][key]
        if not isinstance(value, str) or not value:
            raise ValueError(f'{section}.{key} must be a path')
        expanded = Path(value).expanduser()
        absolute = expanded if expanded.is_absolute() else Path(path).parent / expanded
        config[section][key] = str(Path(os.path.abspath(absolute)))
    for section, keys in [('sampling', FIELDS['sampling']), ('deployment', FIELDS['deployment'] - {'layout'}),
                          ('training', {'num_rollout','save_interval'}), ('gateway', {'port'})]:
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] <= 0:
                raise ValueError(f'{section}.{key} must be a positive integer')
    from harborrl.backends.slime_v032.versions import CONFIG_CONTRACTS
    from harborrl.trajectories.native import finite
    if (not isinstance(config['training']['backend_contract'], str)
            or config['training']['backend_contract'] not in CONFIG_CONTRACTS):
        raise ValueError(f"training.backend_contract must be one of {sorted(CONFIG_CONTRACTS)}")
    if not finite(config['training']['learning_rate']) or config['training']['learning_rate'] <= 0:
        raise ValueError('training.learning_rate must be positive and finite')
    seed = config['training'].get('seed')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
        raise ValueError('training.seed must be null or an integer in 0..2147483647')
    if type(config['training']['export_hf']) is not bool:
        raise ValueError('training.export_hf must be a boolean')
    g, d, s = config['gateway'], config['deployment'], config['sampling']
    if type(g['audited_raw_logprobs']) is not bool or not 0 < g['port'] < 65536:
        raise ValueError('gateway.audited_raw_logprobs must be boolean and gateway.port must be in 1..65535')
    for key in ('host','tokenizer_digest','template_digest'):
        if not isinstance(g[key], str) or not g[key].strip():
            raise ValueError(f'gateway.{key} required')
    if not isinstance(g['advertised_url'], str):
        raise TypeError('gateway.advertised_url must be a string')
    advertised = urlsplit(g['advertised_url'])
    if (advertised.scheme not in ('http', 'https') or not advertised.hostname
            or advertised.username or advertised.password or advertised.query or advertised.fragment
            or advertised.path not in ('', '/') or advertised.port != g['port']):
        raise ValueError('advertised gateway must be a plain origin and match listener port')
    if s['group_size'] < 2:
        raise ValueError('sampling.group_size must be at least 2')
    if s['max_tokens'] >= s['max_context']:
        raise ValueError('sampling.max_tokens must be below sampling.max_context')
    if not isinstance(d['layout'], str) or d['layout'] not in ('split','colocate'):
        raise ValueError('layout must be split or colocate')
    if d['layout'] == 'split' and d['actor_gpus'] + d['rollout_gpus'] > d['num_gpus']:
        raise ValueError('GPU budget exceeded: deployment.actor_gpus + deployment.rollout_gpus > deployment.num_gpus')
    if d['layout'] == 'colocate' and not d['actor_gpus'] == d['rollout_gpus'] == d['num_gpus']:
        raise ValueError('colocate GPU counts must match')
    if d['actor_gpus'] % d['actor_tensor_parallel_size'] or d['rollout_gpus'] % d['rollout_gpus_per_engine']:
        raise ValueError('deployment GPU counts must be divisible by their tensor parallel sizes')
    preset = config['model']['args_file']
    if not isinstance(preset,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',preset):
        raise ValueError('model.args_file must name a Slime model preset; SLIME_DIR is checked by doctor')
    for key in ('tokenizer_digest', 'template_digest'):
        if not re.fullmatch(r'[0-9a-f]{64}', g[key]):
            raise ValueError(f'gateway.{key} must be a SHA-256 digest')
    harbor = config['harbor']
    transport = harbor.get('runner_transport', 'ssh')
    if transport not in ('ssh', 'local'):
        raise ValueError('harbor.runner_transport must be ssh or local')
    environment = harbor.get('environment', {'runtime': 'docker'})
    if not isinstance(environment, dict) or environment.get('runtime') not in ('docker', 'sandbox'):
        raise ValueError('harbor.environment.runtime must be docker or sandbox')
    expected = {'runtime', 'config'} if environment['runtime'] == 'sandbox' else {'runtime'}
    if set(environment) != expected:
        raise ValueError('invalid harbor.environment fields')
    if environment['runtime'] == 'sandbox':
        if not isinstance(environment['config'], str) or not environment['config']:
            raise ValueError('sandbox config path required')
        environment['config'] = str((Path(path).parent / environment['config']).resolve())
    workers = harbor['workers']
    if (not isinstance(workers, list) or (transport == 'ssh' and not workers) or any(
            not isinstance(worker, str) or not re.fullmatch(r'[A-Za-z0-9_.:+@-]+', worker) or worker.startswith('-')
            for worker in workers)):
        raise ValueError('harbor.workers must contain nonempty SSH destinations')
    if transport == 'local' and workers:
        raise ValueError('local runner requires empty harbor.workers')
    from harborrl.math_registry import select_math, validate_combination
    validate_combination(select_math(config), config)
    return config


def validate_files(config):
    """Check harness profile files separately from pure schema validation."""
    from harborrl.math_registry import select_math
    from harborrl.trajectories.native import read_json
    harness = select_math(config).harness
    try:
        harness.adapter.validate_profile(
            read_json(config['harness']['profile']),
            config['gateway']['advertised_url'],
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f'harness.profile: {exc}') from exc
    return config


def catalog(config):
    """Compatibility entry point for task lock validation."""
    from harborrl.tasks.catalog import catalog as inspect_catalog
    return inspect_catalog(config)


def launch_plan(config):
    """Compatibility entry point for backend launch mapping."""
    if config['execution']['backend'] == 'text_math':
        from harborrl.math.config import launch_plan as build_math_plan
        return build_math_plan(config)
    from harborrl.backends.slime_v032.config import launch_plan as build_plan
    return build_plan(config)
