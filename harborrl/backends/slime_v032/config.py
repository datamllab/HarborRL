"""Schema-2 to official Slime launch mapping; no live probes."""
import json
import os
import sys

CONFIG_ENV = {
    "HF_CKPT": ("model", "checkpoint"),
    "REF_LOAD": ("model", "reference"),
    "MODEL_ARGS_FILE": ("model", "args_file"),
    "MODEL_TAG": ("model", "args_file"),
    "NUM_ROLLOUT": ("training", "num_rollout"),
    "SAVE_INTERVAL": ("training", "save_interval"),
    "LR": ("training", "learning_rate"),
    "ROLLOUT_BATCH_SIZE": ("sampling", "groups_per_batch"),
    "N_SAMPLES": ("sampling", "group_size"),
    "ROLLOUT_MAX_RESPONSE_LEN": ("sampling", "max_tokens"),
    "ROLLOUT_MAX_CONTEXT_LEN": ("sampling", "max_context"),
    "NUM_GPUS": ("deployment", "num_gpus"),
    "ACTOR_GPUS": ("deployment", "actor_gpus"),
    "ROLLOUT_GPUS": ("deployment", "rollout_gpus"),
    "TP_SIZE": ("deployment", "actor_tensor_parallel_size"),
    "ROLLOUT_NUM_GPUS_PER_ENGINE": ("deployment", "rollout_gpus_per_engine"),
    "HARBORRL_GPU_LAYOUT": ("deployment", "layout"),
    "HARBORRL_NATIVE_RUNNER_PYTHON": ("harbor", "python"),
    "HARBORRL_NATIVE_PROFILE": ("harness", "profile"),
    "HARBORRL_NATIVE_CATALOG": ("tasks", "catalog"),
    "HARBORRL_NATIVE_GATEWAY_HOST": ("gateway", "host"),
    "HARBORRL_NATIVE_GATEWAY_PORT": ("gateway", "port"),
    "HARBORRL_NATIVE_GATEWAY_URL": ("gateway", "advertised_url"),
    "HARBORRL_NATIVE_TOKENIZER_DIGEST": ("gateway", "tokenizer_digest"),
    "HARBORRL_NATIVE_TEMPLATE_DIGEST": ("gateway", "template_digest"),
    "HARBORRL_NATIVE_MAX_ATTEMPTS": ("sampling", "max_attempts"),
    "RUNS_ROOT": ("output", "root"),
}


def launch_plan(config):
    from harborrl.backends.slime_v032.versions import runtime_contract_id
    from harborrl.config import ROOT
    from harborrl.config.native import validate, validate_files
    from harborrl.math_registry import select_math
    from harborrl.tasks.catalog import catalog
    validate(config, ROOT / "config.yaml")
    validate_files(config)
    selection = select_math(config)
    entries = catalog(config)
    runtime = config['harbor'].get('environment', {'runtime': 'docker'})
    sandbox = None
    if runtime['runtime'] == 'sandbox':
        from harborrl.rollout.harbor_job.sandbox_config import (
            load_sandbox,
            task_binding,
        )
        from harborrl.trajectories.native import read_json
        sandbox = load_sandbox(runtime['config'])
        profile = read_json(config['harness']['profile'])
        if sandbox['installation'] and sandbox['installation']['version'] != profile['cli_version']:
            raise ValueError('sandbox installation version differs from harness profile')
        budget = sum(profile[k] for k in ('agent_timeout_sec', 'setup_timeout_sec', 'verifier_timeout_sec'))
        if sandbox['lifecycle_minutes'] * 60 <= budget + sandbox['create_timeout_s'] + 2 * sandbox['cleanup_timeout_s']:
            raise ValueError('sandbox lifecycle must cover setup, agent, verifier and cleanup budgets')
        for entry in entries:
            task_binding(sandbox, entry['path'])
    c = config
    worker_overrides = os.environ.get('HARBORRL_NATIVE_TASK_WORKER_OVERRIDES', '{}')
    try:
        overrides = json.loads(worker_overrides)
        valid_workers = set(c['harbor']['workers'])
        if (not isinstance(overrides, dict) or any(not isinstance(key, str) or not isinstance(value, str) or value not in valid_workers
                                                   for key, value in overrides.items())):
            raise ValueError
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError('invalid HARBORRL_NATIVE_TASK_WORKER_OVERRIDES') from exc
    environment = {
        **{name: c[section][field] for name, (section, field) in CONFIG_ENV.items()},
        'DATASET': selection.task.id, 'ALGO': selection.algorithm.id,
        'HARNESS_OPTION': f"native-{selection.harness.id.replace('_', '-')}",
        'TRAIN_PYTHON': sys.executable,
        'HARBORRL_STRUCTURED_LAUNCH': '1', 'HARBORRL_VERIFY_POLICY_POOL': '1',
        'HARBORRL_SKIP_GLOBAL_CLEANUP': '1', 'SLIME_RAY_PLACEMENT_GPU_PROBE': '1',
        'PYTORCH_CUDA_ALLOC_CONF': ('max_split_size_mb:2048' if c['deployment']['layout'] == 'colocate'
                                    else 'max_split_size_mb:2048,expandable_segments:True'),
        'HARBORRL_NATIVE_ROLLOUT': '1',
        'HARBORRL_NATIVE_SLIME_CONTRACT': runtime_contract_id(c['training']['backend_contract']),
        'HARBORRL_NATIVE_RUNNER_WORKERS': json.dumps(c['harbor']['workers']),
        'HARBORRL_NATIVE_TASK_WORKER_OVERRIDES': json.dumps(overrides, sort_keys=True),
        'HARBORRL_NATIVE_AUDITED_LOGPROBS': str(int(c['gateway']['audited_raw_logprobs'])),
        'HARBORRL_NATIVE_DRAIN_TIMEOUT': '30',
        'WANDB_ENABLE': '0',
        'HARBORRL_CONSISTENCY': json.dumps(c['consistency'], sort_keys=True),
        'HARBORRL_CONSISTENCY_MODE': c['consistency']['mode'],
    }
    environment['HARBORRL_NATIVE_RUNNER_TRANSPORT'] = c['harbor'].get('runner_transport', 'ssh')
    if c['training'].get('seed') is not None:
        environment['HARBORRL_TRAINING_SEED'] = c['training']['seed']
    if sandbox is not None:
        environment['HARBORRL_NATIVE_SANDBOX'] = json.dumps(sandbox, sort_keys=True)
    return {'config': config, 'math': selection.asdict(),
            'command': [sys.executable, '-m', 'harborrl.platform.native_train'],
            'environment': {key: str(value) for key, value in environment.items()},
            'training_command': ['bash', str(ROOT / 'harborrl/platform/slime_train.sh')],
            'required_services': ['GPU Ray / SGLang / Messages Gateway',
                                  'Sandbox / Harbor runner' if sandbox else 'external CPU Harbor Docker workers'],
            'launcher': f"native {selection.harness.display_name} → Messages Gateway → "
                        f"{selection.task.display_name} verifier → "
                        f"{selection.algorithm.display_name}",
            'layout': {'mode': config['deployment']['layout'], 'offload': config['deployment']['layout']=='colocate'}}


def field_usage():
    """Account for every schema-2 field, including non-environment consumers."""
    usage = {}
    from harborrl.consistency.config import DEFAULTS
    usage.update({f'consistency.{key}': ['environment:HARBORRL_CONSISTENCY'] for key in DEFAULTS})
    for name, (section, field) in CONFIG_ENV.items():
        usage.setdefault(f"{section}.{field}", []).append(f"environment:{name}")
    usage.update({
        "execution.backend": ["schema:harbor_job"],
        "harness.name": ["schema:claude_code", "runner:Claude Code"],
        "harbor.version": ["preflight:worker runtime version"],
        "training.backend_contract": ["environment:HARBORRL_NATIVE_SLIME_CONTRACT", "preflight:backend hooks"],
        "training.seed": ["environment:HARBORRL_TRAINING_SEED"],
        "training.export_hf": ["run_environment:SAVE_HF"],
        "harbor.workers": ["environment:HARBORRL_NATIVE_RUNNER_WORKERS", "preflight:worker probe"],
        "harbor.runner_transport": ["environment:HARBORRL_NATIVE_RUNNER_TRANSPORT"],
        "harbor.environment": ["environment:HARBORRL_NATIVE_SANDBOX", "runner:environment"],
        "gateway.audited_raw_logprobs": ["environment:HARBORRL_NATIVE_AUDITED_LOGPROBS"],
    })
    return usage
