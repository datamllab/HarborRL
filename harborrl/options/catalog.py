"""JSON-safe option metadata. Existing validators remain authoritative."""
from copy import deepcopy
from importlib.resources import files

import yaml

from harborrl.config import ROOT
from harborrl.config.native import FIELDS, OPTIONAL
from harborrl.deployment import schema
from harborrl.math_registry import MATH_REGISTRY

from .presets import PRESETS

TRAINING_FIELDS = {
    "consistency.mode": ("string", "off, audit or enforce"),
    "consistency.max_abs_delta": ("number_or_null", "Explicit deployment maximum token log-probability delta"),
    "consistency.max_mean_abs_delta": ("number_or_null", "Explicit deployment mean token delta"),
    "consistency.bad_token_delta": ("number_or_null", "Threshold identifying a bad token"),
    "consistency.max_bad_fraction": ("number", "Allowed bad-token fraction between zero and one"),
    "consistency.worst_tokens": ("integer", "Maximum diagnostic tokens, zero through 32"),
    "execution.backend": ("string", "Task execution backend; currently harbor_job"),
    "tasks.catalog": ("path", "Existing JSON/JSONL Harbor task lock; resolved on the controller"),
    "harness.name": ("string", "Registered agent harness; currently claude_code"),
    "harness.profile": ("path", "Pinned harness profile JSON on the controller"),
    "harbor.python": ("path", "Runner interpreter on the local runner or SSH worker"),
    "harbor.version": ("string", "Harbor version supported by the selected harness"),
    "harbor.workers": ("array", "Ordered SSH destinations; empty with a local runner"),
    "harbor.runner_transport": ("string", "local or ssh; local requires no workers"),
    "harbor.environment": ("object", "Docker runtime, or Sandbox runtime with a config file reference"),
    "gateway.host": ("string", "Gateway listener address, distinct from its advertised origin"),
    "gateway.port": ("integer", "Gateway listener port, 1 through 65535"),
    "gateway.advertised_url": ("string", "Plain HTTP(S) origin reachable by runners; port must match"),
    "gateway.tokenizer_digest": ("string", "SHA-256 of the actual tokenizer fingerprint; never a placeholder"),
    "gateway.template_digest": ("string", "SHA-256 of the actual chat-template fingerprint"),
    "gateway.audited_raw_logprobs": ("boolean", "Raw serving probability audit, required by GRPO"),
    "model.checkpoint": ("path", "Student model checkpoint; must be accessible on the GPU target"),
    "model.reference": ("path", "Reference checkpoint accessible on the GPU target"),
    "model.args_file": ("string", "Supported Slime preset name; currently the qwen prefix"),
    "training.num_rollout": ("integer", "Positive number of rollout batches"),
    "training.learning_rate": ("number", "Positive finite learning rate"),
    "training.seed": ("number_or_null", "Sampling seed shared by every group; null keeps sampling nondeterministic"),
    "training.export_hf": ("boolean", "Also export an HF checkpoint next to the training checkpoints"),
    "training.save_interval": ("integer", "Positive checkpoint interval in rollout batches"),
    "training.backend_contract": ("string", "Pinned backend/algorithm contract"),
    "sampling.group_size": ("integer", "Samples per prompt, at least two for GRPO"),
    "sampling.groups_per_batch": ("integer", "Positive number of complete groups per batch"),
    "sampling.max_attempts": ("integer", "Bounded invalid-attempt retries; not low-reward retries"),
    "sampling.max_tokens": ("integer", "Positive response limit below max_context"),
    "sampling.max_context": ("integer", "Total context token limit"),
    "deployment.layout": ("string", "split or colocate; controls actor/serving GPU allocation"),
    "deployment.num_gpus": ("integer", "Total positive GPU budget, not a memory-fit guarantee"),
    "deployment.actor_gpus": ("integer", "Actor GPU count; must be divisible by actor TP"),
    "deployment.rollout_gpus": ("integer", "Serving GPU count; must be divisible by engine TP"),
    "deployment.actor_tensor_parallel_size": ("integer", "Positive actor tensor-parallel size"),
    "deployment.rollout_gpus_per_engine": ("integer", "Positive serving tensor-parallel size"),
    "output.root": ("path", "Run output root; service mode must match shared storage.run_root"),
}

DEPLOYMENT_GROUPS = {
    "manager": schema.MANAGER_FIELDS, "gpu_target": schema.GPU_FIELDS,
    "storage": schema.STORAGE_FIELDS, "rollout": schema.ROLLOUT_FIELDS,
    "rollout.workers[]": schema.WORKER_FIELDS, "gpu_target.ssh": schema.SSH_FIELDS,
    "rollout.workers[].ssh": schema.SSH_FIELDS, "connectivity": schema.CONNECTIVITY_FIELDS,
}
DEPLOYMENT_FIELDS = {
    "id": ("string", "Short role identifier"),
    "mode": ("string", "Manager mode or shared storage mode"),
    "bind": ("string", "Manager listener address"),
    "gateway_shards": ("array", "Static gateway shard table; omit to let auto routing own the shard set"),
    "gateway_routing": ("string", "Shard routing rule: modulo, worker_affinity, or auto"),
    "sandboxes_per_shard": ("integer", "Concurrent sandboxes per gateway before auto routing grows one"),
    "worker_shards": ("object", "Worker id (or local) to gateway shard name; worker_affinity only"),
    "gateway_hosts": ("array", "Ssh destinations (user@host) auto growth fills first; empty grows locally"),
    "shard_python": ("path", "Interpreter on gateway_hosts for remotely grown shards; default runner_python"),
    "hub_port": ("integer", "Hub control-plane listener port; hub mode only"),
    "hub_url": ("string", "Hub control-plane origin reachable from every puller node"),
    "hub_pull_interval": ("integer", "Seconds between puller polls in hub mode"),
    "gpu_pool": ("array", "Declared GPU nodes the hub schedules runs onto"),
    "max_concurrent_runs": ("integer", "Hub mode: how many runs may execute at once"),
    "run_queue_size": ("integer", "Hub mode: pending run queue capacity"),
    "provision": ("object", "up/down commands that obtain and release the GPU machine; site-specific"),
    "control_port": ("integer", "Control listener port; different from gateway_port"),
    "control_url": ("string", "Control origin reachable from the GPU target"),
    "gateway_port": ("integer", "Gateway listener port"),
    "gateway_url": ("string", "Gateway origin reachable from runners or the configured relay"),
    "run_token_file": ("path", "Manager run-token file reference; never the secret value"),
    "source_dir": ("path", "Absolute source checkout path on this role's machine"),
    "controller_python": ("path", "Absolute interpreter path on the controller"),
    "transport": ("string", "GPU launch transport: local or ssh"),
    "ssh": ("object", "SSH specification; GPU local transport requires null"),
    "train_python": ("path", "Absolute interpreter path on the GPU target"),
    "environment_file": ("path_or_null", "Controlled JSON environment file on the GPU target"),
    "expected_gpus": ("integer", "Expected target GPU inventory, at least the requested budget"),
    "run_root": ("path", "Absolute shared run root visible to all participating roles"),
    "manager_writable": ("boolean", "Declared shared-root access; verified by doctor"),
    "gpu_writable": ("boolean", "Declared GPU shared-root access; verified by doctor"),
    "worker_writable": ("boolean", "Declared worker shared-root access; verified by doctor"),
    "provider": ("string", "Rollout placement/provider; not a dataset or algorithm"),
    "workers": ("array", "Ordered worker role specifications; empty for local Sandbox runners"),
    "runner_python": ("path", "Absolute interpreter path on the worker"),
    "docker": ("boolean", "Worker Docker requirement"),
    "task_roots": ("array", "Absolute task roots visible on this worker"),
    "host": ("string", "SSH alias/destination, not an automatically verified host"),
    "user": ("string", "Optional SSH username"),
    "port": ("integer", "Optional SSH port"),
    "identity": ("path", "Optional identity-file reference; no private-key contents"),
    "command_timeout_s": ("integer", "Positive remote command timeout"),
    "ssh_timeout_s": ("integer", "Positive SSH connection timeout"),
    "http_timeout_s": ("integer", "Positive HTTP check timeout"),
    "gpu_to_controller": ("boolean", "Declared GPU-to-controller connectivity requirement"),
    "worker_to_gateway": ("boolean", "Declared runner-to-gateway connectivity requirement"),
    "manager_to_gpu_inference": ("boolean", "Declared controller-to-serving connectivity requirement"),
}


def _describe(spec, attributes):
    record = {}
    for key in ("id", "display_name", "version", "evidence", "limitations", *attributes):
        value = getattr(spec, key)
        record[key] = sorted(value) if isinstance(value, frozenset) else list(value) if isinstance(value, tuple) else value
    return record


def dataset_metadata():
    return yaml.safe_load(files("harborrl.options").joinpath("resources/datasets.yaml").read_text(encoding="utf-8"))


def build_catalog():
    training = {}
    for section, fields in sorted(FIELDS.items()):
        for name in sorted(fields):
            key = f"{section}.{name}"
            kind, description = TRAINING_FIELDS[key]
            training[key] = {"type": kind, "description": description,
                             "required": name not in OPTIONAL.get(section, set())}
    deployment = {}
    for section, fields in sorted(DEPLOYMENT_GROUPS.items()):
        for name in sorted(fields):
            kind, description = DEPLOYMENT_FIELDS[name]
            deployment[f"{section}.{name}"] = {"type": kind, "description": description}
    for key, values in {"manager.mode": ("embedded", "service"), "storage.mode": schema.STORAGE_MODES,
                        "rollout.provider": schema.ROLLOUT_PROVIDERS, "gpu_target.transport": schema.TRANSPORTS}.items():
        deployment[key]["values"] = list(values)
    datasets = deepcopy(dataset_metadata()["datasets"])
    for item in datasets:
        path = ROOT / item["task_root"]
        item["available"] = item["status"] != "planned" and item["source"] == "bundled" and path.is_dir()
        item["availability_note"] = "Provide an existing inspected task catalog; no download is performed"
    registry = MATH_REGISTRY
    return {
        "catalog_version": 1, "schema_versions": {"training": 2, "deployment": 1},
        "models": [_describe(s, ("preset_prefix", "serving_backend", "tool_protocol", "gateway_protocols", "training_backend_contracts")) for s in registry.models.items()],
        "algorithms": [_describe(s, ("training_backend_contracts", "minimum_group_size", "requires_audited_raw_logprobs", "reduction")) for s in registry.algorithms.items()],
        "task_providers": [_describe(s, ("execution_backends", "task_format")) for s in registry.tasks.items()],
        "harnesses": [_describe(s, ("request_protocol", "tool_protocol", "harbor_versions")) for s in registry.harnesses.items()],
        "datasets": datasets, "deployment_presets": deepcopy(list(PRESETS)),
        "fields": {"training": training, "deployment": deployment},
        "compatibility": {"scope": "component_metadata", "status": "unknown_until_configuration_validated",
                          "constraints": ["Model and harness tool/request protocols must match",
                                          "Algorithm and model must support the selected backend contract",
                                          "Task provider must support the execution backend",
                                          "Harbor version must be supported by the harness"],
                          "validator": "harborrl.math_registry.validate_combination"},
    }
