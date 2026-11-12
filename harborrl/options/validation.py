"""Offline checks composed from the authoritative configuration validators."""
from copy import deepcopy
from pathlib import Path

import yaml

from harborrl.config import native
from harborrl.deployment import schema


def read_mapping(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("configuration must be a mapping")
    return value


def check(path, status, code, message):
    return {"path": path, "status": status, "code": code, "message": message}


def report(checks, *, check_files=False, source="explicit"):
    states = {item["status"] for item in checks}
    status = "invalid" if "failed" in states else (
        "valid_with_deferred_checks" if "deferred" in states else "valid")
    return {"validation_version": 1, "status": status,
            "scope": "structure_and_local_files" if check_files else "structure",
            "deployment_source": source, "checks": checks}


def cross_checks(training, deployment):
    """Only compare settings actually consumed together by the selected mode."""
    checks = []

    def require(condition, path, message):
        if not condition:
            checks.append(check(path, "failed", "inconsistent", message))

    gpu = deployment["gpu_target"]
    require(training["deployment"]["num_gpus"] <= gpu["expected_gpus"],
            "deployment.gpu_target.expected_gpus", "Expected GPU count is below the training budget")
    if deployment["manager"]["mode"] != "service":
        return checks
    manager, rollout, harbor = deployment["manager"], deployment["rollout"], training["harbor"]
    for key, other in (("host", "bind"), ("port", "gateway_port"), ("advertised_url", "gateway_url")):
        left, right = training["gateway"][key], manager[other]
        if key == "advertised_url":
            left, right = left.rstrip("/"), right.rstrip("/")
        require(left == right, f"training.gateway.{key}", f"Must match deployment.manager.{other}")
    require(training["output"]["root"] == deployment["storage"]["run_root"],
            "training.output.root", "Must match deployment.storage.run_root")
    sandbox = harbor.get("environment", {}).get("runtime") == "sandbox"
    require(sandbox == (rollout["provider"] == "sandbox"), "deployment.rollout.provider",
            "Sandbox provider and training environment must agree")
    workers = rollout["workers"]
    local = harbor.get("runner_transport", "ssh") == "local"
    require(not workers if local else bool(workers), "deployment.rollout.workers",
            "Local runners require no workers; SSH runners require workers")
    if not local and workers:
        destinations = [schema.destination(worker["ssh"]) for worker in workers]
        require(harbor["workers"] == destinations, "training.harbor.workers",
                "Must match deployment worker destinations in order")
        import os
        require(os.path.normpath(harbor["python"]) == os.path.normpath(workers[0]["runner_python"]),
                "training.harbor.python",
                "Must match the deployment's first runner_python")
    return checks


def _local_file(checks, path, label, *, directory=False):
    target = Path(path)
    exists = target.is_dir() if directory else target.is_file()
    checks.append(check(label, "passed" if exists else "failed", "local_path",
                        "Local path exists" if exists else "Required local path is missing"))


def _file_checks(training, deployment):
    checks = []
    manager = deployment["manager"]
    _local_file(checks, manager["source_dir"], "deployment.manager.source_dir", directory=True)
    _local_file(checks, manager["controller_python"], "deployment.manager.controller_python")
    for label, function in (
        ("training.harness.profile", lambda: native.validate_files(training)),
        ("training.tasks.catalog", lambda: native.catalog(training)),
    ):
        try:
            function()
        except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
            checks.append(check(label, "failed", "local_reference", str(exc)))
        else:
            checks.append(check(label, "passed", "local_reference", "Local reference validated"))
    gpu = deployment["gpu_target"]
    if gpu["transport"] == "local":
        for key in ("checkpoint", "reference"):
            _local_file(checks, training["model"][key], f"training.model.{key}", directory=True)
        _local_file(checks, gpu["train_python"], "deployment.gpu_target.train_python")
        _local_file(checks, gpu["source_dir"], "deployment.gpu_target.source_dir", directory=True)
        if gpu.get("environment_file"):
            _local_file(checks, gpu["environment_file"], "deployment.gpu_target.environment_file")
    else:
        checks.append(check("deployment.gpu_target", "deferred", "remote_paths",
                            "Model, interpreter and source paths belong to the GPU target"))
    if training["harbor"].get("runner_transport", "ssh") == "local":
        _local_file(checks, training["harbor"]["python"], "training.harbor.python")
    else:
        checks.append(check("deployment.rollout.workers", "deferred", "remote_paths",
                            "Runner paths must be checked on their worker machines"))
    environment = training["harbor"].get("environment", {})
    if environment.get("runtime") == "sandbox":
        from harborrl.rollout.harbor_job.sandbox_config import (
            load_sandbox,
            task_binding,
        )
        try:
            cfg = load_sandbox(environment["config"])
            if cfg["adapter"] == "harborrl.rollout.harbor_job.environments.opensandbox:ManagedOpenSandboxEnvironment":
                if cfg["options"] or any(binding["options"] for binding in cfg["bindings"]):
                    raise ValueError("standard OpenSandbox does not accept platform options")
            else:
                checks.append(check("sandbox.adapter", "deferred", "runner_adapter",
                                    "Platform adapter validation belongs to its trusted runner environment"))
            for entry in native.catalog(training):
                task_binding(cfg, entry["path"])
        except (ValueError, TypeError, KeyError, OSError, AttributeError, yaml.YAMLError) as exc:
            checks.append(check("training.harbor.environment", "failed", "sandbox_config", str(exc)))
        else:
            checks.append(check("training.harbor.environment", "passed", "sandbox_config",
                                "Sandbox structure and task image bindings validated; credentials not read"))
        checks.append(check("sandbox.platform", "deferred", "platform_access",
                            "Credentials, offline binary, capacity and network need runner preflight"))
    return checks


def validate_mappings(training, deployment, *, config_path, check_files=False, source="explicit"):
    checks = []
    normalized = {}
    for name, function in (
        ("training", lambda: native.validate(deepcopy(training), Path(config_path).resolve())),
        ("deployment", lambda: schema.validate(deepcopy(deployment))),
    ):
        try:
            normalized[name] = function()
        except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
            checks.append(check(name, "failed", "schema", str(exc)))
        else:
            checks.append(check(name, "passed", "schema", "Schema and semantic checks passed"))
    if len(normalized) == 2:
        checks.extend(cross_checks(normalized["training"], normalized["deployment"]))
        if check_files:
            checks.extend(_file_checks(normalized["training"], normalized["deployment"]))
        else:
            checks.append(check("references", "deferred", "files_not_checked", "Use --check-files for local references"))
    checks.append(check("runtime", "deferred", "live_preflight",
                        "GPU memory, model loading, live digests and connectivity require doctor or an explicit live test"))
    return report(checks, check_files=check_files, source=source)


def validate_paths(config_path, deployment_path=None, *, check_files=False):
    try:
        training = read_mapping(config_path)
        deployment, source = schema.load_deployment(deployment_path)
        if source == "embedded-local":
            normalized = native.validate(deepcopy(training), Path(config_path).resolve())
            deployment = schema.embedded_local(run_root=normalized["output"]["root"],
                workers=normalized["harbor"]["workers"], expected_gpus=normalized["deployment"]["num_gpus"])
        # The legacy validator omits an optional null key from its normalized output.
        deployment["gpu_target"].setdefault("environment_file", None)
        return validate_mappings(training, deployment, config_path=config_path,
                                 check_files=check_files, source=source)
    except (ValueError, TypeError, KeyError, OSError, AttributeError, yaml.YAMLError) as exc:
        return report([check("configuration", "failed", "load", str(exc))], check_files=check_files)
