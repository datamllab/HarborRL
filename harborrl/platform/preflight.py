"""Live deployment checks for the native pipeline; no training dispatch.

``doctor`` composes two layers: the deployment topology graph (manager, GPU
target, rollout workers, control-loop addresses) and the machine-local
environment checks. The local layer only runs where it is meaningful: when
the GPU target is remote, dependency probes belong to the GPU job startup,
not to this machine.
"""
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from harborrl.config import ROOT


def _deployment(plan):
    return (plan.get("deployment") or {}).get("config")


def doctor(plan):
    if plan['config']['execution']['backend'] == 'text_math':
        return _math_doctor(plan)
    return _doctor(plan)


def _math_doctor(plan):
    """Use the shared backend probes without invoking Harbor task workers."""
    config = plan['config']
    from harborrl.math.data import load_tasks
    checks = []
    try:
        rows = load_tasks(config['tasks']['path'])
        checks.append({'service': 'locked math tasks', 'ok': True, 'count': len(rows)})
    except (ValueError, OSError, TypeError) as exc:
        checks.append({'service': 'locked math tasks', 'ok': False, 'detail': str(exc)})
    if config['algorithm']['name'] == 'opd':
        from harborrl.math.teacher import check
        try:
            check({**config['teacher'], 'identity': 'preflight'})
            checks.append({'service': 'OPD teacher identity', 'ok': True,
                           'detail': 'live model/version checked; checkpoint and tokenizer hashes freeze at launch'})
        except (ValueError, OSError, TypeError) as exc:
            checks.append({'service': 'OPD teacher identity', 'ok': False, 'detail': str(exc)})
    if 'process_reward' in config:
        from harborrl.math import prm
        try:
            prm.check(prm.freeze(config['process_reward']))
            checks.append({'service': 'frozen PRM identity', 'ok': True})
        except (ValueError, OSError, TypeError, KeyError) as exc:
            checks.append({'service': 'frozen PRM identity', 'ok': False, 'detail': str(exc)})
    return checks + _doctor(plan, math_only=True)


def _doctor(plan, *, math_only=False):
    config = plan["config"]
    checks = []
    deployment = _deployment(plan)
    # Embedded-local is a compatibility record, not an opt-in: only an
    # explicit deployment file activates the topology layers.
    explicit = deployment is not None and plan.get("deployment", {}).get(
        "source", "embedded-local") != "embedded-local"
    local_gpu = deployment is None or deployment["gpu_target"]["transport"] == "local"
    legacy_workers = (not explicit) or deployment["rollout"]["provider"] in ("embedded", "sandbox")

    if explicit:
        from harborrl.deployment.preflight import preflight as deployment_preflight
        checks.extend(deployment_preflight(plan, deployment, live=True))
        if not local_gpu:
            checks.append({
                "role": "gpu_target", "machine": deployment["gpu_target"]["id"],
                "check": "gpu_target.training_dependencies", "ok": True,
                "detail": "Slime/Megatron/SGlang versions are enforced fail-closed "
                          "by slime_train.sh at GPU job startup on the GPU machine",
            })

    runner = Path(config.get("harbor", {}).get("python", '/nonexistent'))
    if not math_only:
        checks.append({"path": "harbor.python", "ok": runner.is_file()})
    from harborrl.rollout.harbor_job.coordinator import runner_command
    if not math_only and legacy_workers and runner.is_file():
        workers = ([None] if config['harbor'].get('runner_transport') == 'local'
                   else config['harbor']['workers'])
        for worker in workers:
            try:
                probe = subprocess.run(
                    runner_command(str(runner), ROOT, worker) + ["--probe"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                probe = SimpleNamespace(returncode=1, stdout="", stderr=str(exc))
            try:
                payload = json.loads(probe.stdout)
                runtime_version = tuple(payload.get("python", "").split(".")[:2])
                version_ok = payload.get("harbor_version") == config["harbor"]["version"]
                runtime_ok = len(runtime_version) == 2 and all(part.isdigit() for part in runtime_version)
            except (ValueError, AttributeError, TypeError):
                runtime_version, version_ok, runtime_ok = (), False, False
            checks.append({
                "service": "local Harbor runner" if worker is None else "external Harbor worker",
                "worker": worker,
                "ok": probe.returncode == 0 and runtime_ok
                      and tuple(map(int, runtime_version)) >= (3, 12) and version_ok,
                "detail": probe.stdout.strip() or probe.stderr.strip()[-1000:],
            })
        if 'HARBORRL_NATIVE_SANDBOX' in plan['environment']:
            for worker in workers:
                try:
                    probe = subprocess.run(
                        runner_command(str(runner), ROOT, worker,
                                       module='harborrl.rollout.harbor_job.sandbox_probe') + ['--stdin'],
                        input=plan['environment']['HARBORRL_NATIVE_SANDBOX'], cwd=ROOT,
                        capture_output=True, text=True, timeout=60, check=False)
                    checks.append({'service': 'sandbox offline runtime', 'worker': worker,
                                   'ok': probe.returncode == 0, 'detail': probe.stdout.strip()})
                except (subprocess.SubprocessError, OSError) as exc:
                    checks.append({'service': 'sandbox offline runtime', 'worker': worker,
                                   'ok': False, 'detail': type(exc).__name__})

    paths = [("model.checkpoint", config["model"]["checkpoint"]),
             ("model.reference", config["model"]["reference"])]
    if not math_only:
        paths.extend([("tasks.catalog", config["tasks"]["catalog"]),
                      ("harness.profile", config["harness"]["profile"])])
    if config['training'].get('resume'):
        paths.append(('training.resume', config['training']['resume']))
    if math_only and config['algorithm']['name'] == 'opd':
        paths.append(('teacher.checkpoint', config['teacher']['checkpoint']))
    for key, path in paths:
        checks.append({"path": key, "ok": Path(path).exists()})

    model_config = Path(config["model"]["checkpoint"]) / "config.json"
    if model_config.is_file():
        try:
            model = json.loads(model_config.read_text())
            heads = model.get("num_attention_heads", 0)
            kv_heads = model.get("num_key_value_heads", heads)
            for tp in {config["deployment"]["actor_tensor_parallel_size"],
                       config["deployment"]["rollout_gpus_per_engine"]}:
                checks.append({"model_tp": tp, "ok": bool(
                    heads and kv_heads and heads % tp == 0 and (kv_heads % tp == 0 or tp % kv_heads == 0))})
        except (OSError, ValueError, TypeError) as exc:
            checks.append({"path": "model.config", "ok": False, "detail": str(exc)})

    from harborrl.tasks.catalog import catalog
    try:
        entries = [] if math_only else catalog(config)
        for entry in entries:
            # catalog() already checks the complete task tree against its lock.
            checks.append({"task": entry["id"], "ok": True, "detail": []})
    except (ValueError, OSError, TypeError) as exc:
        checks.append({"service": "native catalog", "ok": False, "detail": str(exc)})

    if local_gpu:
        for package in ("ray", "torch", "sglang", "transformers"):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = None
            checks.append({"dependency": package, "version": version, "ok": version is not None})

    from harborrl.backends.slime_v032.versions import (
        MEGATRON_COMMIT,
        SGLANG_IMAGE_TAG,
        SLIME_COMMIT,
    )
    if local_gpu:
        slime_dir = os.environ.get("SLIME_DIR", "").strip()
        slime_path = Path(slime_dir) if slime_dir else None
        checks.append({"path": "SLIME_DIR", "ok": bool(slime_path and slime_path.is_dir()),
                       "detail": slime_dir or "not set"})
        if slime_path and slime_path.is_dir():
            checks.append({"path": "slime entrypoint", "ok": (slime_path / "train.py").is_file()})
            preset = slime_path / "scripts" / "models" / f"{config['model']['args_file']}.sh"
            checks.append({"path": "slime model preset", "ok": preset.is_file(), "detail": str(preset)})
            try:
                commit = subprocess.run(["git", "-C", str(slime_path), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, timeout=15, check=True).stdout.strip()
            except (subprocess.SubprocessError, OSError) as exc:
                commit = ""
                checks.append({"service": "slime commit", "ok": False, "detail": str(exc)})
            if commit:
                checks.append({"service": "slime commit", "ok": commit == SLIME_COMMIT,
                               "detail": f"{commit} != {SLIME_COMMIT}" if commit != SLIME_COMMIT else commit})

            megatron_dir = os.environ.get("MEGATRON_DIR", "").strip()
            if megatron_dir:
                try:
                    megatron_commit = subprocess.run(["git", "-C", megatron_dir, "rev-parse", "HEAD"],
                                                     capture_output=True, text=True, timeout=15, check=True).stdout.strip()
                except (subprocess.SubprocessError, OSError) as exc:
                    megatron_commit = ""
                    checks.append({"service": "megatron commit", "ok": False, "detail": str(exc)})
                if megatron_commit:
                    checks.append({"service": "megatron commit", "ok": megatron_commit == MEGATRON_COMMIT,
                                   "detail": f"{megatron_commit} != {MEGATRON_COMMIT}" if megatron_commit != MEGATRON_COMMIT else megatron_commit})
            else:
                checks.append({"service": "megatron commit",
                               "ok": True,
                               "detail": "MEGATRON_DIR not set; official Slime image bundles Megatron"})

            runtime_env = os.environ.copy()
            runtime_pythonpath = os.pathsep.join(filter(None, [
                str(slime_path), megatron_dir, str(ROOT), runtime_env.get("PYTHONPATH", ""),
            ]))
            runtime_env["PYTHONPATH"] = runtime_pythonpath
            for module in ("slime.rollout.base_types", "slime.utils.types",
                           "slime.backends.megatron_utils.loss"):
                try:
                    probe = subprocess.run([sys.executable, "-c", f"import {module}"],
                                           capture_output=True, text=True, timeout=90, check=False,
                                           env=runtime_env)
                except (subprocess.SubprocessError, OSError) as exc:
                    probe = SimpleNamespace(returncode=1, stdout="", stderr=str(exc))
                checks.append({"dependency": module, "ok": probe.returncode == 0,
                               "detail": probe.stderr.strip()[-1000:]})

        sglang_image = os.environ.get("SGLANG_IMAGE", "").strip()
        if sglang_image:
            checks.append({"service": "sglang image", "ok": sglang_image.endswith(SGLANG_IMAGE_TAG),
                           "detail": sglang_image})

        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                        capture_output=True, text=True, timeout=10, check=True)
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                count = (len([item for item in visible.split(",") if item.strip() and item.strip() != "-1"])
                         if visible is not None else len(result.stdout.splitlines()))
                checks.append({"resource": "visible GPU budget", "available": count,
                               "ok": count >= config["deployment"]["num_gpus"]})
            except (subprocess.SubprocessError, OSError) as exc:
                checks.append({"resource": "visible GPU budget", "ok": False, "detail": str(exc)})
        else:
            checks.append({"resource": "visible GPU budget", "ok": False, "detail": "nvidia-smi unavailable"})

    if config["training"]["backend_contract"] != "slime-legacy":
        from harborrl.backends.slime_v032.launcher import hook_modules
        modules = hook_modules()
        if math_only:
            modules.update(rollout='harborrl.backends.slime_v032.math_rollout',
                           converter='harborrl.backends.slime_v032.math_rollout')
            if config['algorithm']['name'] == 'opd':
                modules.update(advantage='harborrl.backends.slime_v032.opd',
                               loss='harborrl.backends.slime_v032.opd')
            elif config['algorithm']['name'] == 'dapo':
                modules.update(loss='harborrl.backends.slime_v032.dapo')
            elif 'process_reward' in config:
                modules.update(advantage='harborrl.backends.slime_v032.prm')
        for hook_name, module_name in modules.items():
            try:
                importlib.import_module(module_name)
                checks.append({"service": "training backend hook", "hook": hook_name, "ok": True})
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                checks.append({
                    "service": "training backend hook", "hook": hook_name,
                    "ok": False, "detail": str(exc),
                })

    checks.append({
        "service": "raw-model logprob audit",
        "ok": True,
        "detail": ("enabled by lock; live serving semantic probe remains a GPU gate"
                   if math_only or config["gateway"]["audited_raw_logprobs"] else
                   "disabled; generated evidence remains EVAL_ONLY and training will fail closed"),
    })
    return checks
