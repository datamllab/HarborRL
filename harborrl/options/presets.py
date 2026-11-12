"""Small configuration recipes; final validity belongs to existing schemas."""
import sys
from copy import deepcopy
from pathlib import Path

from harborrl.config import ROOT
from harborrl.consistency.config import DEFAULTS as CONSISTENCY_DEFAULTS
from harborrl.deployment.schema import destination, embedded_local

PRESETS = (
    {"id": "embedded-local", "description": "Local training and a local Harbor runner"},
    {"id": "service-ssh", "description": "Controller, SSH GPU target and SSH Docker runners"},
    {"id": "service-sandbox-local-runner", "description": "Controller and runner here; GPU target over SSH; Sandbox execution"},
)


class MissingAnswers(ValueError):
    def __init__(self, paths):
        from .onboarding import describe_field
        self.fields = [describe_field(path) for path in paths]
        super().__init__("Missing required answers:\n" + "\n".join(
            f"  - {row['label']} ({row['path']}): {row['help']}" for row in self.fields))


def merge(base, override, path=""):
    if not isinstance(override, dict):
        raise TypeError(f"{path or 'answers'} must be a mapping")
    result = deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"unknown field: {path}{key}")
        result[key] = (merge(base[key], value, f"{path}{key}.")
                       if isinstance(base[key], dict) and base[key] else deepcopy(value))
    return result


def defaults(preset, output_directory):
    if preset not in {item["id"] for item in PRESETS}:
        raise ValueError(f"unknown preset: {preset}")
    output = Path(output_directory).resolve()
    service = preset != "embedded-local"
    sandbox = preset == "service-sandbox-local-runner"
    training = {
        "consistency": deepcopy(CONSISTENCY_DEFAULTS),
        "schema_version": 2,
        "execution": {"backend": "harbor_job"}, "tasks": {"catalog": None},
        "harness": {"name": "claude_code", "profile": None},
        "harbor": {"python": sys.executable if not service else None, "version": "0.23.0", "workers": [],
                   "runner_transport": "ssh" if preset == "service-ssh" else "local",
                   "environment": {"runtime": "sandbox", "config": None} if sandbox else {"runtime": "docker"}},
        "gateway": {"host": "0.0.0.0" if service else "127.0.0.1", "port": 49184,
                    "advertised_url": None if service else "http://127.0.0.1:49184",
                    "tokenizer_digest": None, "template_digest": None, "audited_raw_logprobs": True},
        "model": {"checkpoint": None, "reference": None, "args_file": "qwen3-8B"},
        "training": {"num_rollout": 2, "learning_rate": 1e-6, "save_interval": 1,
                     "backend_contract": "slime-v0.3.2-native-v1",
                     "seed": None, "export_hf": False},
        "sampling": {"group_size": 2, "groups_per_batch": 1, "max_attempts": 1,
                     "max_tokens": 2048, "max_context": 32768},
        "deployment": {"layout": "colocate", "num_gpus": 1, "actor_gpus": 1,
                       "rollout_gpus": 1, "actor_tensor_parallel_size": 1, "rollout_gpus_per_engine": 1},
        "output": {"root": str(output / "runs")},
    }
    deployment = embedded_local(run_root=str(output / "runs"))
    deployment["gpu_target"]["environment_file"] = None
    deployment["manager"]["run_token_file"] = str(output / "run-token")
    if service:
        deployment["manager"].update(id="controller", mode="service", bind="0.0.0.0",
                                      control_url=None, gateway_url=None)
        deployment["gpu_target"].update(id="gpu", transport="ssh", ssh={"host": None, "user": None, "port": None, "identity": None},
                                        source_dir=None, train_python=None)
        deployment["rollout"]["provider"] = "sandbox" if sandbox else "ssh_docker"
    return training, deployment


def check_answers(answers, output_directory):
    if not isinstance(answers, dict) or set(answers) - {
            "answers_version", "preset", "training", "deployment", "sandbox_config", "local_environment"}:
        raise ValueError("unknown answers field or non-mapping answers")
    if type(answers.get("answers_version")) is not int or answers["answers_version"] != 1:
        raise ValueError("answers_version must be 1")
    preset = answers.get("preset")
    training, deployment = defaults(preset, output_directory)
    merge(training, answers.get("training", {}), "training.")
    merge(deployment, answers.get("deployment", {}), "deployment.")
    if "local_environment" in answers:
        if preset != "embedded-local":
            raise ValueError("local_environment requires Local mode")
        merge({"source_dir": str(ROOT), "python": sys.executable}, answers["local_environment"], "local_environment.")


def build_configs(answers, output_directory):
    check_answers(answers, output_directory)
    preset = answers["preset"]
    training, deployment = defaults(preset, output_directory)
    t = answers.get("training", {})
    d = answers.get("deployment", {})
    training = merge(training, t, "training.")
    deployment = merge(deployment, d, "deployment.")
    if "local_environment" in answers:
        environment = {"source_dir": str(ROOT), "python": sys.executable, **answers["local_environment"]}
        deployment["manager"].update(source_dir=environment["source_dir"], controller_python=environment["python"])
        deployment["gpu_target"].update(source_dir=environment["source_dir"], train_python=environment["python"])
        training["harbor"]["python"] = environment["python"]

    # Repeated fields have one source unless both were explicitly supplied.
    for section, field, other_section, other_field in (
        ("gateway", "host", "manager", "bind"),
        ("gateway", "port", "manager", "gateway_port"),
        ("gateway", "advertised_url", "manager", "gateway_url"),
        ("output", "root", "storage", "run_root"),
    ):
        if field in t.get(section, {}) and other_field not in d.get(other_section, {}):
            deployment[other_section][other_field] = training[section][field]
        elif other_field in d.get(other_section, {}) and field not in t.get(section, {}):
            training[section][field] = deployment[other_section][other_field]
        elif field in t.get(section, {}) and other_field in d.get(other_section, {}):
            left, right = training[section][field], deployment[other_section][other_field]
            if field == "root" and isinstance(left, str) and isinstance(right, str):
                left, right = ((Path(output_directory) / value).resolve() for value in (left, right))
            elif field == "advertised_url" and isinstance(left, str) and isinstance(right, str):
                left, right = left.rstrip("/"), right.rstrip("/")
            if left != right:
                raise ValueError(f"training.{section}.{field} conflicts with deployment.{other_section}.{other_field}")
    count = training["deployment"]["num_gpus"]
    if "expected_gpus" not in d.get("gpu_target", {}):
        deployment["gpu_target"]["expected_gpus"] = count
    if training["deployment"]["layout"] == "colocate":
        for field in ("actor_gpus", "rollout_gpus"):
            if field not in t.get("deployment", {}):
                training["deployment"][field] = count
    if training["harbor"]["runner_transport"] == "ssh":
        workers = deployment["rollout"]["workers"]
        if workers:
            if "workers" not in t.get("harbor", {}):
                training["harbor"]["workers"] = [destination(w["ssh"]) for w in workers]
            if "python" not in t.get("harbor", {}):
                training["harbor"]["python"] = workers[0]["runner_python"]
    if answers.get("sandbox_config") is not None:
        if preset != "service-sandbox-local-runner":
            raise ValueError("sandbox_config requires the Sandbox preset")
        environment = training["harbor"]["environment"]
        if environment.get("config") not in (None, answers["sandbox_config"]):
            raise ValueError("sandbox_config conflicts with training.harbor.environment.config")
        environment["config"] = answers["sandbox_config"]
    required = [("training", section, field) for section, field in (
        ("tasks", "catalog"), ("harness", "profile"), ("harbor", "python"),
        ("model", "checkpoint"), ("model", "reference"),
        ("gateway", "tokenizer_digest"), ("gateway", "template_digest"), ("gateway", "advertised_url"))]
    required += [("deployment", "manager", "source_dir"), ("deployment", "manager", "controller_python"),
                 ("deployment", "storage", "run_root")]
    if preset != "embedded-local":
        required += [("deployment", section, field) for section, field in (
            ("manager", "control_url"), ("manager", "gateway_url"),
            ("gpu_target", "source_dir"), ("gpu_target", "train_python"))]
    configs = {"training": training, "deployment": deployment}
    missing = [".".join(keys) for keys in required if configs[keys[0]][keys[1]][keys[2]] in (None, "")]
    if preset != "embedded-local" and not deployment["gpu_target"]["ssh"].get("host"):
        missing.append("deployment.gpu_target.ssh.host")
    if preset == "service-ssh" and not deployment["rollout"]["workers"]:
        missing.append("deployment.rollout.workers")
    for index, worker in enumerate(deployment["rollout"]["workers"]):
        for key in ("id", "source_dir", "runner_python"):
            if not worker.get(key):
                missing.append(f"deployment.rollout.workers.{index}.{key}")
        if not worker.get("ssh", {}).get("host"):
            missing.append(f"deployment.rollout.workers.{index}.ssh.host")
        if not worker.get("task_roots") or not worker["task_roots"][0]:
            missing.append(f"deployment.rollout.workers.{index}.task_roots.0")
    if preset == "service-sandbox-local-runner" and not training["harbor"]["environment"].get("config"):
        missing.append("sandbox_config")
    if missing:
        if "local_environment" in answers:
            aliases = {"deployment.manager.source_dir": "local_environment.source_dir",
                       "deployment.manager.controller_python": "local_environment.python",
                       "training.harbor.python": "local_environment.python"}
            missing = list(dict.fromkeys(aliases.get(path, path) for path in missing))
        raise MissingAnswers(missing)
    for field in ("tokenizer_digest", "template_digest"):
        if training["gateway"][field] == "0" * 64:
            raise ValueError(f"gateway.{field}: a placeholder digest cannot be used by init")
    expected_mode = "embedded" if preset == "embedded-local" else "service"
    expected_provider = {"embedded-local": "embedded", "service-ssh": "ssh_docker",
                         "service-sandbox-local-runner": "sandbox"}[preset]
    if deployment["manager"]["mode"] != expected_mode or deployment["rollout"]["provider"] != expected_provider:
        raise ValueError("deployment mode/provider conflicts with the selected preset")
    # All generated path references are rooted at the final directory, not staging.
    deployment["storage"]["run_root"] = str((Path(output_directory) / deployment["storage"]["run_root"]).resolve())
    return training, deployment


def worker_defaults():
    return {"id": "worker-0", "ssh": {"host": None}, "source_dir": str(ROOT),
            "runner_python": sys.executable, "docker": True, "task_roots": [str(ROOT)]}
