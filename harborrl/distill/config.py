"""Independent distill schema; the legacy GRPO schema is unchanged."""
from copy import deepcopy
from pathlib import Path

import yaml

from harborrl.consistency.config import DEFAULTS as CONSISTENCY_DEFAULTS
from harborrl.consistency.config import validate as validate_consistency
from harborrl.trajectories.native import finite

CONTRACT = "transformers-math-onpolicy-v1"
METHODS = ("reinforce", "prm", "opd", "prm_opd")
DEFAULTS = {
    "consistency": CONSISTENCY_DEFAULTS,
    "schema_version": 1, "backend_contract": CONTRACT, "method": "opd",
    "tasks": {"path": None, "train_split": "train", "eval_split": "test"},
    "student": {"kind": "tiny_qwen", "checkpoint": None, "hidden_size": 32, "layers": 1, "heads": 2},
    "teacher": {"kind": "fake", "checkpoint": None},
    "prm": {"kind": "none", "weight": 1.0},
    "sampling": {"max_new_tokens": 8, "max_context": 256, "temperature": 1.0, "top_p": 1.0},
    "training": {"steps": 2, "batch_size": 8, "microbatch_size": 2, "learning_rate": 0.0003,
                 "seed": 17, "max_grad_norm": 1.0, "opd_weight": 1.0, "outcome_weight": 0.0,
                 "save_interval": 1, "behavior_tolerance": 0.0001},
    "deployment": {"file": None, "device": "cpu", "world_size": 1},
    "output": {"directory": None},
}


def merge(base, override, prefix=""):
    if not isinstance(override, dict):
        raise TypeError(f"{prefix or 'config'} must be a mapping")
    result = deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown field: {prefix}{key}")
        result[key] = merge(base[key], value, f"{prefix}{key}.") if isinstance(base[key], dict) else value
    return result


def validate(raw, path, *, allow_synthetic=False):
    config = merge(DEFAULTS, raw)
    config['consistency'] = validate_consistency(config['consistency'])
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["backend_contract"] != CONTRACT:
        raise ValueError("distill requires schema 1 / transformers-math-onpolicy-v1")
    if config["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    student, teacher, prm = config["student"], config["teacher"], config["prm"]
    if student["kind"] not in {"tiny_qwen", "hf_local"} or teacher["kind"] not in {"none", "fake", "tiny_qwen", "hf_local"}:
        raise ValueError("Unsupported student or teacher adapter")
    if prm["kind"] not in {"none", "fake", "equation"}:
        raise ValueError("Unsupported PRM adapter")
    if not allow_synthetic and (student["kind"] == "tiny_qwen" or teacher["kind"] in {"fake", "tiny_qwen"} or prm["kind"] == "fake"):
        raise ValueError("Tiny/fake adapters require explicit --allow-synthetic")
    opd = config["method"] in {"opd", "prm_opd"}
    process = config["method"] in {"prm", "prm_opd"}
    if opd == (teacher["kind"] == "none") or process == (prm["kind"] == "none"):
        raise ValueError("Selected method requires exactly its teacher/PRM adapters; disable unused adapters")
    if teacher["kind"] == "tiny_qwen" and student["kind"] != "tiny_qwen":
        raise ValueError("Tiny teacher requires the same tiny tokenizer")
    for section, fields in (("student", ("hidden_size", "layers", "heads")),
                           ("sampling", ("max_new_tokens", "max_context")),
                           ("training", ("steps", "batch_size", "microbatch_size", "save_interval")),
                           ("deployment", ("world_size",))):
        for field in fields:
            if type(config[section][field]) is not int or config[section][field] <= 0:
                raise ValueError(f"{section}.{field} must be a positive integer")
    if type(config["training"]["seed"]) is not int or config["training"]["seed"] < 0:
        raise ValueError("training.seed must be a nonnegative integer")
    for section, fields in (("training", ("learning_rate", "max_grad_norm", "behavior_tolerance")),):
        for field in fields:
            if not finite(config[section][field]) or config[section][field] <= 0:
                raise ValueError(f"{section}.{field} must be positive and finite")
    for section, field in (("training", "opd_weight"), ("training", "outcome_weight"), ("prm", "weight")):
        if not finite(config[section][field]) or config[section][field] < 0:
            raise ValueError(f"{section}.{field} must be nonnegative and finite")
    if (opd and config["training"]["opd_weight"] == 0) or (process and prm["weight"] == 0):
        raise ValueError("The selected method's loss weight must be positive")
    if config["method"] == "reinforce" and config["training"]["outcome_weight"] == 0:
        raise ValueError("reinforce requires positive outcome_weight")
    if config["sampling"]["temperature"] != 1.0 or config["sampling"]["top_p"] != 1.0:
        raise ValueError("Only unmodified raw-policy sampling (temperature=top_p=1) is supported")
    if config["sampling"]["max_new_tokens"] >= config["sampling"]["max_context"]:
        raise ValueError("Response limit must be below context limit")
    if student["hidden_size"] % student["heads"] or (student["hidden_size"] // student["heads"]) % 2:
        raise ValueError("Tiny Qwen head size must be an even integer")
    if config["deployment"]["device"] not in {"cpu", "cuda"}:
        raise ValueError("deployment.device must be cpu or cuda")
    if config["training"]["batch_size"] % config["deployment"]["world_size"]:
        raise ValueError("Global batch must divide evenly among data-parallel ranks")
    for key in ("train_split", "eval_split"):
        if config["tasks"][key] not in {"train", "validation", "test"}:
            raise ValueError("Unknown task split")
    if config["tasks"]["train_split"] != "train" or config["tasks"]["eval_split"] == "train":
        raise ValueError("Training uses train; evaluation must be held out")
    for section, field, required in (("tasks", "path", True), ("output", "directory", True),
                                    ("deployment", "file", False), ("student", "checkpoint", student["kind"] == "hf_local"),
                                    ("teacher", "checkpoint", teacher["kind"] == "hf_local")):
        value = config[section][field]
        if value is None and not required:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{field} requires a path")
        config[section][field] = str((Path(path).resolve().parent / Path(value).expanduser()).resolve())
    return config


def load(path, *, allow_synthetic=False):
    return validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")), path, allow_synthetic=allow_synthetic)
