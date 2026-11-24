"""Deployment schema v1: parsing, strictness, safety and defaults."""
from __future__ import annotations

import json

import pytest

from harborrl.deployment.schema import (
    DeploymentError,
    embedded_local,
    load_deployment,
    summary,
    validate,
)


def deployment_mapping(tmp_path, **overrides):
    values = {
        "schema_version": 1,
        "manager": {
            "id": "manager-a", "mode": "service", "bind": "0.0.0.0",
            "control_port": 49200, "control_url": "http://manager-a:49200",
            "gateway_port": 49184, "gateway_url": "http://manager-a:49184",
            "run_token_file": str(tmp_path / "secrets" / "run-token"),
            "source_dir": str(tmp_path / "manager-source"),
            "controller_python": str(tmp_path / "manager-venv" / "bin" / "python"),
        },
        "gpu_target": {
            "id": "gpu-8x-h200", "transport": "ssh",
            "ssh": {"host": "gpu-cluster-a"},
            "source_dir": str(tmp_path / "gpu-source"),
            "train_python": str(tmp_path / "gpu-venv" / "bin" / "python"),
            "environment_file": str(tmp_path / "gpu-env.json"),
            "expected_gpus": 8,
        },
        "storage": {
            "mode": "shared_fs", "run_root": str(tmp_path / "runs"),
            "manager_writable": True, "gpu_writable": True, "worker_writable": True,
        },
        "rollout": {
            "provider": "ssh_docker",
            "workers": [
                {"id": "worker-a", "ssh": {"host": "worker-a"},
                 "source_dir": str(tmp_path / "worker-source"),
                 "runner_python": str(tmp_path / "runner-venv" / "bin" / "python"),
                 "docker": True, "task_roots": [str(tmp_path / "tasks")]},
                {"id": "worker-b", "ssh": {"host": "worker-b", "port": 2222},
                 "source_dir": str(tmp_path / "worker-source"),
                 "runner_python": str(tmp_path / "runner-venv" / "bin" / "python"),
                 "docker": True, "task_roots": [str(tmp_path / "tasks")]},
            ],
        },
        "connectivity": {
            "command_timeout_s": 30, "ssh_timeout_s": 15, "http_timeout_s": 10,
            "gpu_to_controller": True, "worker_to_gateway": True,
            "manager_to_gpu_inference": True,
        },
    }
    for section, fields in overrides.items():
        if section in values and isinstance(values[section], dict) and isinstance(fields, dict):
            values[section].update(fields)
        else:
            values[section] = fields
    return values


def test_valid_deployment_normalizes_and_summarizes(tmp_path):
    config = validate(deployment_mapping(tmp_path))
    assert config["manager"]["control_url"] == "http://manager-a:49200"
    assert config["rollout"]["workers"][1]["ssh"] == {"host": "worker-b", "port": 2222}
    assert config["gpu_target"]["ssh"] == {"host": "gpu-cluster-a"}
    plan_summary = summary(config, "deployment.yaml")
    assert plan_summary["mode"] == "service"
    assert plan_summary["gpu"]["transport"] == "ssh"
    assert plan_summary["worker_ids"] == ["worker-a", "worker-b"]
    assert plan_summary["config"] == config
    assert summary(config, "deployment.yaml")["digest"] == plan_summary["digest"]


def test_hub_deployment_retains_pull_configuration(tmp_path):
    raw = deployment_mapping(tmp_path)
    gpu = dict(raw['gpu_target'], transport='http_pull', ssh=None)
    raw['gpu_target'] = gpu
    raw['manager'].update(mode='hub', hub_port=49300,
                          hub_url='http://manager-a:49300',
                          gpu_pool=[dict(gpu, pull_url='http://manager-a:49300')])
    raw['rollout'] = dict(provider='hub_pull', workers=[])
    actual = validate(raw)
    assert actual['manager']['mode'] == 'hub'
    assert actual['manager']['hub_url'] == 'http://manager-a:49300'
    assert actual['manager']['gpu_pool'][0]['transport'] == 'http_pull'
    assert actual['gpu_target']['transport'] == 'http_pull'
    assert actual['rollout']['provider'] == 'hub_pull'


@pytest.mark.parametrize("mutation, message", [
    ({"manager": {"mode": "service", "control_url": "http://127.0.0.1:49200"}}, "loopback"),
    ({"manager": {"control_url": "http://manager-a:49300"}}, "matching its declared port"),
    ({"manager": {"gateway_port": 49200}}, "must differ"),
    ({"manager": {"bind": ""}}, "bind"),
    ({"gpu_target": {"transport": "local", "ssh": {"host": "gpu"}}}, "null for local"),
    ({"gpu_target": {"transport": "rdma"}}, "transport"),
    ({"gpu_target": {"expected_gpus": 0}}, "expected_gpus"),
    ({"rollout": {"workers": []}}, "nonempty list"),
    ({"connectivity": {"ssh_timeout_s": 0}}, "ssh_timeout_s"),
    ({"storage": {"run_root": "relative/runs"}}, "absolute path"),
])
def test_invalid_deployments_fail_closed(tmp_path, mutation, message):
    with pytest.raises(DeploymentError, match=message):
        validate(deployment_mapping(tmp_path, **mutation))


def test_duplicate_worker_ids_and_secret_fields_rejected(tmp_path):
    values = deployment_mapping(tmp_path)
    values["rollout"]["workers"][1]["id"] = "worker-a"
    with pytest.raises(DeploymentError, match="duplicate"):
        validate(values)
    values = deployment_mapping(tmp_path)
    values["gpu_target"]["ssh"]["password"] = "hunter2"
    with pytest.raises(DeploymentError, match="never stored"):
        validate(values)


def test_service_mode_requires_reachable_urls_but_embedded_allows_loopback(tmp_path):
    values = deployment_mapping(tmp_path, manager={"mode": "embedded"})
    values["manager"]["control_url"] = "http://127.0.0.1:49200"
    values["manager"]["gateway_url"] = "http://127.0.0.1:49184"
    assert validate(values)["manager"]["mode"] == "embedded"


def test_embedded_local_default_is_valid():
    config = embedded_local(run_root="/tmp/harborrl-runs", workers=("worker-a",))
    assert config["manager"]["mode"] == "embedded"
    assert config["gpu_target"]["transport"] == "local"
    assert config["rollout"]["provider"] == "embedded"
    assert [worker["id"] for worker in config["rollout"]["workers"]] == ["worker-0"]
    # Training-config SSH destinations are addresses, not ids: the embedded
    # topology keeps slug ids while preserving the destination for access.
    config = embedded_local(workers=("root@192.0.2.10", "worker-b"))
    assert [worker["id"] for worker in config["rollout"]["workers"]] == [
        "worker-0", "worker-1"]
    assert config["rollout"]["workers"][0]["ssh"]["host"] == "root@192.0.2.10"


def test_load_deployment_explicit_default_and_embedded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, source = load_deployment()
    assert source == "embedded-local"
    assert config["manager"]["mode"] == "embedded"

    (tmp_path / "deployment.yaml").write_text(
        json.dumps(deployment_mapping(tmp_path)))
    config, source = load_deployment()
    assert source == str(tmp_path / "deployment.yaml")
    assert config["manager"]["mode"] == "service"
    with pytest.raises(DeploymentError, match="not found"):
        load_deployment(tmp_path / "missing.yaml")


def test_cli_dry_run_embeds_deployment_summary(tmp_path, capsys, monkeypatch):
    from pathlib import Path

    from test_native_training import native_config

    config_path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native

    catalog_path = Path(
        json.loads(config_path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"])
    catalog = json.loads(catalog_path.read_text())
    catalog[0]["task_digest"] = inspect_native(task)["task_digest"]
    catalog_path.write_text(json.dumps(catalog))

    deployment_file = tmp_path / "deployment.yaml"
    deployment_file.write_text(json.dumps(deployment_mapping(tmp_path)))

    from harborrl.cli import main

    assert main(["train", str(config_path), "--deployment", str(deployment_file),
                 "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["deployment"]["mode"] == "service"
    assert plan["deployment"]["gpu"]["transport"] == "ssh"
    assert plan["deployment"]["source"] == str(deployment_file)

    # Without a deployment file the embedded-local topology is attached.
    monkeypatch.chdir(tmp_path)
    deployment_file.unlink()
    assert main(["train", str(config_path), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["deployment"]["mode"] == "embedded"
    assert plan["deployment"]["source"] == "embedded-local"
