"""TrainingTarget (P1): local child process and SSH launcher over shared storage."""
from __future__ import annotations

import json
import subprocess
import time

import pytest

from harborrl.deployment import training_target
from harborrl.deployment.training_target import (
    LocalTrainingTarget,
    SshTrainingTarget,
    TrainingTargetError,
)


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_local_target_preserves_cuda_runtime_environment(monkeypatch):
    runtime = {
        "CUDA_HOME": "/opt/cuda",
        "LD_LIBRARY_PATH": "/opt/cuda/lib64",
        "LD_PRELOAD": "/opt/cuda/lib64/libnvrtc.so.13",
        "NVTE_CUDA_INCLUDE_DIR": "/opt/cuda/include",
    }
    for key, value in runtime.items():
        monkeypatch.setenv(key, value)
    target = training_target.local_target({"training_command": ["true"]}, environment={})
    assert {key: target.env[key] for key in runtime} == runtime


def test_local_target_runs_and_forwards_exit_code(tmp_path):
    target = LocalTrainingTarget(["bash", "-c", "exit 3"], cwd=str(tmp_path), env={"PATH": "/bin"})
    target.launch()
    assert target.wait() == 3
    assert target.query()["state"] == "exited"


def test_local_target_cancel_interrupts(tmp_path):
    target = LocalTrainingTarget(["bash", "-c", "trap '' INT; sleep 30"],
                                 cwd=str(tmp_path), env={"PATH": "/bin"})
    target.launch()
    time.sleep(0.2)
    assert target.query()["state"] == "running"
    target.cancel()
    with pytest.raises(subprocess.TimeoutExpired):
        target.child.wait(timeout=2)
    import os
    import signal
    os.killpg(os.getpgid(target.child.pid), signal.SIGKILL)
    target.child.wait()


def test_ssh_launcher_quotes_environment_and_captures_exit_code(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = SshTrainingTarget({"host": "gpu"}, source_dir="/gpu/source", run_dir=run_dir)
    launcher = target.write_launcher(
        ["/manager/source/harborrl/platform/slime_train.sh", "--flag", "value with spaces"],
        {"LR": "1e-6", "NOTE": "two  spaces"},
        path_rewrites=[("/manager/source", "/gpu/source")])
    text = launcher.read_text()
    assert "cd /gpu/source" in text
    assert "/gpu/source/harborrl/platform/slime_train.sh" in text
    assert "/manager/source" not in text
    assert "export LR=1e-6" in text
    assert "export NOTE='two  spaces'" in text
    assert "'value with spaces'" in text
    assert "exit-code" in text
    assert launcher.stat().st_mode & 0o111, "launcher must be executable"


def test_ssh_target_environment_file_is_loaded_via_python(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    environment_file = tmp_path / "gpu-env.json"
    environment_file.write_text(json.dumps({"SLIME_DIR": "/opt/slime"}))
    target = SshTrainingTarget({"host": "gpu"}, source_dir="/gpu/source", run_dir=run_dir,
                               environment_file=environment_file)
    launcher = target.write_launcher(["true"], {})
    text = launcher.read_text()
    assert "HARBORRL_ENV_JSON" in text
    assert "json.load" in text


def test_ssh_target_lifecycle_with_fake_remote(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = SshTrainingTarget({"host": "gpu", "port": 2222}, source_dir="/gpu/source",
                               run_dir=run_dir, poll_interval=0.05)
    calls = []
    killed = {"done": False}

    def fake_run_ssh(spec, command, *, timeout=30, connect_timeout=15):
        calls.append((dict(spec), command))
        if "setsid" in command:
            (run_dir / "launcher.pid").write_text("4242\n")
            return completed()
        if "kill -0" in command:
            return (completed(stdout="ok\n") if not killed["done"]
                    else completed(returncode=1))
        if command.startswith("kill -INT"):
            killed["done"] = True
            (run_dir / "exit-code").write_text("130\n")
            return completed()
        return completed()

    monkeypatch.setattr(training_target.remote, "run_ssh", fake_run_ssh)
    target.write_launcher(["true"], {})
    target.launch()
    with pytest.raises(TrainingTargetError):
        target.launch()  # double launch is refused
    assert target.query() == {"state": "running", "exit_code": None, "pid": 4242,
                              "run_dir": str(run_dir)}
    assert target.cancel()
    assert target.wait() == 130
    assert target.query()["exit_code"] == 130
    # The ssh destination honors user/port from the normalized spec.
    assert any(spec.get("port") == 2222 for spec, _ in calls), (
        "spec with port 2222 must reach run_ssh")
    assert any("setsid" in command for _, command in calls)


def test_ssh_target_wait_timeout(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = SshTrainingTarget({"host": "gpu"}, source_dir="/gpu/source", run_dir=run_dir,
                               poll_interval=0.05)

    def fake_run_ssh(spec, command, *, timeout=30, connect_timeout=15):
        if "setsid" in command:
            (run_dir / "launcher.pid").write_text("99\n")
        return completed()

    monkeypatch.setattr(training_target.remote, "run_ssh", fake_run_ssh)
    target.write_launcher(["true"], {})
    target.launch()
    with pytest.raises(TimeoutError):
        target.wait(timeout=0.2)


def test_ssh_command_assembles_noninteractive_ssh():
    from harborrl.deployment.remote import ssh_command

    argv = ssh_command({"host": "gpu", "user": "root", "port": 2222,
                        "identity": "/key"}, "true", connect_timeout=7)
    assert argv[0] == "ssh"
    assert argv[1:7] == ["-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
                         "-o", "StrictHostKeyChecking=accept-new"]
    assert argv[7:9] == ["-o", "ConnectTimeout=7"]
    assert argv[9:13] == ["-p", "2222", "-i", "/key"]
    assert argv[-2:] == ["root@gpu", "true"]


@pytest.mark.parametrize("shutdown_error", [False, True])
def test_service_dispatch_launches_gpu_job_through_controller(tmp_path, monkeypatch, capsys, shutdown_error):
    """P1+P3 manager flow: materialize, register the run, launch, wait, report."""
    import json as json_module
    from pathlib import Path

    from test_deployment_config import deployment_mapping
    from test_native_training import native_config

    from harborrl.config import launch_plan, load_config
    from harborrl.data.harbor.native_inspector import inspect_native
    from harborrl.deployment import controller as controller_module
    from harborrl.deployment.schema import summary, validate

    config_path, task = native_config(tmp_path)
    catalog_path = Path(json_module.loads(
        config_path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"])
    catalog = json_module.loads(catalog_path.read_text())
    catalog[0]["task_digest"] = inspect_native(task)["task_digest"]
    catalog_path.write_text(json_module.dumps(catalog))
    config = load_config(config_path)
    config["output"]["root"] = str(tmp_path / "shared-runs")
    plan = launch_plan(config)

    deployment = validate(deployment_mapping(tmp_path, storage={
        "run_root": str(tmp_path / "shared-runs")}))
    manager = deployment["manager"]
    token_file = Path(manager["run_token_file"])
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text("token\n")
    token_file.chmod(0o600)
    (tmp_path / "shared-runs").mkdir()
    plan["deployment"] = summary(deployment, "deployment.yaml")

    launched = {}

    class FakeControl:
        def __init__(self, service, host, port):
            self.address = (host, port)

        def serve_forever(self):
            pass

        def shutdown(self):
            launched["control_shutdown"] = True

        def server_close(self):
            launched["control_closed"] = True

    class FakeTarget:
        def __init__(self, spec, *, source_dir, run_dir, **kwargs):
            self.source_dir, self.run_dir = source_dir, Path(run_dir)

        def write_launcher(self, command, environment, *, path_rewrites=()):
            launched["command"] = list(command)
            launched["environment"] = dict(environment)
            launched["rewrites"] = list(path_rewrites)
            return self.run_dir / "launcher.sh"

        def launch(self):
            launched["run_dir"] = str(self.run_dir)
            (self.run_dir / "exit-code").write_text("0\n")

        def wait(self):
            return 0

        def cancel(self):
            return True

    monkeypatch.setattr(controller_module, "make_control_server", FakeControl)
    monkeypatch.setattr(controller_module, "SshTrainingTarget", FakeTarget)
    monkeypatch.setattr(controller_module, 'check_gpu_control', lambda _: {'gpu_to_controller': True})

    if shutdown_error:
        original_close = controller_module.RolloutService.close
        def failed_close(service):
            original_close(service)
            raise RuntimeError("cleanup failure")
        monkeypatch.setattr(controller_module.RolloutService, "close", failed_close)
        with pytest.raises(RuntimeError, match="cleanup failure"):
            controller_module.dispatch_service(plan, deployment, checks=[])
        assert launched["control_shutdown"] and launched["control_closed"]
        return
    exit_code = controller_module.dispatch_service(plan, deployment, checks=[])
    assert exit_code == 0
    assert launched["control_shutdown"] and launched["control_closed"]
    run_dir = Path(launched["run_dir"])
    assert run_dir.is_dir()
    environment = launched["environment"]
    assert environment["HARBORRL_ROLLOUT_MODE"] == "service"
    assert environment["HARBORRL_ROLLOUT_SERVICE_URL"] == manager["control_url"]
    assert environment['HARBORRL_ROLLOUT_SERVICE_TOKEN_FILE'] == str(token_file)
    assert 'HARBORRL_ROLLOUT_SERVICE_TOKEN' not in environment
    assert environment["HARBORRL_NATIVE_GATEWAY_URL"] == manager["gateway_url"]
    assert json_module.loads(environment["HARBORRL_NATIVE_RUNNER_WORKERS"]) == [
        "worker-a", "worker-b"]
    # The launcher runs inside the GPU checkout with a rewritten script path.
    assert launched["rewrites"] == [(manager["source_dir"],
                                     deployment["gpu_target"]["source_dir"])]
    launch_record = json_module.loads((run_dir / "launch.json").read_text())
    assert launch_record["control_url"] == manager["control_url"]
    effective = json_module.loads((run_dir / "effective-config.json").read_text())
    assert effective["components"]["execution_backend"] == "harbor_job"
