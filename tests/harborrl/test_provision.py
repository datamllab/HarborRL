"""Provision contract: obtain and release the GPU machine around a dispatch.

The core owns the lifecycle; the commands are site-specific and opaque. These
tests pin the parts that must not drift: the one-line JSON record, the
fail-closed parsing, the evidence trail and the release outcome.
"""
from __future__ import annotations

import json
import sys

import pytest

from harborrl.deployment import provision
from harborrl.deployment.schema import DeploymentError, validate


def mapping(tmp_path, *, provision_block=None, gpu_over=None):
    """A minimal valid deployment; provision is declared when asked for."""
    gpu = {
        "id": "gpu-8x-h200", "transport": "ssh",
        "ssh": None if provision_block is not None else {"host": "fixed-host"},
        "source_dir": str(tmp_path / "gpu-source"),
        "train_python": str(tmp_path / "gpu-bin" / "python"),
        "environment_file": None, "expected_gpus": 8,
    }
    if provision_block is not None:
        gpu["provision"] = provision_block
    gpu.update(gpu_over or {})
    return {
        "schema_version": 1,
        "manager": {
            "id": "manager-a", "mode": "service", "bind": "0.0.0.0",
            "control_port": 49200, "control_url": "http://manager-a:49200",
            "gateway_port": 49184, "gateway_url": "http://manager-a:49184",
            "run_token_file": str(tmp_path / "run-token"),
            "source_dir": str(tmp_path / "manager-source"),
            "controller_python": str(tmp_path / "bin" / "python"),
        },
        "gpu_target": gpu,
        "storage": {
            "mode": "shared_fs", "run_root": str(tmp_path / "runs"),
            "manager_writable": True, "gpu_writable": True, "worker_writable": True,
        },
        "rollout": {"provider": "sandbox", "workers": []},
        "connectivity": {
            "command_timeout_s": 30, "ssh_timeout_s": 15, "http_timeout_s": 10,
            "gpu_to_controller": True, "worker_to_gateway": False,
            "manager_to_gpu_inference": True,
        },
    }


def target(tmp_path, **provision_overrides):
    """The normalized gpu_target for a deployment that provisions its machine."""
    block = {"up": [sys.executable, "-c", "print('x')"], "timeout_s": 30}
    block.update(provision_overrides)
    return validate(mapping(tmp_path, provision_block=block))["gpu_target"]


def spec(tmp_path, **provision_overrides):
    """The normalized provision block the core actually consumes."""
    return target(tmp_path, **provision_overrides)["provision"]


def runnable(body):
    """A provision command that runs `body` in a fresh interpreter."""
    return [sys.executable, "-c", body]


def test_provision_record_normalizes_into_an_ssh_spec(tmp_path):
    gpu = target(tmp_path)
    assert gpu["ssh"] is None
    assert gpu["provision"]["up"] == [sys.executable, "-c", "print('x')"]
    assert gpu["provision"]["timeout_s"] == 30
    record = {"host": "10.0.0.7", "user": "root", "port": 2222,
              "identity": str(tmp_path / "key"), "handle": "job-1"}
    assert provision.ssh_spec(record) == {
        "host": "10.0.0.7", "user": "root", "port": 2222,
        "identity": str(tmp_path / "key")}


@pytest.mark.parametrize("gpu_over, block, message", [
    ({"transport": "local", "ssh": None}, {"up": ["true"]}, "requires ssh transport"),
    ({"ssh": {"host": "fixed"}}, {"up": ["true"]}, "declare no gpu_target.ssh"),
    ({"ssh": None}, {"down": ["true"]}, "requires up"),
    ({"ssh": None}, {"up": []}, "nonempty"),
    ({"ssh": None}, {"up": ["true"], "bogus": 1}, "supports"),
    ({"ssh": None}, {"up": ["true"], "timeout_s": 0}, "timeout_s"),
])
def test_provision_validation_fails_closed(tmp_path, gpu_over, block, message):
    with pytest.raises(DeploymentError, match=message):
        validate(mapping(tmp_path, provision_block=block, gpu_over=gpu_over))


def test_a_fixed_target_keeps_the_exact_historical_shape(tmp_path):
    """Declaring no provision leaves a fixed-ssh target byte for byte."""
    config = validate(mapping(tmp_path))
    assert "provision" not in config["gpu_target"]
    assert config["gpu_target"]["ssh"] == {"host": "fixed-host"}


def test_up_returns_the_record_and_publishes_evidence(tmp_path):
    provision_spec = spec(tmp_path, up=runnable(
        "import json;print('submitting');"
        "print(json.dumps({'host':'10.1.2.3','user':'root','port':22,"
        "'identity':'/keys/id','handle':'job-9'}))"))
    record = provision.up(provision_spec, evidence_path=tmp_path / "up.json")
    assert record == {"host": "10.1.2.3", "user": "root", "port": 22,
                      "identity": "/keys/id", "handle": "job-9"}
    assert json.loads((tmp_path / "up.json").read_text())["record"] == record


def test_up_failure_is_loud_and_carries_the_command_stderr(tmp_path):
    provision_spec = spec(tmp_path, up=runnable(
        "import sys;sys.stderr.write('quota exhausted');sys.exit(3)"))
    with pytest.raises(provision.ProvisionError, match="failed with 3.*quota exhausted"):
        provision.up(provision_spec)


@pytest.mark.parametrize("stdout, message", [
    ("", "no record"),
    ("not json", "did not report a JSON record"),
    ("[1, 2]", "must be a JSON object"),
    ('{"user": "root"}', "requires an ssh destination host"),
    ('{"host": "h", "secret": "x"}', "unsupported fields"),
    ('{"host": "h", "port": 0}', "port must be an integer"),
    ('{"host": "h", "port": "22"}', "port must be an integer"),
    ('{"host": "-oProxyCommand"}', "ssh destination host"),
])
def test_up_rejects_a_malformed_record(tmp_path, stdout, message):
    provision_spec = spec(tmp_path, up=runnable(
        f"import sys;sys.stdout.write({stdout!r})"))
    with pytest.raises(provision.ProvisionError, match=message):
        provision.up(provision_spec)


def test_down_passes_the_record_on_stdin_and_reports_released(tmp_path):
    echo = tmp_path / "echo.json"
    provision_spec = spec(
        tmp_path,
        up=runnable("print('{\"host\": \"h\"}')"),
        down=runnable(
            "import sys;"
            f"open({str(echo)!r},'w').write(sys.stdin.read())"))
    record = provision.up(provision_spec)
    outcome, detail = provision.down(
        provision_spec, record, evidence_path=tmp_path / "down.json")
    assert outcome == "released"
    assert json.loads(echo.read_text()) == record
    assert json.loads((tmp_path / "down.json").read_text())["outcome"] == "released"


def test_down_failure_is_reported_not_raised(tmp_path):
    provision_spec = spec(
        tmp_path,
        up=runnable("print('{\"host\": \"h\"}')"),
        down=runnable("import sys;sys.stderr.write('gone');sys.exit(7)"))
    outcome, detail = provision.down(provision_spec, {"host": "h"})
    # The machine's fate is data: a teardown failure must never mask the
    # dispatch result that the caller is already handling.
    assert outcome == "failed"
    assert "7" in detail and "gone" in detail


def test_down_undeclared_still_records_an_outcome(tmp_path):
    provision_spec = spec(tmp_path)
    outcome, detail = provision.down(
        provision_spec, {"host": "h"}, evidence_path=tmp_path / "down.json")
    assert outcome == "undeclared"
    assert json.loads((tmp_path / "down.json").read_text())["outcome"] == "undeclared"
