"""CPU lifecycle failures: no Harbor, Docker or GPU needed."""
import asyncio
import json
import sys
from dataclasses import replace

import pytest
from test_native_contracts import identity

from harborrl.gateway.trace import TraceRegistry
from harborrl.rollout.harbor_job.bindings import claude_config, sanitized_runner_env
from harborrl.rollout.harbor_job.coordinator import GroupSlots, run_attempt
from harborrl.trajectories.native import RewardProfile

PROFILE = {"cli_version": "2.1.0", "model": "policy", "max_turns": 4,
           "agent_timeout_sec": 60, "setup_timeout_sec": 60, "verifier_timeout_sec": 60}


def test_credential_cannot_rebind_another_attempt(tmp_path):
    registry = TraceRegistry(tmp_path)
    registry.register(identity(), "trial", credential="secret")
    for closed in (False, True):
        if closed:
            registry.close(identity().attempt_id)
        with pytest.raises(ValueError, match="unique"):
            registry.register(identity("1"), "other", credential="secret")
    assert identity("1").attempt_id not in registry.attempts


def test_restart_cannot_reopen_persisted_attempt(tmp_path):
    TraceRegistry(tmp_path).register(identity(), "trial")
    with pytest.raises(FileExistsError):
        TraceRegistry(tmp_path).register(identity(), "trial")


def test_profile_controls_reach_agent_and_host():
    config = claude_config(PROFILE, "http://gateway:8000", "token")
    assert config["kwargs"]["disallowed_tools"] == "Agent,Task"
    disabled = claude_config({**PROFILE, "disallowed_tools": "Agent,Task,WebSearch,WebFetch"},
                             "http://gateway:8000", "token")
    assert disabled["kwargs"]["disallowed_tools"] == "Agent,Task,WebSearch,WebFetch"
    assert config["env"]["DISABLE_AUTO_COMPACT"] == "1"
    assert config["env"]["ANTHROPIC_MODEL"] == "policy"
    assert config["env"]["NO_PROXY"] == "gateway"
    assert config["env"]["no_proxy"] == "gateway"
    assert config["kwargs"]["version"] == "2.1.0"
    assert sanitized_runner_env({"PATH": "/bin", "ANTHROPIC_API_KEY": "external", "HTTPS_PROXY": "external"})["PATH"] == "/bin"
    assert "ANTHROPIC_API_KEY" not in sanitized_runner_env({"ANTHROPIC_API_KEY": "external"})
    with pytest.raises(ValueError):
        claude_config({**PROFILE, "cli_version": "latest"}, "http://gateway", "token")
    with pytest.raises(ValueError):
        claude_config(PROFILE, "http://gateway/v1", "token")


def test_explicit_harness_prompt_preserves_literal_content():
    prompt = "Use the declared tools.\nFinish with a final response; preserve 'quotes' and $literal."
    config = claude_config({**PROFILE, "append_system_prompt": prompt}, "http://gateway", "token")
    assert config["kwargs"]["append_system_prompt"] == prompt
    assert "append_system_prompt" not in claude_config(PROFILE, "http://gateway", "token")["kwargs"]


@pytest.mark.parametrize('base', [None, 300, 600.0, 1200])
def test_environment_timeout_is_separate_from_agent_setup(base):
    from harborrl.rollout.harbor_job.bindings import environment_timeouts

    resources = {} if base is None else {'build_timeout_sec': base}
    original = dict(resources)
    assert environment_timeouts(PROFILE, resources) == {}
    profile = {**PROFILE, 'environment_timeout_sec': 720}
    config = claude_config(profile, 'http://gateway', 'token')
    assert config['override_setup_timeout_sec'] == 60
    assert 'environment_timeout_sec' not in config['kwargs']
    multiplier = environment_timeouts(profile, resources)['environment_build_timeout_multiplier']
    assert multiplier * (600 if base is None else base) == pytest.approx(720)
    assert resources == original


@pytest.mark.parametrize('budget', [None, False, 0, -1, 1.5, '720'])
def test_invalid_environment_timeout_is_rejected(budget):
    from harborrl.rollout.harbor_job.bindings import environment_timeouts

    profile = {**PROFILE, 'environment_timeout_sec': budget}
    with pytest.raises(ValueError, match='environment_timeout_sec'):
        claude_config(profile, 'http://gateway', 'token')
    with pytest.raises(ValueError, match='environment_timeout_sec'):
        environment_timeouts(profile, {})


@pytest.mark.parametrize('base', [None, False, 0, -1, float('nan'), float('inf'), '300'])
def test_invalid_task_environment_timeout_cannot_be_overridden(base):
    from harborrl.rollout.harbor_job.bindings import environment_timeouts

    with pytest.raises(ValueError, match='task environment'):
        environment_timeouts({**PROFILE, 'environment_timeout_sec': 720}, {'build_timeout_sec': base})


@pytest.mark.parametrize("prompt", [None, False, 1, "", " \n", "text\x00text", ["text"]])
def test_invalid_harness_prompt_is_rejected(prompt):
    with pytest.raises(ValueError, match="append_system_prompt"):
        claude_config({**PROFILE, "append_system_prompt": prompt}, "http://gateway", "token")


def test_retry_budget_and_new_lineage():
    group = GroupSlots([identity(), identity("1")], max_attempts=2)
    first = group.start("0")
    with pytest.raises(ValueError, match="running"):
        group.start("0")
    group.finish("0", error="transport")
    second = group.start("0")
    assert first.attempt_id != second.attempt_id
    assert first.trajectory_id != second.trajectory_id
    assert first.group_key == second.group_key
    assert group.history["0"][-1]["parent_attempt_id"] == first.attempt_id
    group.finish("0", error="transport")
    with pytest.raises(ValueError, match="exhausted"):
        group.start("0")
    with pytest.raises(ValueError, match="incomplete"):
        group.export()
    with pytest.raises(ValueError, match="slots"):
        GroupSlots([identity(), replace(identity(), slot_id="1")])


def test_runner_can_be_dispatched_to_external_worker():
    from harborrl.rollout.harbor_job.coordinator import runner_command
    local = runner_command("/opt/python", "/repo")
    remote = runner_command("/opt/python", "/re po", "worker-a")
    assert local == ["/opt/python", "-m", "harborrl.rollout.harbor_job.runner"]
    assert remote == ["ssh", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
                      "-o", "StrictHostKeyChecking=accept-new", "worker-a",
                      "cd '/re po' && exec /opt/python -m harborrl.rollout.harbor_job.runner"]


def test_real_runner_source_compiles():
    import py_compile

    from harborrl.rollout.harbor_job import runner
    py_compile.compile(runner.__file__, doraise=True)


def test_runner_launch_failure_is_recorded(tmp_path):
    root = tmp_path / "attempt"
    with pytest.raises(FileNotFoundError):
        asyncio.run(run_attempt(identity(), task_path=tmp_path, profile=PROFILE,
            gateway_url="http://gateway", runner_python=str(tmp_path / "missing-python"),
            root=root, registry=TraceRegistry(tmp_path / "trace"), reward_profile=RewardProfile()))
    assert (root / "launch-failed.json").is_file()


@pytest.mark.parametrize("mode", ["crash", "wrong-event", "timeout"])
def test_subprocess_failure_closes_admission_and_records_cleanup(tmp_path, monkeypatch, mode):
    # Exercise the real subprocess pipes with an explicitly synthetic worker.
    worker = tmp_path / "worker.py"
    worker.write_text('''import json, sys, time
req = json.loads(sys.stdin.readline())
print(json.dumps({"type": "prepared", "trial_id": "trial"}), flush=True)
sys.stdin.readline()
''' + {
        "crash": 'sys.exit(7)\n',
        "wrong-event": 'print(json.dumps({"type":"event", "trial_id":"other"}), flush=True)\nsys.stdin.readline()\n',
        "timeout": 'time.sleep(30)\n',
    }[mode])
    spawn = asyncio.create_subprocess_exec
    async def fake_spawn(*args, **kwargs):
        return await spawn(sys.executable, "-S", str(worker), **kwargs)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    registry = TraceRegistry(tmp_path / "traces")
    root = tmp_path / "attempt"
    with pytest.raises((RuntimeError, ValueError, asyncio.TimeoutError)):
        asyncio.run(run_attempt(identity(), task_path=tmp_path, profile=PROFILE,
            gateway_url="http://gateway", runner_python=sys.executable, root=root,
            registry=registry, reward_profile=RewardProfile(), timeout=2, cleanup_timeout=.5))
    assert registry.attempts[identity().attempt_id]["closed"]
    assert (root / "failure.json").is_file()
    if mode != "wrong-event":
        assert (root / "cleanup-required.json").is_file()
    assert json.loads((root / "runner-events.json").read_text())[0]["type"] == "prepared"


@pytest.mark.parametrize('mode,existing_marker', [('terminal', False), ('crash', False), ('crash', True)])
def test_cleanup_marker_preserves_original_failure(tmp_path, monkeypatch, mode, existing_marker):
    from harborrl.trajectories.native import digest, publish, read_json

    root = tmp_path / 'attempt'
    marker = root / 'cleanup-required.json'
    sandbox = {'create_timeout_s': 1, 'cleanup_timeout_s': 1}
    worker = tmp_path / 'worker.py'
    terminal = {'type': 'terminal', 'trial_id': 'trial'}
    worker.write_text(chr(10).join(['import sys, json', 'json.loads(sys.stdin.readline())',
        'print(json.dumps({"type": "prepared", "trial_id": "trial"}), flush=True)',
        'sys.stdin.readline()', 'sys.exit(7)' if mode == 'crash' else
        f'print({json.dumps(terminal)!r}, flush=True)', '']))
    spawn = asyncio.create_subprocess_exec

    async def fake_spawn(*args, **kwargs):
        return await spawn(sys.executable, '-S', str(worker), **kwargs)

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_spawn)
    registry = TraceRegistry(tmp_path / 'traces')
    register = registry.register
    prior = {'identity': identity().to_dict(), 'reason': 'Existing cleanup evidence'}

    def registered(*args, **kwargs):
        publish(root / 'runner/cleanup-required.json', {'reason': 'Unconfirmed platform deletion'})
        publish(root / 'runner/sandbox-lifecycle.json',
                {'state': 'delete_failed', 'runtime_digest': digest(sandbox)})
        if existing_marker:
            publish(marker, prior)
        return register(*args, **kwargs)

    monkeypatch.setattr(registry, 'register', registered)
    reason = 'sandbox deletion unconfirmed' if mode == 'terminal' else 'Runner exited before terminal result'
    with pytest.raises(RuntimeError, match=reason):
        asyncio.run(run_attempt(identity(), task_path=tmp_path, profile=PROFILE,
            gateway_url='http://gateway', runner_python=sys.executable, root=root,
            registry=registry, reward_profile=RewardProfile(), sandbox=sandbox))
    expected = prior if existing_marker else {'identity': identity().to_dict(), 'reason':
        'Sandbox deletion unconfirmed' if mode == 'terminal' else 'Inspect runner/cleanup-required.json'}
    assert read_json(marker) == expected
    assert read_json(root / 'failure.json')['error_type'] == 'RuntimeError'
    assert read_json(root / 'runner/cleanup-required.json') == {'reason': 'Unconfirmed platform deletion'}
    assert registry.attempts[identity().attempt_id]['closed']
    assert (root / 'runner-events.json').exists()
    assert not (root / 'trajectory.v2.json').exists()


def test_offline_http_to_export_is_explicitly_synthetic(tmp_path):
    from harborrl.rollout.harbor_job.offline import smoke
    from harborrl.trajectories.native import require_ready
    report = smoke(tmp_path / "smoke")
    assert report["status"] == "CONTRACT_READY"
    assert report["group_mean"] == .5
    assert report["turns"] == 4
    ir = json.loads((tmp_path / "smoke/attempt-0/trajectory.v2.json").read_text())
    with pytest.raises(ValueError, match="synthetic"):
        require_ready(ir)


def test_native_inspector_preserves_original_verifier(tmp_path):
    from harborrl.data.harbor.native_inspector import inspect_native
    (tmp_path / "environment").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "task.toml").write_text('version = "1.0"\n')
    (tmp_path / "instruction.md").write_text("Do the original task")
    (tmp_path / "environment/Dockerfile").write_text("FROM debian:bookworm")
    (tmp_path / "tests/test.sh").write_text('echo \'{"score":75}\' > /logs/verifier/reward.json')
    report = inspect_native(tmp_path)
    assert report["status"] == "NEEDS_PROBE"
    assert inspect_native(tmp_path, expected_digest="changed")["status"] == "INVALID"
    (tmp_path / "task.toml").write_text('[[steps]]\nname="step1"\n')
    assert "multi_step_not_enabled" in inspect_native(tmp_path)["reasons"]
