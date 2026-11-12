"""CPU contracts for schema-2 launch and Slime tensor boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from harborrl.config import launch_plan, load_config


def native_config(tmp_path, **overrides):
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("original instruction\n")
    (task / "environment/Dockerfile").write_text("FROM debian:bookworm\n")
    (task / "tests/test.sh").write_text("true\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{
        "id": "task", "revision": "1", "path": str(task),
        "task_digest": "0" * 64, "reward_profile": {"key": "reward"},
    }]))
    profile = tmp_path / "claude.json"
    profile.write_text(json.dumps({
        "cli_version": "2.1.0", "model": "policy", "max_turns": 4,
        "agent_timeout_sec": 60, "setup_timeout_sec": 60, "verifier_timeout_sec": 60,
    }))
    values = {
        "schema_version": 2,
        "execution": {"backend": "harbor_job"},
        "tasks": {"catalog": str(catalog)},
        "harness": {"name": "claude_code", "profile": str(profile)},
        "harbor": {"python": "/opt/harbor/bin/python", "version": "0.23.0",
                    "workers": ["worker-a", "worker-b"]},
        "gateway": {"host": "gateway", "port": 49183,
                    "advertised_url": "http://gateway:49183",
                    "tokenizer_digest": "1" * 64, "template_digest": "2" * 64,
                    "audited_raw_logprobs": True},
        "model": {"checkpoint": "/model", "reference": "/reference", "args_file": "qwen3-8B"},
        "training": {"num_rollout": 1, "learning_rate": 0.0001, "save_interval": 1},
        "sampling": {"group_size": 2, "groups_per_batch": 1, "max_attempts": 2,
                     "max_tokens": 128, "max_context": 1024},
        "deployment": {"layout": "split", "num_gpus": 8, "actor_gpus": 4,
                       "rollout_gpus": 4, "actor_tensor_parallel_size": 4,
                       "rollout_gpus_per_engine": 4},
        "output": {"root": str(tmp_path / "runs")},
    }
    values.update(overrides)
    path = tmp_path / "native.yaml"
    lines = ["schema_version: 2"]
    for section, fields in values.items():
        if section == "schema_version":
            continue
        lines.append(f"{section}: {json.dumps(fields)}")
    path.write_text("\n".join(lines) + "\n")
    return path, task


def test_schema2_launch_plan_and_strict_override(tmp_path, capsys):
    path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native
    digest = inspect_native(task)["task_digest"]
    catalog_path = json.loads(path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"]
    catalog = json.loads(Path(catalog_path).read_text())
    catalog[0]["task_digest"] = digest
    Path(catalog_path).write_text(json.dumps(catalog))

    runner = tmp_path / "venv" / "python"
    runner.parent.mkdir()
    runner.symlink_to("/opt/absolute-base/python")
    path.write_text(path.read_text().replace("/opt/harbor/bin/python", str(runner)))
    assert load_config(path)["harbor"]["python"] == str(runner)

    plan = launch_plan(load_config(path))
    assert plan["environment"]["HARBORRL_NATIVE_ROLLOUT"] == "1"
    assert plan["environment"]["HARBORRL_NATIVE_SLIME_CONTRACT"] == "slime-v032"
    with pytest.raises(ValueError, match="backend_contract"):
        load_config(path, ["training.backend_contract=unknown"])
    assert plan["environment"]["HARBORRL_NATIVE_AUDITED_LOGPROBS"] == "1"
    assert json.loads(plan["environment"]["HARBORRL_NATIVE_RUNNER_WORKERS"]) == ["worker-a", "worker-b"]
    assert plan["environment"]["N_SAMPLES"] == "2"
    assert plan["environment"]["TRAIN_PYTHON"] == sys.executable
    assert 'HARBORRL_TRAINING_SEED' not in plan['environment']
    for seed in (0, 42, 2147483647):
        seeded = launch_plan(load_config(path, [f'training.seed={seed}']))
        assert seeded['environment']['HARBORRL_TRAINING_SEED'] == str(seed)
    for value in ('true', '-1', '2147483648', '1.5'):
        with pytest.raises(ValueError, match='training.seed'):
            load_config(path, [f'training.seed={value}'])
    from harborrl.platform.evidence import run_environment
    prompt_path = tmp_path / 'prompts.jsonl'
    assert 'SAVE_HF' not in run_environment(plan, tmp_path, Path(catalog_path), prompt_path)
    exported = launch_plan(load_config(path, ['training.export_hf=true']))
    assert run_environment(exported, tmp_path, Path(catalog_path), prompt_path)['SAVE_HF'] == str(
        tmp_path / 'hf' / 'rollout-{rollout_id}')
    for value in ('1', 'null', '"true"'):
        with pytest.raises(ValueError, match='training.export_hf'):
            load_config(path, [f'training.export_hf={value}'])
    assert json.loads(plan["environment"]["HARBORRL_NATIVE_TASK_WORKER_OVERRIDES"]) == {}
    assert plan["training_command"][0] == "bash"
    assert plan["launcher"].startswith("native Claude Code")
    from harborrl.cli import main
    assert main(["train", "--config", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["schema_version"] == 2

    config = load_config(path, ["sampling.group_size=3"])
    assert config["sampling"]["group_size"] == 3
    with pytest.raises(ValueError, match="unknown native override"):
        load_config(path, ["sampling.unknown=3"])
    with pytest.raises(ValueError, match="GPU budget"):
        load_config(path, ["deployment.rollout_gpus=5"])
    config["harbor"]["workers"] = ["bad worker"]
    path.write_text(path.read_text().replace(
        json.dumps(["worker-a", "worker-b"]), json.dumps(["bad worker"])))
    with pytest.raises(ValueError, match="harbor.workers"):
        load_config(path)



def test_cli_command_mode_selects_experiment_dimensions(tmp_path, capsys, monkeypatch):
    path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native
    digest = inspect_native(task)["task_digest"]
    catalog_path = json.loads(path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"]
    catalog = json.loads(Path(catalog_path).read_text())
    catalog[0]["task_digest"] = digest
    Path(catalog_path).write_text(json.dumps(catalog))
    private_config = tmp_path / "config.yaml"
    private_config.write_text(path.read_text())
    monkeypatch.chdir(tmp_path)

    from harborrl.config.native import catalog as native_catalog
    selected = native_catalog(load_config(private_config, task_ids=["task"]))
    assert [entry["id"] for entry in selected] == ["task"]

    from harborrl.cli import main
    assert main([
        "train",
        "-model", "qwen3-8B",
        "-harness", "claude-code",
        "-task", "task",
        "-alg", "grpo",
        "--dry-run",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["config"]["model"]["args_file"] == "qwen3-8B"
    assert plan["config"]["harness"]["name"] == "claude_code"

    assert main(["train", str(private_config), "--dry-run"]) == 0
    positional_plan = json.loads(capsys.readouterr().out)
    assert positional_plan["config"]["schema_version"] == 2


@pytest.mark.parametrize('overrides, expected_lr', [
    ([], '0.0001'),
    (['training.learning_rate=0.000025'], '2.5e-05'),
])
def test_native_shell_dry_run_uses_official_v032_hooks(tmp_path, overrides, expected_lr):
    config_path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native
    config = load_config(config_path, overrides)
    catalog_path = Path(config['tasks']['catalog'])
    entries = json.loads(catalog_path.read_text())
    entries[0]['task_digest'] = inspect_native(task)['task_digest']
    catalog_path.write_text(json.dumps(entries))
    plan = launch_plan(config)
    prompt = tmp_path / "native.jsonl"
    prompt.write_text('{"task":"fixture","metadata":{"harborrl_native":true}}\n')
    root = tmp_path / "run"
    slime = tmp_path / "slime"
    model_args = slime / "scripts" / "models"
    model_args.mkdir(parents=True)
    (slime / "train.py").write_text("# official Slime v0.3.2 entrypoint\n")
    (model_args / "qwen3-8B.sh").write_text("MODEL_ARGS=(--hidden-size 4096)\n")
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR"}}
    env.update(plan["environment"])
    env.update({
        "DRY_RUN": "1", "DATASET": "native", "ALGO": "grpo",
        "HARNESS_OPTION": "native-claude-code", "HARBORRL_NATIVE_ROLLOUT": "1",
        "HARBORRL_SKIP_GLOBAL_CLEANUP": "1", "ROLLOUT_PROMPT_DATA": str(prompt),
        "HF_CKPT": "/model", "REF_LOAD": "/reference", "MODEL_ARGS_FILE": "qwen3-8B",
        "MODEL_TAG": "qwen3-8B", "NUM_GPUS": "8", "ACTOR_GPUS": "4",
        "ROLLOUT_GPUS": "4", "TP_SIZE": "4", "ROLLOUT_NUM_GPUS_PER_ENGINE": "4",
        "HARBORRL_GPU_LAYOUT": "split", "NUM_ROLLOUT": "1", "SAVE_INTERVAL": "1",
        "RUN_DIR": str(root), "RUN_ID": "run",
        "RUNS_ROOT": str(tmp_path), "CKPT_ROOT": str(root / "checkpoints"),
        "HARBORRL_NATIVE_MAX_ATTEMPTS": "2", "HARBORRL_NATIVE_DRAIN_TIMEOUT": "1",
        "HARBORRL_NATIVE_RUNNER_WORKERS": '["worker-a", "worker-b"]',
        "HARBORRL_NATIVE_GATEWAY_HOST": "0.0.0.0",
        "HARBORRL_NATIVE_SLIME_CONTRACT": "slime-v032",
        "SLIME_DIR": str(slime),
        "TRAIN_PYTHON": sys.executable,
        "HARBORRL_TRAINING_SEED": "0",
    })
    env['LR'] = plan['environment']['LR']
    script = Path(__file__).resolve().parents[2] / "harborrl/platform/slime_train.sh"
    output = tmp_path / "shell.log"
    with output.open("w") as stream:
        result = subprocess.run(["bash", str(script)], stdout=stream, stderr=subprocess.STDOUT,
                                env=env, check=False, timeout=30)
    result.stdout = output.read_text()
    assert result.returncode == 0, result.stdout
    command = next(line for line in result.stdout.splitlines() if line.startswith("[dry-run] "))
    assert "--loss-type custom_loss" in command
    import shlex
    arguments = shlex.split(command.removeprefix('[dry-run] '))
    assert arguments[arguments.index('--lr') + 1] == expected_lr
    expected_flags = {
        '--num-rollout': '1', '--save-interval': '1',
        '--rollout-batch-size': '1', '--n-samples-per-prompt': '2',
        '--rollout-max-response-len': '128', '--rollout-max-context-len': '1024',
        '--num-gpus-per-node': '8', '--actor-num-gpus-per-node': '4',
        '--rollout-num-gpus': '4', '--tensor-model-parallel-size': '4',
        '--rollout-num-gpus-per-engine': '4', '--hf-checkpoint': '/model',
        '--ref-load': '/reference', '--prompt-data': str(prompt),
        '--save': str(root / 'checkpoints' / 'run'),
        '--seed': '0', '--rollout-seed': '0',
    }
    for flag, value in expected_flags.items():
        assert arguments[arguments.index(flag) + 1] == value
    recorded = json.loads(next((root / 'dry-run-commands').glob('*.json')).read_text())
    assert recorded['command'] == arguments
    assert recorded['mode'] == 'dry_run'
    for argument in (
        "--rollout-function-path harborrl.backends.slime_v032.rollout.generate_rollout",
        "--custom-convert-samples-to-train-data-path harborrl.backends.slime_v032.converter.convert_samples_to_train_data",
        "--custom-advantage-function-path harborrl.backends.slime_v032.advantage.compute_advantages_and_returns",
        "--custom-loss-function-path harborrl.backends.slime_v032.loss.loss_function",
        "--rollout-data-postprocess-path harborrl.backends.slime_v032.postprocess.rollout_data_postprocess",
    ):
        assert argument in command
    assert "native_slime" not in command
    assert "--use-rollout-logprobs" in command
    assert '--sglang-enable-deterministic-inference' in command
    assert "--megatron-to-hf-mode bridge" in command
    assert "--no-save-optim" not in command
    assert "--load /model" in command
    assert command.count(" --load ") == 1
    assert "--num-steps-per-rollout 2" in command
    assert "--use-kl-loss" not in command
    assert "harborrl.rollout.entrypoint.generate" not in command

    env["RESUME_LOAD"] = "/checkpoint"
    with output.open("w") as stream:
        result = subprocess.run(["bash", str(script)], stdout=stream, stderr=subprocess.STDOUT,
                                env=env, check=False, timeout=30)
    result.stdout = output.read_text()
    assert result.returncode == 0, result.stdout
    command = next(line for line in result.stdout.splitlines() if line.startswith("[dry-run] "))
    assert "--load /checkpoint" in command
    assert "--no-load-optim" not in command
    assert "--use-checkpoint-opt-param-scheduler" in command


def test_slime_runtime_config_maps_args_and_environment(tmp_path, monkeypatch):
    from harborrl.backends.slime_v032 import rollout as slime_rollout
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"model": "policy"}))
    environment = {
        "RUN_ID": "run-1", "RUN_DIR": str(tmp_path),
        "HARBORRL_NATIVE_CATALOG": str(catalog),
        "HARBORRL_NATIVE_PROFILE": str(profile),
        "HARBORRL_NATIVE_GATEWAY_HOST": "127.0.0.1",
        "HARBORRL_NATIVE_GATEWAY_PORT": "49183",
        "HARBORRL_NATIVE_GATEWAY_URL": "http://gateway:49183",
        "HARBORRL_NATIVE_RUNNER_PYTHON": "/runner/python",
        "HARBORRL_NATIVE_RUNNER_WORKERS": '["worker-a", "worker-b"]',
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": "0" * 64,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": "1" * 64,
        "ROLLOUT_MAX_RESPONSE_LEN": "128", "ROLLOUT_MAX_CONTEXT_LEN": "1024",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    args = SimpleNamespace(rollout_num_gpus=8, rollout_num_gpus_per_engine=4,
                           sglang_server_concurrency=2, hf_checkpoint="/model",
                           sglang_router_ip="127.0.0.1", sglang_router_port=30000,
                           rollout_max_context_len=1024, rollout_max_response_len=128,
                           n_samples_per_prompt=4)
    config = slime_rollout.runtime_config_from_args(args)
    assert isinstance(config, RolloutRuntimeConfig)
    assert config.endpoint == "http://127.0.0.1:30000"
    assert config.concurrency_limit == 4  # 2 concurrency x 2 engines
    assert config.group_size == 4
    assert config.workers == ("worker-a", "worker-b")
    assert config.max_output_tokens == 128

    # The tokenizer loader keeps the single Slime dependency at the boundary.
    slime = ModuleType("slime")
    utils = ModuleType("slime.utils")
    processing = ModuleType("slime.utils.processing_utils")
    tokenizer = SimpleNamespace()
    processing.load_tokenizer = lambda *args, **kwargs: tokenizer
    slime.utils = utils
    utils.processing_utils = processing
    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", processing)
    assert slime_rollout.slime_tokenizer_loader(args)() is tokenizer

def test_native_policy_identity_uses_synchronized_pool_version(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as rollout_runtime

    class Backend:
        endpoint = "http://router:30000"
        def pool_states(self):
            return [{"endpoint": self.endpoint, "healthy": True, "weight_version": "2"}]

    class Registry:
        def __init__(self):
            self.published = None
        def drain(self, timeout):
            self.timeout = timeout
        def publish_version(self, states, expected):
            self.published = (states, expected)

    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    registry = Registry()
    runtime = SimpleNamespace(run_root=tmp_path, backend=Backend(), registry=registry,
                              config=SimpleNamespace(drain_timeout=30))
    assert rollout_runtime.current_policy_version(runtime) == "2"
    payload = json.loads((tmp_path / "policy-pool.json").read_text())
    assert payload["weight_version"] == "2"
    assert registry.published[1] == "2"
    assert registry.timeout == 30


def test_native_slots_spread_workers_and_shift_retries(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as rollout_runtime

    entry = {"id": "task", "revision": "1", "path": "/task", "task_digest": "d" * 64,
             "reward_profile": {"key": "reward", "scale": 1.0, "offset": 0.0, "raw_range": [0.0, 1.0]}}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"model": "policy"}))
    environment = {
        "RUN_ID": "run", "RUN_DIR": str(tmp_path), "HARBORRL_NATIVE_PROFILE": str(profile),
        "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(["worker-a", "worker-b"]),
        "HARBORRL_NATIVE_MAX_ATTEMPTS": "2",
        "HARBORRL_NATIVE_GATEWAY_URL": "http://gateway:32123",
        "HARBORRL_NATIVE_GATEWAY_PORT": "32123",
        # Fake runtimes carry no lease bookkeeping: pin static routing.
        "HARBORRL_NATIVE_GATEWAY_ROUTING": "modulo",
        "HARBORRL_NATIVE_RUNNER_PYTHON": "/python",
        "HARBORRL_NATIVE_CATALOG": str(tmp_path / "catalog.json"),
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": "0" * 64,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": "1" * 64,
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    config = rollout_runtime.RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://router:30000", concurrency_limit=8, group_size=4)

    class FakeSlots:
        max_attempts = 2

        def __init__(self):
            self.started = {}

        def start(self, slot):
            self.started[slot] = self.started.get(slot, 0) + 1
            return SimpleNamespace(attempt_id=f"{slot}-{self.started[slot]}", group_id="group-0",
                                   slot_id=slot, attempt_no=self.started[slot])

        def finish(self, slot, ir=None, error=None):
            assert (ir is None) != (error is None)

    def fake_group_slots(identities, *, max_attempts=2):
        return FakeSlots()

    calls = []

    async def fake_run_attempt(identity, *, runner_host, **kwargs):
        calls.append((identity.slot_id, runner_host))
        if identity.slot_id == "1" and identity.attempt_no == 1:
            raise RuntimeError("simulated first-attempt failure")
        return {"identity": identity.__dict__}

    monkeypatch.setattr(rollout_runtime, "GroupSlots", fake_group_slots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", fake_run_attempt)

    runtime_ = SimpleNamespace(
        rollout_root=tmp_path, semaphore=asyncio.Semaphore(8),
        shards=(SimpleNamespace(name="gateway", url="http://gateway:32123",
                                registry=SimpleNamespace()),),
        profile={"model": "policy"})

    async def gather_slots():
        return await asyncio.gather(*[
            rollout_runtime.run_slot(config, runtime_, 0, slot, 0, entry, "sampling", "1")
            for slot in range(4)])

    results = asyncio.run(gather_slots())
    assert len(results) == 4
    by_slot = {}
    for slot, host in calls:
        by_slot.setdefault(slot, []).append(host)
    assert by_slot == {"0": ["worker-a"], "1": ["worker-b", "worker-a"],
                       "2": ["worker-a"], "3": ["worker-b"]}


def test_native_dispatch_passes_materialized_prompt_data(tmp_path, monkeypatch):
    path, task = native_config(tmp_path)
    from harborrl.data.harbor.native_inspector import inspect_native
    digest = inspect_native(task)["task_digest"]
    catalog_path = json.loads(path.read_text().split("tasks: ", 1)[1].split("\n", 1)[0])["catalog"]
    catalog = json.loads(Path(catalog_path).read_text())
    catalog[0]["task_digest"] = digest
    Path(catalog_path).write_text(json.dumps(catalog))

    from harborrl.platform import native_train
    plan = launch_plan(load_config(path))
    output = tmp_path / "prompt-data-env"
    path_output = tmp_path / "path-env"
    workers_output = tmp_path / "workers-env"
    gateway_output = tmp_path / "gateway-env"
    backend_output = tmp_path / "backend-env"
    plan["training_command"] = [
        "bash", "-c",
        (
            f"printf %s \"$ROLLOUT_PROMPT_DATA\" > {output}; "
            f"printf %s \"$PATH\" > {path_output}; "
            f"printf %s \"$HARBORRL_NATIVE_RUNNER_WORKERS\" > {workers_output}; "
            f"printf %s \"$HARBORRL_NATIVE_GATEWAY_HOST\" > {gateway_output}; "
            f"printf %s \"$SLIME_DIR:$MEGATRON_DIR:$SGLANG_IMAGE\" > {backend_output}"
        ),
    ]
    monkeypatch.setenv("SLIME_DIR", "/opt/slime-v0.3.2")
    monkeypatch.setenv("MEGATRON_DIR", "/opt/megatron")
    monkeypatch.setenv("SGLANG_IMAGE", "sglang:v0.5.15.post1-cu129")
    monkeypatch.setenv("LR", "99")
    monkeypatch.setenv("N_SAMPLES", "99")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-provider-credential")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:credential@proxy.example:8080")
    monkeypatch.setattr(native_train, "doctor", lambda *_args, **_kwargs: [])
    assert native_train.dispatch(plan) == 0
    launch_path = next((tmp_path / "runs" / "training").glob("*/launch.json"))
    launch = json.loads(launch_path.read_text())
    assert launch["prompt_data"] == output.read_text()
    effective_text = (launch_path.parent / "effective-config.json").read_text()
    effective = json.loads(effective_text)
    assert effective['environment']['LR'] == '0.0001'
    assert effective['environment']['N_SAMPLES'] == '2'
    assert effective['environment']['ROLLOUT_PROMPT_DATA'] == output.read_text()
    assert effective['components']['algorithm'] == 'grpo'
    assert effective['update_semantics'] == {
        'advantage': 'reward_minus_complete_group_mean',
        'std_normalization': False,
        'reduction': 'trajectory_weighted_clipped_pg_v1',
    }
    assert effective['deployment_environment']['SLIME_DIR'] == '/opt/slime-v0.3.2'
    assert 'ambient-provider-credential' not in effective_text
    assert 'user:credential' not in effective_text
    assert path_output.read_text().split(os.pathsep, 1)[0] == str(Path(plan["command"][0]).parent)
    assert json.loads(workers_output.read_text()) == ["worker-a", "worker-b"]
    assert gateway_output.read_text() == "gateway"
    assert backend_output.read_text() == (
        "/opt/slime-v0.3.2:/opt/megatron:sglang:v0.5.15.post1-cu129")


def test_all_schema_fields_have_declared_consumers():
    from harborrl.backends.slime_v032.config import field_usage
    from harborrl.config.native import FIELDS
    assert set(field_usage()) == {
        f'{section}.{field}' for section, fields in FIELDS.items() for field in fields
    }


@pytest.mark.parametrize('cancel_batch', [False, True])
@pytest.mark.parametrize('uneven_cleanup', [False, True])
@pytest.mark.parametrize('cancel_during_cleanup', [False, True])
def test_native_batch_waits_for_all_group_cleanup(
        monkeypatch, cancel_batch, uneven_cleanup, cancel_during_cleanup):
    from harborrl.backends.slime_v032 import rollout
    from harborrl.rollout import runtime as rollout_runtime

    monkeypatch.delenv('HARBORRL_ROLLOUT_SERVICE_URL', raising=False)
    cleaned, closed = [], []
    failure = RuntimeError('one group failed')

    async def scenario():
        started = set()
        all_started, blocked = asyncio.Event(), asyncio.Event()
        cleanup_started = asyncio.Event()

        async def slot(config, runtime_, group, index, entry, reward_profile):
            started.add((group, index))
            if len(started) == 4:
                all_started.set()
            try:
                await all_started.wait()
                if not cancel_batch and (group, index) == (0, 0):
                    raise failure
                await blocked.wait()
            finally:
                # Cleanup must be awaited across both levels of group tasks.
                if (group, index) == (1, 1):
                    cleanup_started.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0.02 if uneven_cleanup and index else 0)
                cleaned.append((group, index))

        async def generate(args, group, params, rollout_id):
            return await rollout_runtime.run_group(
                SimpleNamespace(group_size=2), None, entry={}, group_index=group,
                rollout_id=rollout_id, sampling_digest='sampling', policy_version='1')

        monkeypatch.setattr(rollout_runtime, '_group_slots',
                            lambda config, runtime_, group, *args: (group, None))
        monkeypatch.setattr(rollout_runtime, '_run_slot', slot)
        monkeypatch.setattr(rollout, 'generate_group', generate)
        monkeypatch.setattr(rollout, 'close_runtime', lambda: closed.append(set(cleaned)))
        batch = asyncio.create_task(rollout.run_native_groups(
            SimpleNamespace(rollout_max_response_len=128), [0, 1], '1', 0))
        await asyncio.wait_for(all_started.wait(), timeout=5)
        if cancel_batch:
            batch.cancel()
        if cancel_during_cleanup:
            await asyncio.wait_for(cleanup_started.wait(), timeout=5)
            batch.cancel()
        expected_error = asyncio.CancelledError if cancel_batch else RuntimeError
        with pytest.raises(expected_error) as exc:
            await asyncio.wait_for(batch, timeout=5)
        if not cancel_batch:
            assert exc.value is failure
        assert closed == [{(0, 0), (0, 1), (1, 0), (1, 1)}]
        assert len(cleaned) == 4

    asyncio.run(scenario())


def test_schema_validation_does_not_read_harness_files(tmp_path):
    from harborrl.config.native import validate, validate_files
    config_path, _ = native_config(tmp_path)
    config = load_config(config_path)
    Path(config['harness']['profile']).unlink()
    assert validate(config, config_path) is config
    with pytest.raises(FileNotFoundError):
        validate_files(config)


@pytest.mark.parametrize('field, value, message', [
    ('training', None, 'training'),
    ('training.backend_contract', [], 'backend_contract'),
    ('gateway.advertised_url', [], 'gateway.advertised_url'),
    ('sampling.max_tokens', 'many', 'sampling.max_tokens'),
])
def test_invalid_config_types_are_cli_errors(tmp_path, capsys, field, value, message):
    import yaml

    from harborrl.cli import main
    config_path, _ = native_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    section, _, key = field.partition('.')
    if key:
        config[section][key] = value
    else:
        config[section] = value
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit) as exc:
        main(['train', str(config_path), '--dry-run'])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_invalid_override_yaml_is_cli_error(tmp_path, capsys):
    from harborrl.cli import main
    config_path, _ = native_config(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(['train', str(config_path), '--set', 'training.learning_rate=[', '--dry-run'])
    assert exc.value.code == 2
    assert 'training.learning_rate: invalid YAML override' in capsys.readouterr().err


def test_harness_file_override_is_checked_after_override(tmp_path):
    config_path, _ = native_config(tmp_path)
    config = load_config(config_path)
    old_profile = Path(config['harness']['profile'])
    new_profile = tmp_path / 'replacement.json'
    old_profile.rename(new_profile)
    config = load_config(config_path, [f'harness.profile={new_profile}'])
    assert config['harness']['profile'] == str(new_profile)


def test_catalog_rejects_nonmapping_entry(tmp_path):
    from harborrl.tasks.catalog import catalog
    config_path, _ = native_config(tmp_path)
    config = load_config(config_path)
    Path(config['tasks']['catalog']).write_text('[null]')
    with pytest.raises(ValueError, match='catalog identity'):
        catalog(config)
