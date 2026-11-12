"""Extracted rollout core (P2): config mapping, slot scheduling, boundaries."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_runtime_config_maps_environment(tmp_path, monkeypatch):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = {
        "RUN_ID": "run-1", "RUN_DIR": str(tmp_path),
        "HARBORRL_NATIVE_CATALOG": str(tmp_path / "catalog.json"),
        "HARBORRL_NATIVE_PROFILE": str(tmp_path / "profile.json"),
        "HARBORRL_NATIVE_GATEWAY_HOST": "127.0.0.1",
        "HARBORRL_NATIVE_GATEWAY_PORT": "49183",
        "HARBORRL_NATIVE_GATEWAY_URL": "http://gateway:49183",
        "HARBORRL_NATIVE_RUNNER_PYTHON": "/runner/python",
        "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(["worker-a", "worker-b"]),
        "HARBORRL_NATIVE_TASK_WORKER_OVERRIDES": json.dumps({"task": "worker-b"}),
        "HARBORRL_NATIVE_MAX_ATTEMPTS": "3",
        "ROLLOUT_MAX_RESPONSE_LEN": "128",
        "ROLLOUT_MAX_CONTEXT_LEN": "1024",
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": "0" * 64,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": "1" * 64,
        "HARBORRL_NATIVE_AUDITED_LOGPROBS": "1",
        "HARBORRL_NATIVE_DRAIN_TIMEOUT": "5",
        "MODEL_TAG": "qwen3-8B",
        "HARBORRL_TRAINING_SEED": "0",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    config = RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://router:30000/", concurrency_limit=4, group_size=2)
    assert config.workers == ("worker-a", "worker-b")
    assert config.task_worker_overrides == {"task": "worker-b"}
    assert config.max_attempts == 3 and config.max_output_tokens == 128
    assert config.endpoint == "http://router:30000"
    assert config.sampling_seed == 0
    assert config.environment()['HARBORRL_TRAINING_SEED'] == '0'
    assert config.environment()["HARBORRL_NATIVE_RUNNER_WORKERS"] == \
        json.dumps(["worker-a", "worker-b"])
    with pytest.raises(ValueError, match="group size"):
        RolloutRuntimeConfig.from_environment(environment, endpoint="x", concurrency_limit=1,
                                              group_size=1)
    stripped = dict(environment)
    stripped.pop("HARBORRL_NATIVE_RUNNER_WORKERS")
    with pytest.raises(ValueError, match="RUNNER_WORKERS"):
        RolloutRuntimeConfig.from_environment(stripped, endpoint="x",
                                              concurrency_limit=1, group_size=2)


def test_runtime_binds_gateway_with_registry_and_backend(tmp_path, monkeypatch):
    from harborrl.gateway import server as gateway_server
    from harborrl.gateway import sglang_backend, trace
    from harborrl.rollout import runtime as rollout_runtime

    catalog = tmp_path / "catalog.json"
    catalog.write_text("[]")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"model": "policy"}))
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [1, 2],
        get_vocab=lambda: {"token": 1},
    )
    tokenizer_digest, template_digest = rollout_runtime.tokenizer_fingerprint(tokenizer)
    environment = {
        "RUN_DIR": str(tmp_path), "RUN_ID": "run-1",
        "HARBORRL_NATIVE_CATALOG": str(catalog),
        "HARBORRL_NATIVE_PROFILE": str(profile),
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": tokenizer_digest,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": template_digest,
        "HARBORRL_NATIVE_GATEWAY_HOST": "127.0.0.1",
        "HARBORRL_NATIVE_GATEWAY_PORT": "49183",
        "HARBORRL_NATIVE_GATEWAY_URL": "http://gateway:49183",
        "HARBORRL_NATIVE_RUNNER_PYTHON": "/runner/python",
        "HARBORRL_NATIVE_RUNNER_WORKERS": '["worker-a"]',
        "ROLLOUT_MAX_RESPONSE_LEN": "128",
        "ROLLOUT_MAX_CONTEXT_LEN": "1024",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    calls = {}

    class FakeGateway:
        def __init__(self, registry, backend, *, model, max_output_tokens):
            calls["gateway"] = (registry, backend, model, max_output_tokens)
            calls["gateway_object"] = self

    class FakeServer:
        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    class FakeTraceRegistry:
        def __init__(self, path):
            calls["registry_path"] = path

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            calls["backend_args"] = (args, kwargs)

        def weight_version(self):
            return "7"

    def fake_make_server(gateway, host, port, *, max_inflight=None):
        calls["server"] = (gateway, host, port)
        calls["server_max_inflight"] = max_inflight
        return FakeServer()

    monkeypatch.setattr(gateway_server, "Gateway", FakeGateway)
    monkeypatch.setattr(gateway_server, "make_server", fake_make_server)
    monkeypatch.setattr(sglang_backend, "SGLangBackend", FakeBackend)
    monkeypatch.setattr(trace, "TraceRegistry", FakeTraceRegistry)

    config = rollout_runtime.RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://127.0.0.1:30000", concurrency_limit=4, group_size=2)
    runtime_ = rollout_runtime.RolloutRuntime(config, tokenizer)
    try:
        assert calls["gateway"][2:] == ("policy", 128)
        assert calls["server"][0] is calls["gateway_object"]
        assert calls["server"][1:] == ("127.0.0.1", 49183)
        # The implicit single gateway keeps the historical trace layout.
        assert calls["registry_path"] == runtime_.run_root / "native-traces"
        # Twice the scheduler budget (concurrency_limit=4) with a floor of 4.
        assert calls["server_max_inflight"] == 8
        assert calls["backend_args"][0][0] == "http://127.0.0.1:30000"
        assert runtime_.backend.weight_version() == "7"
    finally:
        runtime_.close()

    # A tokenizer that disagrees with the native lock fails closed.
    other = SimpleNamespace(apply_chat_template=lambda *a, **k: [9],
                            get_vocab=lambda: {"other": 2})
    with pytest.raises(ValueError, match="digest"):
        rollout_runtime.RolloutRuntime(config, other)


def test_policy_identity_uses_synchronized_pool_version(tmp_path, monkeypatch):
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
    runtime_ = SimpleNamespace(run_root=tmp_path, backend=Backend(), registry=registry,
                               config=SimpleNamespace(drain_timeout=30))
    assert rollout_runtime.current_policy_version(runtime_) == "2"
    payload = json.loads((tmp_path / "policy-pool.json").read_text())
    assert payload["weight_version"] == "2"
    assert registry.published[1] == "2"
    assert registry.timeout == 30


def test_group_shares_slot_state_and_shifts_retry_workers(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as rollout_runtime

    entry = {"id": "task", "revision": "1", "path": "/task", "task_digest": "d" * 64,
             "reward_profile": {"key": "reward", "scale": 1.0, "offset": 0.0,
                                "raw_range": [0.0, 1.0]}}
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
            return SimpleNamespace(attempt_id=f"{slot}-{self.started[slot]}",
                                   group_id="group-0", slot_id=slot,
                                   attempt_no=self.started[slot])

        def finish(self, slot, ir=None, error=None):
            assert (ir is None) != (error is None)

    created_groups = []

    def fake_group_slots(identities, *, max_attempts=2):
        created_groups.append(identities)
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

    # The run owns its loaded profile; group setup must not reread the source file.
    profile.unlink()
    results = asyncio.run(rollout_runtime.run_group(
        config, runtime_, entry=entry, group_index=0, rollout_id=0,
        sampling_digest="sampling", policy_version="1"))
    assert len(results) == 4
    assert len(created_groups) == 1
    assert len(created_groups[0]) == 4
    assert {item.harness_digest for item in created_groups[0]} == {
        rollout_runtime.digest(runtime_.profile)}
    by_slot = {}
    for slot, host in calls:
        by_slot.setdefault(slot, []).append(host)
    assert by_slot == {"0": ["worker-a"], "1": ["worker-b", "worker-a"],
                       "2": ["worker-a"], "3": ["worker-b"]}


def test_task_worker_override_wins(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as rollout_runtime

    entry = {"id": "task", "revision": "1", "path": "/task", "task_digest": "d" * 64,
             "reward_profile": {"key": "reward"}}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"model": "policy"}))
    environment = {
        "RUN_ID": "run", "RUN_DIR": str(tmp_path), "HARBORRL_NATIVE_PROFILE": str(profile),
        "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(["worker-a", "worker-b"]),
        "HARBORRL_NATIVE_TASK_WORKER_OVERRIDES": json.dumps({"task": "worker-b"}),
        "HARBORRL_NATIVE_GATEWAY_URL": "http://gateway:32123",
        "HARBORRL_NATIVE_GATEWAY_PORT": "32123", "HARBORRL_NATIVE_RUNNER_PYTHON": "/python",
        # Fake runtimes carry no lease bookkeeping: pin static routing.
        "HARBORRL_NATIVE_GATEWAY_ROUTING": "modulo",
        "HARBORRL_NATIVE_CATALOG": str(tmp_path / "catalog.json"),
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": "0" * 64,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": "1" * 64,
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    config = rollout_runtime.RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://router:30000", concurrency_limit=2, group_size=2)

    seen = []

    class PermissiveSlots:
        max_attempts = 2

        def __init__(self, identities, *, max_attempts=2):
            self.identities = {identity.slot_id: identity for identity in identities}

        def start(self, slot):
            return self.identities[slot]

        def finish(self, slot, ir=None, error=None):
            assert (ir is None) != (error is None)

    async def fake_run_attempt(identity, *, runner_host, **kwargs):
        seen.append(runner_host)
        return {"identity": identity.to_dict()}

    monkeypatch.setattr(rollout_runtime, "GroupSlots", PermissiveSlots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", fake_run_attempt)
    runtime_ = SimpleNamespace(
        rollout_root=tmp_path, semaphore=asyncio.Semaphore(2),
        shards=(SimpleNamespace(name="gateway", url="http://gateway:32123",
                                registry=SimpleNamespace()),),
        profile={"model": "policy"})
    irs = asyncio.run(rollout_runtime.run_group(
        config, runtime_, entry=entry, group_index=0, rollout_id=0,
        sampling_digest="sampling", policy_version="1"))
    assert len(irs) == 2
    assert seen == ["worker-b", "worker-b"]


def test_rollout_core_does_not_import_training_frameworks():
    import re

    import harborrl.rollout

    root = Path(harborrl.rollout.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"\s*(from|import)\s+(slime|ray)(\.|\s|$)", line):
                offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
    assert not offenders, "rollout core must stay framework-free: " + "; ".join(offenders)


def test_native_generate_shim_reexports_the_core():
    from harborrl import rollout as package
    from harborrl.rollout import native_generate, runtime

    for name in ("NativeRolloutRuntime", "RolloutRuntime", "RolloutRuntimeConfig",
                 "close_runtime", "current_policy_version", "publish_native_version",
                 "run_attempt", "run_group", "run_slot", "runtime", "task_entry",
                 "tokenizer_fingerprint", "GroupSlots"):
        assert getattr(native_generate, name) is getattr(runtime, name), name
    assert package.native_generate.__doc__ is not None
