"""CPU contract evidence for static gateway sharding (same-process form).

These tests exercise the shard table, static routing and the cross-shard
admission barrier without SSH, GPU or a live rollout service: shard ports
and trace subtrees stay exclusive, routing never falls back to another
shard's admission, and one pending generation on ANY shard blocks the next
policy epoch on ALL of them.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def shard_environment(tmp_path, **extra):
    environment = {
        "RUN_ID": "run-1", "RUN_DIR": str(tmp_path),
        "HARBORRL_NATIVE_CATALOG": str(tmp_path / "catalog.json"),
        "HARBORRL_NATIVE_PROFILE": str(tmp_path / "profile.json"),
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": "0" * 64,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": "1" * 64,
        "HARBORRL_NATIVE_GATEWAY_HOST": "127.0.0.1",
        "HARBORRL_NATIVE_GATEWAY_PORT": "49184",
        "HARBORRL_NATIVE_GATEWAY_URL": "http://127.0.0.1:49184",
        "HARBORRL_NATIVE_RUNNER_PYTHON": "/python",
        "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(["root@worker-a"]),
        "ROLLOUT_MAX_RESPONSE_LEN": "128",
        "ROLLOUT_MAX_CONTEXT_LEN": "1024",
    }
    environment.update(extra)
    return environment


def shard_config(tmp_path, monkeypatch, **extra):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(tmp_path, **extra)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://router:30000", concurrency_limit=2, group_size=2)


def shard_identity(version, slot):
    from harborrl.trajectories.native import Identity

    return Identity(run_id="run-1", batch_id="rollout-0", group_id="group-0",
                    slot_id=slot, attempt_id=f"attempt-{version}-{slot}",
                    trajectory_id=f"trajectory-{version}-{slot}",
                    task_digest="d" * 64, policy_version=version,
                    harness_digest="h" * 64, reward_digest="r" * 64,
                    sampling_digest="s" * 64)


# -- shard table --------------------------------------------------------------


def test_config_parses_the_static_shard_table(tmp_path, monkeypatch):
    from harborrl.rollout.runtime import GatewayShardSpec

    # An absent table keeps the single implicit gateway, now routed "auto":
    # one gateway until the live sandbox count passes sandboxes_per_shard.
    legacy = shard_config(tmp_path, monkeypatch)
    assert legacy.shards == () and legacy.gateway_routing == "auto"

    config = shard_config(tmp_path, monkeypatch, HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps(
        [{"name": "gw0"}, {"name": "gw1", "port": 49300}]))
    assert config.shards == (
        GatewayShardSpec(name="gw0", port=49184, url="http://127.0.0.1:49184"),
        GatewayShardSpec(name="gw1", port=49300, url="http://127.0.0.1:49300"))
    # The rendered environment round-trips the resolved ports.
    assert json.loads(config.environment()["HARBORRL_NATIVE_GATEWAY_SHARDS"]) == [
        {"name": "gw0", "port": 49184}, {"name": "gw1", "port": 49300}]


@pytest.mark.parametrize("shards, message", [
    (json.dumps([{"name": "gw0"}, {"name": "gw0"}]), "unique"),
    (json.dumps([{"name": "gw0", "port": 49184}, {"name": "gw1", "port": 49184}]),
     "unique integers"),
    (json.dumps([{"name": "GW0"}]), "lowercase slugs"),
    (json.dumps([{"name": "gw0", "extra": 1}]), "name, port"),
    (json.dumps([]), "nonempty"),
    ("{not-json", "invalid HARBORRL_NATIVE_GATEWAY_SHARDS"),
])
def test_shard_table_fails_closed(tmp_path, monkeypatch, shards, message):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(tmp_path, HARBORRL_NATIVE_GATEWAY_SHARDS=shards)
    with pytest.raises(ValueError, match=message):
        RolloutRuntimeConfig.from_environment(
            environment, endpoint="http://router:30000", concurrency_limit=2, group_size=2)


@pytest.mark.parametrize("routing, worker_shards, message", [
    ("worker_affinity", {}, "missing"),
    ("worker_affinity", {"root@worker-a": "gwX"}, "unknown gateway shards"),
    ("worker_affinity", {"root@other": "gw0"}, "unknown runners"),
    ("round_robin", {}, "gateway_routing"),
])
def test_worker_affinity_routing_table_fails_closed(
        tmp_path, monkeypatch, routing, worker_shards, message):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(
        tmp_path,
        HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([{"name": "gw0"}, {"name": "gw1"}]),
        HARBORRL_NATIVE_GATEWAY_ROUTING=routing,
        HARBORRL_NATIVE_GATEWAY_WORKER_SHARDS=json.dumps(worker_shards))
    with pytest.raises(ValueError, match=message):
        RolloutRuntimeConfig.from_environment(
            environment, endpoint="http://router:30000", concurrency_limit=2, group_size=2)


# -- runtime construction -----------------------------------------------------


def test_runtime_splits_ports_and_trace_trees_across_shards(tmp_path, monkeypatch):
    from harborrl.gateway import server as gateway_server
    from harborrl.gateway import sglang_backend, trace
    from harborrl.rollout import runtime as rollout_runtime

    (tmp_path / "catalog.json").write_text("[]")
    (tmp_path / "profile.json").write_text(json.dumps({"model": "policy"}))
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [1, 2],
        get_vocab=lambda: {"token": 1},
    )
    tokenizer_digest, template_digest = rollout_runtime.tokenizer_fingerprint(tokenizer)
    environment = shard_environment(
        tmp_path,
        HARBORRL_NATIVE_TOKENIZER_DIGEST=tokenizer_digest,
        HARBORRL_NATIVE_TEMPLATE_DIGEST=template_digest,
        HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([{"name": "gw0"}, {"name": "gw1", "port": 49300}]))
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    registries, servers, gateways = [], [], []

    class FakeGateway:
        def __init__(self, registry, backend, *, model, max_output_tokens):
            gateways.append((registry, backend))

    class FakeServer:
        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    class FakeTraceRegistry:
        version = None
        accepting = True

        def __init__(self, path):
            registries.append(path)

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        def weight_version(self):
            return "7"

    def fake_make_server(gateway, host, port, *, max_inflight=None):
        servers.append((host, port, max_inflight))
        return FakeServer()

    monkeypatch.setattr(gateway_server, "Gateway", FakeGateway)
    monkeypatch.setattr(gateway_server, "make_server", fake_make_server)
    monkeypatch.setattr(sglang_backend, "SGLangBackend", FakeBackend)
    monkeypatch.setattr(trace, "TraceRegistry", FakeTraceRegistry)

    config = rollout_runtime.RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://127.0.0.1:30000", concurrency_limit=2, group_size=2)
    runtime_ = rollout_runtime.RolloutRuntime(config, tokenizer)
    try:
        assert [shard.name for shard in runtime_.shards] == ["gw0", "gw1"]
        # Exclusive ports and per-shard trace subtrees.
        assert servers == [("127.0.0.1", 49184, 4), ("127.0.0.1", 49300, 4)]
        assert registries == [runtime_.run_root / "native-traces" / "shard-gw0",
                              runtime_.run_root / "native-traces" / "shard-gw1"]
        # One shared serving backend; registries stay per shard.
        assert gateways[0][1] is gateways[1][1] is runtime_.backend
        assert gateways[0][0] is not gateways[1][0]
        # The service-facing admission epoch spans every shard.
        assert isinstance(runtime_.registry, rollout_runtime.ShardAdmission)
        assert runtime_.registry.shards == runtime_.shards
        assert runtime_.registry.version is None and runtime_.registry.accepting
        evidence = json.loads((runtime_.run_root / "gateway-shards.json").read_text())
        assert [(s["name"], s["port"], s["url"]) for s in evidence["shards"]] == [
            ("gw0", 49184, "http://127.0.0.1:49184"),
            ("gw1", 49300, "http://127.0.0.1:49300")]
        assert evidence["gateway_routing"] == "modulo"
    finally:
        runtime_.close()


# -- static routing -----------------------------------------------------------


def test_modulo_routing_spreads_attempts_like_worker_rotation():
    from harborrl.rollout.runtime import route_shard

    config = SimpleNamespace(gateway_routing="modulo", worker_shards={})
    shards = [SimpleNamespace(name=f"gw{index}") for index in range(3)]
    picked = [route_shard(config, shards, runner_host=None, slot=slot,
                          attempt_index=attempt).name
              for slot in range(4) for attempt in range(2)]
    assert picked == ["gw0", "gw1", "gw1", "gw2", "gw2", "gw0", "gw0", "gw1"]


def test_worker_affinity_routing_is_static_and_fails_closed():
    from harborrl.rollout.runtime import route_shard

    config = SimpleNamespace(gateway_routing="worker_affinity",
                             worker_shards={"root@worker-a": "gw0", "local": "gw1"})
    gw0, gw1 = SimpleNamespace(name="gw0"), SimpleNamespace(name="gw1")
    shards = (gw0, gw1)
    assert route_shard(config, shards, runner_host="root@worker-a",
                       slot=0, attempt_index=0) is gw0
    # Retries of the same slot keep their worker's shard.
    assert route_shard(config, shards, runner_host="root@worker-a",
                       slot=5, attempt_index=1) is gw0
    # The local runner transport pins to its own shard.
    assert route_shard(config, shards, runner_host=None, slot=1, attempt_index=0) is gw1
    unmapped = SimpleNamespace(gateway_routing="worker_affinity",
                               worker_shards={"local": "gw1"})
    with pytest.raises(ValueError, match="no gateway shard for runner"):
        route_shard(unmapped, shards, runner_host="root@worker-b", slot=0, attempt_index=0)
    unknown = SimpleNamespace(gateway_routing="worker_affinity",
                              worker_shards={"root@worker-a": "gwX"})
    with pytest.raises(ValueError, match="unknown gateway shard"):
        route_shard(unknown, shards, runner_host="root@worker-a", slot=0, attempt_index=0)
    # A single shard needs no routing table.
    assert route_shard(unmapped, shards[:1], runner_host="anywhere",
                       slot=0, attempt_index=0) is gw0


# -- cross-shard version barrier ----------------------------------------------


def test_pending_generation_on_any_shard_blocks_the_whole_epoch(tmp_path):
    from harborrl.gateway.trace import TraceRegistry
    from harborrl.rollout.runtime import ShardAdmission

    registries = [TraceRegistry(tmp_path / f"shard-{index}") for index in range(2)]
    admission = ShardAdmission([SimpleNamespace(registry=r) for r in registries])
    first = registries[0].register(shard_identity("1", "0"), "trial-0")
    second = registries[1].register(shard_identity("1", "1"), "trial-1")
    aid0, request0, *_ = registries[0].begin(first)
    registries[0].finish(aid0, request0, error="interrupted_by_drain")
    aid1, request1, *_ = registries[1].begin(second)
    turn = {"request_id": request1, "response_id": "resp-1",
            "response": {"content": [{"type": "text", "text": "pending"}]}}
    # One in-flight generation on shard 1 keeps admission closed everywhere.
    with pytest.raises(TimeoutError):
        admission.drain(0.05)
    assert not admission.accepting
    assert admission.version == "1"
    # Settle, seal and reopen: the next epoch lands on every shard at once.
    registries[1].finish(aid1, request1, turn)
    registries[1].delivery(aid1, "resp-1", "sent")
    registries[1].consume(aid1, "resp-1", turn["response"]["content"], "session#1")
    for registry, aid in ((registries[0], aid0), (registries[1], aid1)):
        registry.close(aid)
        registry.seal(aid)
    states = [{"endpoint": "http://router:30000", "healthy": True, "weight_version": "2"}]
    admission.drain(1.0)
    admission.publish_version(states, "2")
    assert admission.version == "2" and admission.accepting
    for index, registry in enumerate(registries):
        assert registry.version == "2" and registry.accepting
        # The old epoch never admits new attempts on any shard.
        with pytest.raises(ValueError, match="version mismatch"):
            registry.register(shard_identity("1", f"old-{index}"), f"trial-old-{index}")


def test_service_activation_spans_every_shard(tmp_path):
    from harborrl.gateway.trace import TraceRegistry
    from harborrl.rollout.runtime import ShardAdmission
    from harborrl.rollout.service import RolloutService

    registries = [TraceRegistry(tmp_path / f"shard-{index}") for index in range(2)]
    states = [{"endpoint": "http://router:30000", "healthy": True, "weight_version": "2"}]

    class Pool:
        @staticmethod
        def pool_states():
            return [dict(item) for item in states]

    runtime_ = SimpleNamespace(
        registry=ShardAdmission([SimpleNamespace(registry=r) for r in registries]),
        backend=Pool())
    service = RolloutService(run_root=tmp_path)
    try:
        service.register_run("run-1", {"run_id": "run-1"})
        service.attach_runtime("run-1", SimpleNamespace(drain_timeout=1.0), runtime_)
        result = service.activate_policy("run-1", states, "2")
        assert result["admission"] == "open"
        for registry in registries:
            assert registry.version == "2" and registry.accepting
        assert json.loads(
            (tmp_path / "policy-pool.json").read_text())["weight_version"] == "2"
    finally:
        service.close()


# -- deployment schema --------------------------------------------------------


def test_deployment_accepts_a_static_shard_table(tmp_path):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_shards": [{"name": "gw0"}, {"name": "gw1", "port": 49300}],
        "gateway_routing": "worker_affinity",
        "worker_shards": {"worker-a": "gw0", "worker-b": "gw1"}}))
    manager = config["manager"]
    assert manager["gateway_shards"] == [{"name": "gw0"}, {"name": "gw1", "port": 49300}]
    assert manager["gateway_routing"] == "worker_affinity"
    assert manager["worker_shards"] == {"worker-a": "gw0", "worker-b": "gw1"}
    # Legacy deployments keep their exact historical shape.
    legacy = validate(deployment_mapping(tmp_path))["manager"]
    assert not ({"gateway_shards", "gateway_routing", "worker_shards"} & set(legacy))


@pytest.mark.parametrize("manager_over, message", [
    ({"gateway_shards": []}, "nonempty"),
    ({"gateway_shards": [{"name": "gw0"}, {"name": "gw0"}]}, "duplicate"),
    ({"gateway_shards": [{"name": "gw0", "extra": 1}]}, "name"),
    ({"gateway_shards": [{"name": "GW0"}]}, "lowercase slugs"),
    ({"gateway_shards": [{"name": "gw0", "port": 49200}]}, "control_port"),
    ({"gateway_shards": [{"name": "gw0", "port": 1}, {"name": "gw1", "port": 1}]}, "duplicate"),
    ({"gateway_routing": "round_robin"}, "gateway_routing"),
    ({"worker_shards": {"worker-a": "gw0"}}, "requires manager.gateway_shards"),
    ({"gateway_shards": [{"name": "gw0"}],
      "worker_shards": {"worker-a": "gwX"}}, "unknown shards"),
    ({"gateway_shards": [{"name": "gw0"}],
      "worker_shards": {"worker-c": "gw0"}}, "unknown workers"),
    ({"gateway_shards": [{"name": "gw0"}], "gateway_routing": "worker_affinity"}, "missing"),
    ({"gateway_routing": "worker_affinity"}, "requires manager.gateway_shards"),
])
def test_deployment_shard_routing_fails_closed(tmp_path, manager_over, message):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import DeploymentError, validate

    with pytest.raises(DeploymentError, match=message):
        validate(deployment_mapping(tmp_path, manager=manager_over))
