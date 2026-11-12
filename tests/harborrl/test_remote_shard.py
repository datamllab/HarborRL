"""CPU contract evidence for remote gateway shards (cross-process form).

The admin plane and the manager-side proxy run against real loopback HTTP
servers; no SSH, GPU or SGLang is involved. Contracts under test: the
remote registry surface behaves exactly like the local one, the shared
admission barrier spans local and remote shards, and a dead or unreachable
shard fails closed instead of corrupting the epoch.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest
from test_gateway_shards import shard_environment, shard_identity

STATES = [{"endpoint": "http://router:30000", "healthy": True, "weight_version": "2"}]


@pytest.fixture
def admin(tmp_path):
    """One real admin plane over one real TraceRegistry on loopback."""
    from harborrl.gateway.shard_service import make_admin_server
    from harborrl.gateway.trace import TraceRegistry

    registry = TraceRegistry(tmp_path / "shard-gw1")
    try:
        server = make_admin_server(registry, shard_name="gw1", token="tok",
                                   host="127.0.0.1", port=0)
    except PermissionError:
        pytest.skip("this CPU sandbox does not permit loopback socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", registry
    server.shutdown()
    thread.join(5)
    server.server_close()


# -- manager-side registry surface --------------------------------------------


def test_remote_registry_proxies_the_attempt_surface(tmp_path, admin):
    from harborrl.rollout.remote_shard import RemoteRegistry
    from harborrl.trajectories.native import digest

    url, registry = admin
    remote = RemoteRegistry(url, "tok")
    identity = shard_identity("1", "0")
    token = remote.register(identity, "trial-1", credential="runner-credential")
    assert token == "runner-credential"
    # The shard process drives begin/finish/delivery locally (gateway side).
    aid, request_id, *_ = registry.begin("runner-credential")
    assert aid == identity.attempt_id
    turn = {"request_id": request_id, "response_id": "resp-1",
            "response": {"content": [{"type": "text", "text": "done"}]}}
    registry.finish(aid, request_id, turn)
    registry.delivery(aid, "resp-1", "sent")
    # ... while audits and sealing cross the admin channel.
    missing = remote.unconsumed_responses(aid)
    assert [item["response_id"] for item in missing] == ["resp-1"]
    remote.consume(aid, "resp-1", turn["response"]["content"], "session#1")
    remote.wait_delivered(aid, 1.0)
    remote.close(aid)
    turns, seal = remote.seal(aid)
    assert turns[0]["response_id"] == "resp-1" and seal["status"] == "sealed"
    # The sealed trace lands on disk exactly like a local shard's.
    trace = registry.root / digest(identity.to_dict()) / "trace.json"
    assert json.loads(trace.read_text())["seal"]["status"] == "sealed"
    with pytest.raises(ValueError, match="already registered"):
        remote.register(identity, "trial-1", credential="other")


def test_remote_admission_epoch_rejects_old_versions(admin):
    from harborrl.rollout.remote_shard import RemoteRegistry

    remote = RemoteRegistry(admin[0], "tok")
    remote.close_admission()
    assert remote.version is None and remote.accepting is False
    remote.publish_version(STATES, "2")
    assert remote.version == "2" and remote.accepting is True
    with pytest.raises(ValueError, match="version mismatch"):
        remote.register(shard_identity("1", "old"), "trial-old")


def test_remote_drain_surfaces_the_pending_generation(admin):
    from harborrl.rollout.remote_shard import RemoteRegistry

    url, registry = admin
    remote = RemoteRegistry(url, "tok")
    token = remote.register(shard_identity("1", "0"), "trial-1")
    aid, request_id, *_ = registry.begin(token)
    with pytest.raises(TimeoutError):
        remote.drain(0.05)
    assert remote.accepting is False
    registry.finish(aid, request_id, error="aborted")
    remote.close(aid)
    remote.seal(aid)
    remote.drain(0.5)
    remote.publish_version(STATES, "2")
    assert remote.version == "2"


def test_admin_plane_requires_the_run_token(admin):
    from harborrl.rollout.remote_shard import RemoteRegistry

    remote = RemoteRegistry(admin[0], "wrong-token")
    with pytest.raises(PermissionError):
        remote.unconsumed_responses("attempt")
    correct = RemoteRegistry(admin[0], "tok")
    with pytest.raises(ValueError, match="unknown registry action"):
        correct._call("explode", {"attempt_id": "attempt"})


def test_unreachable_shard_fails_closed(tmp_path):
    from harborrl.rollout.remote_shard import RemoteRegistry
    from harborrl.rollout.runtime import ShardAdmission

    dead = RemoteRegistry("http://127.0.0.1:1", "tok", status_timeout=0.2, timeout=0.2)
    assert dead.version is None and dead.accepting is False
    with pytest.raises(OSError):
        dead.register(shard_identity("1", "0"), "trial")
    admission = ShardAdmission([SimpleNamespace(registry=dead)])
    assert admission.version is None and admission.accepting is False
    # A dead shard blocks the shared epoch instead of half-publishing it.
    with pytest.raises(OSError):
        admission.drain(0.1)


def test_shared_epoch_spans_local_and_remote_shards(tmp_path, admin):
    from harborrl.gateway.trace import TraceRegistry
    from harborrl.rollout.remote_shard import RemoteRegistry
    from harborrl.rollout.runtime import ShardAdmission

    url, remote_registry = admin
    remote = RemoteRegistry(url, "tok")
    local = TraceRegistry(tmp_path / "shard-gw0")
    admission = ShardAdmission([
        SimpleNamespace(registry=local), SimpleNamespace(registry=remote)])
    token = remote.register(shard_identity("1", "0"), "trial-1")
    aid, request_id, *_ = remote_registry.begin(token)
    with pytest.raises(TimeoutError):
        admission.drain(0.05)
    remote_registry.finish(aid, request_id, error="aborted")
    remote.close(aid)
    remote.seal(aid)
    admission.drain(0.5)
    admission.publish_version(STATES, "2")
    assert local.version == "2" and remote.version == "2"
    with pytest.raises(ValueError, match="version mismatch"):
        local.register(shard_identity("1", "x"), "trial-old")


# -- shard process lifecycle ---------------------------------------------------


def test_remote_shard_probes_identity_and_shuts_down(tmp_path):
    from harborrl.gateway.shard_service import make_admin_server
    from harborrl.gateway.trace import TraceRegistry
    from harborrl.rollout.remote_shard import RemoteShard
    from harborrl.rollout.runtime import GatewayShardSpec

    registry = TraceRegistry(tmp_path / "shard-gw1")
    stopped = threading.Event()
    try:
        server = make_admin_server(registry, shard_name="gw1", token="tok",
                                   host="127.0.0.1", port=0, on_shutdown=stopped.set)
    except PermissionError:
        pytest.skip("this CPU sandbox does not permit loopback socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    admin_url = f"http://127.0.0.1:{server.server_address[1]}"
    spec = GatewayShardSpec("gw1", 49300, "http://10.98.250.180:49300",
                            mode="remote", admin_url=admin_url)
    shard = RemoteShard(spec, token="tok", trace_dir=tmp_path)
    try:
        shard.start()  # Answers, authenticated, under its declared name.
        imposter = GatewayShardSpec("gwX", 49300, "http://10.98.250.180:49300",
                                    mode="remote", admin_url=admin_url)
        with pytest.raises(RuntimeError, match="answers as"):
            RemoteShard(imposter, token="tok", trace_dir=tmp_path).start()
    finally:
        shard.close()
        assert stopped.wait(2)
        server.shutdown()
        thread.join(5)
        server.server_close()


def test_shard_service_config_parses_its_environment(tmp_path):
    from harborrl.gateway.shard_service import ShardServiceConfig

    env = {
        "RUN_ID": "run-1", "RUN_DIR": str(tmp_path),
        "HARBORRL_SHARD_NAME": "gw1",
        "HARBORRL_SHARD_GATEWAY_PORT": "49300",
        "HARBORRL_SHARD_GATEWAY_URL": "http://10.98.250.180:49300",
        "HARBORRL_SHARD_ADMIN_PORT": "49301",
        "HARBORRL_SHARD_TOKEN_FILE": str(tmp_path / "run-token"),
        "HARBORRL_SHARD_ENDPOINT": "http://router:30000/",
        "HARBORRL_NATIVE_PROFILE": str(tmp_path / "profile.json"),
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": "0" * 64,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": "1" * 64,
    }
    config = ShardServiceConfig.from_environment(env)
    assert config.endpoint == "http://router:30000"
    # The trace subtree spelling matches the manager-side derivation.
    assert config.trace_dir == tmp_path / "native-traces" / "shard-gw1"
    missing = dict(env)
    del missing["HARBORRL_SHARD_TOKEN_FILE"]
    with pytest.raises(ValueError, match="HARBORRL_SHARD_TOKEN_FILE"):
        ShardServiceConfig.from_environment(missing)


# -- shard table parsing --------------------------------------------------------


def test_config_parses_mixed_local_and_remote_shards(tmp_path, monkeypatch):
    from harborrl.rollout.runtime import GatewayShardSpec, RolloutRuntimeConfig

    token = tmp_path / "run-token"
    token.write_text("run-secret\n")
    environment = shard_environment(
        tmp_path,
        HARBORRL_NATIVE_SHARD_TOKEN_FILE=str(token),
        HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([
            {"name": "gw0"},
            {"name": "gw1", "mode": "remote", "url": "http://10.98.250.180:49300",
             "admin_url": "http://10.98.250.180:49301"}]))
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    config = RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://router:30000", concurrency_limit=2, group_size=2)
    assert config.shards[0].mode == "local"
    assert config.shards[1] == GatewayShardSpec(
        name="gw1", port=49300, url="http://10.98.250.180:49300",
        mode="remote", admin_url="http://10.98.250.180:49301")
    # The rendered environment round-trips remote placement losslessly.
    rendered = json.loads(config.environment()["HARBORRL_NATIVE_GATEWAY_SHARDS"])
    assert rendered[1] == {"name": "gw1", "mode": "remote",
                           "url": "http://10.98.250.180:49300",
                           "admin_url": "http://10.98.250.180:49301"}


@pytest.mark.parametrize("entry, message", [
    ({"name": "gw1", "mode": "remote", "admin_url": "http://dev:49301"}, "url"),
    ({"name": "gw1", "mode": "remote", "url": "http://dev:49300"}, "admin_url"),
    ({"name": "gw1", "mode": "remote", "url": "http://dev:49300",
      "admin_url": "http://dev:49301", "port": 7}, "takes its port from url"),
    ({"name": "gw0", "url": "http://dev:49300"}, "binds a port"),
    ({"name": "gw1", "mode": "flying"}, "mode"),
])
def test_remote_shard_entries_fail_closed(tmp_path, monkeypatch, entry, message):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(
        tmp_path, HARBORRL_NATIVE_SHARD_TOKEN_FILE=str(tmp_path / "token"),
        HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([entry]))
    with pytest.raises(ValueError, match=message):
        RolloutRuntimeConfig.from_environment(
            environment, endpoint="http://router:30000", concurrency_limit=2, group_size=2)


def test_remote_shards_require_the_shared_token_file(tmp_path, monkeypatch):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(tmp_path, HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([
        {"name": "gw1", "mode": "remote", "url": "http://dev:49300",
         "admin_url": "http://dev:49301"}]))
    with pytest.raises(ValueError, match="SHARD_TOKEN_FILE"):
        RolloutRuntimeConfig.from_environment(
            environment, endpoint="http://router:30000", concurrency_limit=2, group_size=2)


def test_runtime_mixes_local_and_remote_shards(tmp_path, monkeypatch):
    from harborrl.gateway import server as gateway_server
    from harborrl.gateway import sglang_backend, trace
    from harborrl.rollout import remote_shard as remote_module
    from harborrl.rollout import runtime as rollout_runtime

    (tmp_path / "catalog.json").write_text("[]")
    (tmp_path / "profile.json").write_text(json.dumps({"model": "policy"}))
    (tmp_path / "run-token").write_text("run-secret\n")
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [1, 2],
        get_vocab=lambda: {"token": 1},
    )
    tokenizer_digest, template_digest = rollout_runtime.tokenizer_fingerprint(tokenizer)
    environment = shard_environment(
        tmp_path,
        HARBORRL_NATIVE_TOKENIZER_DIGEST=tokenizer_digest,
        HARBORRL_NATIVE_TEMPLATE_DIGEST=template_digest,
        HARBORRL_NATIVE_SHARD_TOKEN_FILE=str(tmp_path / "run-token"),
        HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([
            {"name": "gw0"},
            {"name": "gw1", "mode": "remote", "url": "http://10.98.250.180:49300",
             "admin_url": "http://10.98.250.180:49301"}]))
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    registries, made_servers = [], []

    class FakeGateway:
        def __init__(self, registry, backend, *, model, max_output_tokens):
            pass

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

    class FakeRemoteShard:
        placed: ClassVar[list] = []

        def __init__(self, spec, *, token, trace_dir):
            assert token == "run-secret"
            self.spec = spec
            self.trace_dir = trace_dir
            self.registry = SimpleNamespace(version=None, accepting=True)
            FakeRemoteShard.placed.append(spec)

        @property
        def name(self):
            return self.spec.name

        @property
        def url(self):
            return self.spec.url

        @property
        def port(self):
            return self.spec.port

        def start(self):
            pass

        def close(self):
            pass

    def fake_make_server(gateway, host, port, *, max_inflight=None):
        made_servers.append(port)
        return FakeServer()

    monkeypatch.setattr(gateway_server, "Gateway", FakeGateway)
    monkeypatch.setattr(gateway_server, "make_server", fake_make_server)
    monkeypatch.setattr(sglang_backend, "SGLangBackend", FakeBackend)
    monkeypatch.setattr(trace, "TraceRegistry", FakeTraceRegistry)
    monkeypatch.setattr(remote_module, "RemoteShard", FakeRemoteShard)

    config = rollout_runtime.RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://127.0.0.1:30000", concurrency_limit=2, group_size=2)
    runtime_ = rollout_runtime.RolloutRuntime(config, tokenizer)
    try:
        # Exactly one local bind; the remote shard contributes no server.
        assert made_servers == [49184]
        assert len(registries) == 1
        assert [spec.name for spec in FakeRemoteShard.placed] == ["gw1"]
        assert FakeRemoteShard.placed[0].admin_url == "http://10.98.250.180:49301"
        evidence = json.loads((runtime_.run_root / "gateway-shards.json").read_text())
        assert [(item["name"], item["mode"]) for item in evidence["shards"]] == [
            ("gw0", "local"), ("gw1", "remote")]
        assert evidence["shards"][1]["admin_url"] == "http://10.98.250.180:49301"
    finally:
        runtime_.close()


# -- deployment schema and controller ------------------------------------------


def test_deployment_accepts_remote_shards(tmp_path):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_shards": [{"name": "gw0"}, {
            "name": "gw1", "mode": "remote", "url": "http://dev:49300",
            "admin_url": "http://dev:49301"}]}))
    shards = config["manager"]["gateway_shards"]
    assert shards[1] == {"name": "gw1", "mode": "remote", "url": "http://dev:49300",
                         "admin_url": "http://dev:49301"}


@pytest.mark.parametrize("manager_over, message", [
    ({"gateway_shards": [{"name": "gw1", "mode": "remote", "url": "http://dev:49300"}]},
     "admin_url"),
    ({"gateway_shards": [{"name": "gw1", "mode": "remote", "url": "http://dev:49300",
                          "admin_url": "http://dev:49301", "port": 49300}]},
     "takes its port from url"),
    ({"gateway_shards": [{"name": "gw0", "url": "http://dev:49300"}]}, "binds a port"),
    ({"gateway_shards": [{"name": "gw1", "mode": "remote", "url": "http://dev:49300",
                          "admin_url": "http://dev:49301"}], "mode": "embedded"},
     "service-mode manager"),
    ({"gateway_shards": [{"name": "gw1", "mode": "flying"}]}, "mode"),
])
def test_remote_shard_deployments_fail_closed(tmp_path, manager_over, message):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import DeploymentError, validate

    if manager_over.get("mode") == "embedded":
        manager_over = {key: value for key, value in manager_over.items() if key != "mode"}
        values = deployment_mapping(tmp_path, manager=manager_over)
        values["manager"]["mode"] = "embedded"
    else:
        values = deployment_mapping(tmp_path, manager=manager_over)
    with pytest.raises(DeploymentError, match=message):
        validate(values)


def test_controller_leaves_remote_shard_origins_alone():
    from test_run_port_allocation import deployment as port_deployment

    from harborrl.deployment import controller as dispatch

    sharded = port_deployment(gateway_shards=[
        {"name": "gw0"},
        {"name": "gw1", "mode": "remote", "url": "http://10.98.250.180:49300",
         "admin_url": "http://10.98.250.180:49301"}])
    patched = dispatch._with_allocated_ports(sharded, "run-r")
    manager = patched["manager"]
    assert manager["gateway_port"] == 53910
    assert manager["gateway_url"] == "http://127.0.0.1:53910"
    assert manager["gateway_shards"][0]["port"] == 53910
    remote = manager["gateway_shards"][1]
    assert remote == {"name": "gw1", "mode": "remote",
                      "url": "http://10.98.250.180:49300",
                      "admin_url": "http://10.98.250.180:49301"}
    dispatch._release_run_ports("run-r")


def test_rollout_environment_carries_remote_placement(tmp_path):
    from test_run_port_allocation import deployment as port_deployment

    from harborrl.deployment import controller as dispatch

    token = tmp_path / "run-token"
    token.write_text("run-secret\n")
    sharded = port_deployment(
        run_token_file=str(token),
        gateway_shards=[{"name": "gw0", "port": 53910}, {
            "name": "gw1", "mode": "remote", "url": "http://10.98.250.180:49300",
            "admin_url": "http://10.98.250.180:49301"}])
    sharded["gpu_target"] = {"train_python": "/train/python"}
    sharded["rollout"] = {"workers": [
        {"id": "worker-a", "ssh": {"host": "worker-a"},
         "runner_python": "/runner/python"}], "provider": "ssh_docker"}
    overlay = dispatch._rollout_environment({"config": {"harbor": {}}}, sharded, {})
    assert json.loads(overlay["HARBORRL_NATIVE_GATEWAY_SHARDS"]) == [
        {"name": "gw0", "port": 53910},
        {"name": "gw1", "mode": "remote", "url": "http://10.98.250.180:49300",
         "admin_url": "http://10.98.250.180:49301"}]
    # The remote process reads the same shared-storage token file.
    assert overlay["HARBORRL_NATIVE_SHARD_TOKEN_FILE"] == str(token)


def test_gateway_shard_preflight(tmp_path, admin):
    from harborrl.deployment import controller as dispatch

    admin_url = admin[0]
    token_file = tmp_path / "run-token"
    token_file.write_text("tok\n")

    def manager_with(name):
        return {"run_token_file": str(token_file), "gateway_shards": [{
            "name": name, "mode": "remote", "url": "http://10.98.250.180:49300",
            "admin_url": admin_url}]}

    checks = dispatch.check_gateway_shards({"manager": manager_with("gw1")})
    assert checks == [{"name": "gw1", "admin_url": admin_url,
                       "gateway_url": "http://10.98.250.180:49300", "ok": True}]
    # A shard answering under a different name fails dispatch loudly.
    with pytest.raises(RuntimeError, match="answers as"):
        dispatch.check_gateway_shards({"manager": manager_with("gwX")})
    # Local-only deployments have nothing to preflight.
    assert dispatch.check_gateway_shards({"manager": {"gateway_shards": [
        {"name": "gw0", "port": 49184}]}}) is None


# -- shard service process contract (ephemeral gateway bind) ---------------------


def service_environment(**extra):
    """A complete ShardServiceConfig environment; overrides win."""
    environment = {
        "RUN_ID": "run-1", "RUN_DIR": "/run", "HARBORRL_SHARD_NAME": "gw1",
        "HARBORRL_SHARD_GATEWAY_HOST": "0.0.0.0",
        "HARBORRL_SHARD_GATEWAY_PORT": "49300",
        "HARBORRL_SHARD_GATEWAY_URL": "http://gpu-a:49300",
        "HARBORRL_SHARD_ADMIN_HOST": "0.0.0.0",
        "HARBORRL_SHARD_ADMIN_PORT": "49301",
        "HARBORRL_SHARD_TOKEN_FILE": "/run/run-token",
        "HARBORRL_SHARD_ENDPOINT": "http://router:30000",
        "HARBORRL_NATIVE_PROFILE": "/run/profile.json",
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": "0" * 64,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": "1" * 64,
        "HF_CKPT": "/shared/ckpt",
    }
    environment.update(extra)
    return environment


def test_ephemeral_gateway_bind_needs_no_declared_url():
    """Port 0 is how an auto-growing manager launches this service.

    The URL cannot exist before the bind, so the ready line carries the
    resolved port instead and the launcher composes the URL.
    """
    from harborrl.gateway.shard_service import ShardServiceConfig

    config = ShardServiceConfig.from_environment(
        service_environment(HARBORRL_SHARD_GATEWAY_PORT="0"))
    assert config.gateway_port == 0 and config.gateway_url == ""


def test_fixed_gateway_port_still_requires_the_declared_url():
    from harborrl.gateway.shard_service import ShardServiceConfig

    environment = service_environment()
    del environment["HARBORRL_SHARD_GATEWAY_URL"]
    with pytest.raises(ValueError, match="HARBORRL_SHARD_GATEWAY_URL"):
        ShardServiceConfig.from_environment(environment)


def test_ready_record_reports_the_resolved_ports():
    from harborrl.gateway.shard_service import ShardServiceConfig, _ready_record

    ephemeral = ShardServiceConfig.from_environment(
        service_environment(HARBORRL_SHARD_GATEWAY_PORT="0"))
    record = _ready_record(ephemeral, gateway_port=40585, admin_port=49301)
    assert record["gateway_port"] == 40585 and record["admin_port"] == 49301
    assert record["shard"] == "gw1" and record["service"]
    # No declared URL: the placeholder names the resolved port; the launcher
    # reads gateway_port and composes the runner-facing URL itself.
    assert record["gateway"] == "http://localhost:40585"

    fixed = ShardServiceConfig.from_environment(service_environment())
    fixed_record = _ready_record(fixed, gateway_port=49300, admin_port=49301)
    # A fixed port keeps one spelling everywhere.
    assert fixed_record["gateway"] == "http://gpu-a:49300"
    assert fixed_record["gateway_port"] == 49300
