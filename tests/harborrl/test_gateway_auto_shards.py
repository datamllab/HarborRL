"""CPU contract evidence for elastic auto-sharded gateways.

Default is exactly one gateway. Auto routing leases one gateway per RUNNING
sandbox attempt, keeps at most ``sandboxes_per_shard`` concurrent attempts
per gateway (default 30, configurable), and grows the fleet when capacity is
exhausted — on a declared remote host first (a shard process launched over
ssh), locally only when no candidate answers. Either way the new shard joins
the live admission epoch atomically, so group admission never observes shard
disagreement. Group settlement retires idle spawned gateways symmetrically
(tail-only, never the initial gateway, names never reused). Everything fails
closed: bad capacity values, a static table combined with auto routing, and
leaked leases are all loud.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
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


def build_auto_runtime(tmp_path, monkeypatch, *, sandboxes_per_shard, concurrency=2,
                       gateway_hosts=(), shard_python=None, shards_json=None):
    """One real auto-routing runtime over fake server/registry/backend parts."""
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
    extra = {
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": tokenizer_digest,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": template_digest,
        "HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD": str(sandboxes_per_shard),
    }
    if shards_json:
        # A declared table coexisting with the growth pool: remote entries are
        # reached through the patched RemoteShard, never a real connection.
        # A table alone pins routing to modulo, so the coexistence case must
        # declare auto explicitly (the schema default does this for a pool).
        extra["HARBORRL_NATIVE_GATEWAY_SHARDS"] = shards_json
        if gateway_hosts:
            extra["HARBORRL_NATIVE_GATEWAY_ROUTING"] = "auto"
    if gateway_hosts:
        # Remote growth needs the shared run token and a tokenizer checkpoint
        # the spawned process loads itself.
        (tmp_path / "run-token").write_text("shared-run-token")
        extra.update(
            HARBORRL_NATIVE_GATEWAY_HOSTS=json.dumps(list(gateway_hosts)),
            HARBORRL_NATIVE_SHARD_TOKEN_FILE=str(tmp_path / "run-token"),
            HF_CKPT="/shared/ckpt",
        )
        if shard_python:
            extra["HARBORRL_NATIVE_SHARD_PYTHON"] = shard_python
    environment = shard_environment(tmp_path, **extra)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    state = SimpleNamespace(servers=[], registries=[], closed=[])

    class FakeGateway:
        def __init__(self, registry, backend, *, model, max_output_tokens):
            pass

    class FakeServer:
        sequence = 0

        def __init__(self):
            FakeServer.sequence += 1
            self.server_address = ("127.0.0.1", 40000 + FakeServer.sequence)

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            state.closed.append(self.server_address[1])

    class FakeTraceRegistry:
        def __init__(self, path):
            self.path = str(path)
            self.version = None
            self.accepting = True
            state.registries.append(self)

        def publish_version(self, states, expected):
            self.version, self.accepting = str(expected), True

        def close_admission(self):
            self.accepting = False

        def drain(self, timeout):
            return None

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

    def fake_make_server(gateway, host, port, *, max_inflight=None):
        server = FakeServer()
        state.servers.append((host, port, server.server_address[1], max_inflight))
        return server

    monkeypatch.setattr(gateway_server, "Gateway", FakeGateway)
    monkeypatch.setattr(gateway_server, "make_server", fake_make_server)
    monkeypatch.setattr(sglang_backend, "SGLangBackend", FakeBackend)
    monkeypatch.setattr(trace, "TraceRegistry", FakeTraceRegistry)
    if shards_json:
        # A declared remote shard would otherwise construct the real
        # RemoteShard and dial its admin origin. The static path imports it
        # lazily inside __init__, so the seam is the source module's
        # attribute, not the runtime module's re-export.
        from harborrl.rollout import remote_shard as remote_module

        monkeypatch.setattr(remote_module, "RemoteShard", FakeRemoteShard)

    config = rollout_runtime.RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://127.0.0.1:30000",
        concurrency_limit=concurrency, group_size=2)
    runtime_ = rollout_runtime.RolloutRuntime(config, tokenizer)
    return runtime_, state


def group_entry(tmp_path):
    return {"id": "task-0", "revision": 1, "path": str(tmp_path / "task.md"),
            "task_digest": "d" * 64, "reward_profile": {}}


class PermissiveSlots:
    """Slot bookkeeping that accepts fake IRs.

    The real GroupSlots runs ``require_ready`` on every result and no fake
    attempt can produce a valid trajectory.v2 record; the lease behavior
    under test only needs start/finish bookkeeping.
    """

    max_attempts = 2

    def __init__(self, identities, *, max_attempts=2):
        self.identities = {identity.slot_id: identity for identity in identities}

    def start(self, slot):
        return self.identities[slot]

    def finish(self, slot, ir=None, error=None):
        assert (ir is None) != (error is None)


def lease(runtime_):
    """lease_shard is async: remote growth waits on an ssh subprocess."""
    return asyncio.run(runtime_.lease_shard())


class FakeRemoteShard:
    """A SpawnedRemoteShard stand-in carrying a REAL TraceRegistry.

    The join replay must cross publish_version's closed-barrier precondition
    — fake registries cannot see it (see the regression test below) — so the
    remote-spawn tests keep the real registry behind the fake transport.
    """

    def __init__(self, spec, *, token, trace_dir, process=None, timeout=30.0):
        from harborrl.gateway.trace import TraceRegistry

        self.spec = spec
        self.trace_dir = Path(trace_dir)
        self.registry = TraceRegistry(trace_dir)
        self.process = process
        self.started = False
        self.closed = False

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
        self.started = True

    def close(self):
        self.closed = True


def fake_remote_launch(monkeypatch, runtime_module, *, delay=0.0, failures=0):
    """Patch the remote transport; record every launch attempt.

    The first ``failures`` launches raise, exercising the candidate loop and
    the local fallback; every launch resolves distinct ports like a real
    ready line would.
    """
    from harborrl.gateway.shard_service import SHARD_SERVICE

    launched = []

    class FakeProcess:
        returncode = 0

    async def start_process(host, environment, *, port=None, identity=None):
        entry = {"host": host, "name": environment["HARBORRL_SHARD_NAME"]}
        if port is not None:
            entry["port"] = port
        if identity is not None:
            entry["identity"] = identity
        launched.append(entry)
        if len(launched) <= failures:
            raise OSError(f"connection timed out: {host}")
        if delay:
            await asyncio.sleep(delay)
        port = 41000 + len(launched)
        return (FakeProcess(), {"shard": environment["HARBORRL_SHARD_NAME"],
                                "service": SHARD_SERVICE,
                                "gateway_port": port, "admin_port": port + 1})

    monkeypatch.setattr(runtime_module, "SpawnedRemoteShard", FakeRemoteShard)
    return start_process, launched


# -- config -------------------------------------------------------------------


def test_absent_table_defaults_to_one_auto_gateway_of_thirty(tmp_path, monkeypatch):
    config = shard_config(tmp_path, monkeypatch)
    assert config.shards == ()
    assert config.gateway_routing == "auto"
    assert config.sandboxes_per_shard == 30
    rendered = config.environment()
    assert rendered["HARBORRL_NATIVE_GATEWAY_ROUTING"] == "auto"
    assert rendered["HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD"] == "30"


def test_sandboxes_per_shard_is_configurable_and_round_trips(tmp_path, monkeypatch):
    config = shard_config(
        tmp_path, monkeypatch, HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD="7")
    assert config.sandboxes_per_shard == 7
    assert config.environment()["HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD"] == "7"


@pytest.mark.parametrize("value", ["0", "-3", "many"])
def test_sandboxes_per_shard_fails_closed(tmp_path, monkeypatch, value):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(
        tmp_path, HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD=value)
    with pytest.raises(ValueError, match="(?i)sandboxes_per_shard"):
        RolloutRuntimeConfig.from_environment(
            environment, endpoint="http://router:30000",
            concurrency_limit=2, group_size=2)


def test_auto_routing_accepts_a_static_table_with_a_growth_pool(tmp_path, monkeypatch):
    """A long-lived declared shard is the floor growth expands beyond.

    The tunnel-host topology: one statically declared remote shard exists at
    startup, and auto routing still grows extra gateways on the pool when
    concurrency passes the per-shard budget.
    """
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(
        tmp_path,
        HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([
            {"name": "static-1", "mode": "remote",
             "url": "http://tunnel-host:5010", "admin_url": "http://tunnel-host:5011"}]),
        HARBORRL_NATIVE_GATEWAY_HOSTS=json.dumps(["gpu-a"]),
        HARBORRL_NATIVE_SHARD_TOKEN_FILE=str(tmp_path / "t"),
        HF_CKPT="/ckpt",
        HARBORRL_NATIVE_GATEWAY_ROUTING="auto")
    (tmp_path / "t").write_text("token")
    config = RolloutRuntimeConfig.from_environment(
        environment, endpoint="http://router:30000",
        concurrency_limit=2, group_size=2)
    assert config.gateway_routing == "auto"
    assert [spec.name for spec in config.shards] == ["static-1"]
    assert config.gateway_hosts == ("gpu-a",)


def test_growth_names_skip_names_a_static_table_claims(tmp_path, monkeypatch):
    """A regrown shard must never reuse a declared static name."""
    from harborrl.rollout import runtime as rollout_runtime

    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, gateway_hosts=("gpu-a",),
        shards_json=json.dumps([{"name": "gw1", "mode": "remote",
                                 "url": "http://declared:5010",
                                 "admin_url": "http://declared:5011"}]))
    start_process, launched = fake_remote_launch(monkeypatch, rollout_runtime)
    monkeypatch.setattr(runtime_, "_start_shard_process", start_process)
    try:
        lease(runtime_)  # fills the declared gw1
        states = [{"endpoint": "http://router:30000", "healthy": True,
                   "weight_version": "2"}]
        runtime_.registry.publish_version(states, "2")
        grown = lease(runtime_)
        # gw1 is taken by the table, so growth starts at gw2.
        assert grown.name == "gw2"
        assert launched == [{"host": "gpu-a", "name": "gw2"}]
    finally:
        runtime_.close()


def test_explicit_modulo_without_a_table_opts_out_of_auto(tmp_path, monkeypatch):
    config = shard_config(
        tmp_path, monkeypatch, HARBORRL_NATIVE_GATEWAY_ROUTING="modulo")
    assert config.gateway_routing == "modulo"


def test_declared_table_keeps_the_static_modulo_default(tmp_path, monkeypatch):
    config = shard_config(
        tmp_path, monkeypatch,
        HARBORRL_NATIVE_GATEWAY_SHARDS=json.dumps([{"name": "gw0"}, {"name": "gw1"}]))
    assert config.gateway_routing == "modulo"


def test_gateway_hosts_round_trip_through_the_environment(tmp_path, monkeypatch):
    config = shard_config(
        tmp_path, monkeypatch,
        HARBORRL_NATIVE_GATEWAY_HOSTS=json.dumps(["root@gpu-a", "gpu-b"]),
        HARBORRL_NATIVE_SHARD_TOKEN_FILE=str(tmp_path / "run-token"),
        HF_CKPT="/shared/ckpt")
    assert config.gateway_hosts == ("root@gpu-a", "gpu-b")
    rendered = config.environment()
    assert rendered["HARBORRL_NATIVE_GATEWAY_HOSTS"] == '["root@gpu-a", "gpu-b"]'
    assert rendered["HF_CKPT"] == "/shared/ckpt"
    assert "HARBORRL_NATIVE_SHARD_PYTHON" not in rendered


def test_shard_python_overrides_the_growth_interpreter(tmp_path, monkeypatch):
    """Growth hosts run the serving stack, not the runner's sandbox framework."""
    runtime_, _ = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1,
        gateway_hosts=("gpu-a",), shard_python="/serving/python")
    try:
        assert runtime_._shard_launch_python() == "/serving/python"
        assert (runtime_.config.environment()["HARBORRL_NATIVE_SHARD_PYTHON"]
                == "/serving/python")
    finally:
        runtime_.close()

    runtime_, _ = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, gateway_hosts=("gpu-a",))
    try:
        # Undeclared keeps the runner's interpreter (uniform deployments).
        assert runtime_._shard_launch_python() == runtime_.config.runner_python
        assert "HARBORRL_NATIVE_SHARD_PYTHON" not in runtime_.config.environment()
    finally:
        runtime_.close()


@pytest.mark.parametrize("overrides, message", [
    # Remote growth shares the run token and needs a tokenizer checkpoint the
    # spawned process loads itself.
    ({"HARBORRL_NATIVE_GATEWAY_HOSTS": json.dumps(["gpu-a"]),
      "HF_CKPT": "/ckpt"}, "SHARD_TOKEN_FILE"),
    ({"HARBORRL_NATIVE_GATEWAY_HOSTS": json.dumps(["gpu-a"]),
      "HARBORRL_NATIVE_SHARD_TOKEN_FILE": "t"}, "HF_CKPT"),
    ({"HARBORRL_NATIVE_GATEWAY_HOSTS": json.dumps(["gpu-a"]),
      "HARBORRL_NATIVE_SHARD_TOKEN_FILE": "t", "HF_CKPT": "/ckpt",
      "HARBORRL_NATIVE_GATEWAY_ROUTING": "modulo"}, "auto routing only"),
])
def test_gateway_hosts_fail_closed(tmp_path, monkeypatch, overrides, message):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(tmp_path, **overrides)
    with pytest.raises(ValueError, match=message):
        RolloutRuntimeConfig.from_environment(
            environment, endpoint="http://router:30000",
            concurrency_limit=2, group_size=2)


@pytest.mark.parametrize("value", ["gpu-a", '["", "gpu-a"]', '{"host": "a"}', "[[]]"])
def test_gateway_hosts_env_garbage_fails_closed(tmp_path, monkeypatch, value):
    from harborrl.rollout.runtime import RolloutRuntimeConfig

    environment = shard_environment(
        tmp_path, HARBORRL_NATIVE_GATEWAY_HOSTS=value,
        HARBORRL_NATIVE_SHARD_TOKEN_FILE="t", HF_CKPT="/ckpt")
    with pytest.raises(ValueError, match="(?i)gateway_hosts|ssh destinations"):
        RolloutRuntimeConfig.from_environment(
            environment, endpoint="http://router:30000",
            concurrency_limit=2, group_size=2)


# -- lease, spawn, join -------------------------------------------------------


def test_leases_fill_one_gateway_then_spawn_the_next(tmp_path, monkeypatch):
    runtime_, state = build_auto_runtime(tmp_path, monkeypatch, sandboxes_per_shard=2)
    try:
        first = lease(runtime_)
        second = lease(runtime_)
        assert first is second is runtime_.shards[0]
        assert first.name == "gateway"
        # Capacity n reached: the (n+1)-th concurrent sandbox spawns gw1 on an
        # ephemeral port with its own trace subtree.
        third = lease(runtime_)
        assert third.name == "gw1"
        assert third.port != first.port and third.url != first.url
        assert third.url == f"http://127.0.0.1:{third.port}"
        assert third.trace_dir == runtime_.run_root / "native-traces" / "shard-gw1"
        assert state.servers[1][1] == 0  # the OS owns spawned-shard exclusivity
        # The evidence record grows with the live shard set.
        evidence = json.loads((runtime_.run_root / "gateway-shards.json").read_text())
        assert evidence["gateway_routing"] == "auto"
        assert evidence["sandboxes_per_shard"] == 2
        assert [(s["name"], s["port"]) for s in evidence["shards"]] == [
            ("gateway", 49184), ("gw1", third.port)]
        # A released lease reuses freed capacity instead of spawning again.
        runtime_.release_shard(third)
        again = lease(runtime_)
        assert again is third
        assert len(state.servers) == 2
    finally:
        runtime_.close()
    assert len(state.closed) == 2


def test_spawned_shard_joins_the_live_policy_epoch(tmp_path, monkeypatch):
    runtime_, state = build_auto_runtime(tmp_path, monkeypatch, sandboxes_per_shard=1)
    try:
        lease(runtime_)
        states = [{"endpoint": "http://router:30000", "healthy": True,
                   "weight_version": "2"}]
        runtime_.registry.publish_version(states, "2")
        # The (n+1)-th lease spawns gw1 straight onto the live epoch: joining
        # with version None would read as shard disagreement and fail-close
        # group admission until the next activation.
        spawned = lease(runtime_)
        assert spawned.name == "gw1"
        assert spawned.registry.version == "2" and spawned.registry.accepting
        assert runtime_.registry.version == "2" and runtime_.registry.accepting
        # The epoch barrier spans the late shard too.
        runtime_.registry.drain(0.05)
        assert all(registry.accepting is False for registry in state.registries)
    finally:
        runtime_.close()


def test_spawned_shard_before_the_first_epoch_stays_unversioned(tmp_path, monkeypatch):
    runtime_, _ = build_auto_runtime(tmp_path, monkeypatch, sandboxes_per_shard=1)
    try:
        lease(runtime_)
        spawned = lease(runtime_)
        assert spawned.registry.version is None
        assert runtime_.registry.version is None and runtime_.registry.accepting
    finally:
        runtime_.close()


def test_join_replays_a_published_epoch_onto_a_real_registry(tmp_path):
    """Regression: the replay must satisfy publish_version's barrier.

    A fresh TraceRegistry opens accepting and publish_version refuses to
    open a barrier that was never closed; replaying an epoch without closing
    first raised "all attempts must be sealed behind a closed admission
    barrier" and every auto-grown shard failed to spawn mid-run (the fake
    registries used elsewhere cannot see this precondition).
    """
    from harborrl.gateway.trace import TraceRegistry
    from harborrl.rollout.runtime import ShardAdmission

    shards = []
    admission = ShardAdmission(shards)
    shards.append(SimpleNamespace(registry=TraceRegistry(tmp_path / "gw0")))
    states = [{"endpoint": "http://router:30000", "healthy": True,
               "weight_version": "2"}]
    admission.drain(0.05)
    admission.publish_version(states, "2")

    grown = SimpleNamespace(registry=TraceRegistry(tmp_path / "gw1"))
    admission.join(grown)

    # The late shard lands on the live epoch, open for admission.
    assert grown.registry.version == "2" and grown.registry.accepting
    assert admission.version == "2" and admission.accepting
    assert len(admission.shards) == 2
    # The epoch barrier spans the late shard too.
    admission.drain(0.05)
    assert not grown.registry.accepting


# -- remote-first growth ------------------------------------------------------


def test_growth_prefers_declared_remote_hosts(tmp_path, monkeypatch):
    """Growth fills the declared pool before ever binding a local port."""
    from harborrl.rollout import runtime as rollout_runtime

    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, gateway_hosts=("root@gpu-a",))
    start_process, launched = fake_remote_launch(monkeypatch, rollout_runtime)
    monkeypatch.setattr(runtime_, "_start_shard_process", start_process)
    try:
        first = lease(runtime_)
        states = [{"endpoint": "http://router:30000", "healthy": True,
                   "weight_version": "2"}]
        runtime_.registry.publish_version(states, "2")
        grown = lease(runtime_)
        assert first.name == "gateway"
        assert launched == [{"host": "root@gpu-a", "name": "gw1"}]
        assert grown.name == "gw1" and grown.spec.mode == "remote"
        # The ssh destination spelling becomes the URL host (user@ stripped),
        # with the ports the ready line resolved.
        assert grown.url == "http://gpu-a:41001"
        assert grown.spec.admin_url == "http://gpu-a:41002"
        assert grown.trace_dir == runtime_.run_root / "native-traces" / "shard-gw1"
        assert grown.started and not grown.closed
        # The grown shard joined the live epoch over its real registry — the
        # replay crossed publish_version's barrier precondition.
        assert grown.registry.version == "2" and grown.registry.accepting
        assert runtime_.registry.version == "2"
        # Growth bound no local port and stayed quiet about fallbacks.
        assert len(state.servers) == 1
        assert not (runtime_.run_root / "gateway-spawn-fallback.json").exists()
        evidence = json.loads((runtime_.run_root / "gateway-shards.json").read_text())
        assert [(s["name"], s["mode"]) for s in evidence["shards"]] == [
            ("gateway", "local"), ("gw1", "remote")]
        assert evidence["shards"][1]["admin_url"] == "http://gpu-a:41002"
    finally:
        runtime_.close()


def test_remote_candidates_rotate_and_fall_back_to_local(tmp_path, monkeypatch):
    """Every declared host is tried in rotation; only then does growth bind locally."""
    from harborrl.rollout import runtime as rollout_runtime

    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1,
        gateway_hosts=("gpu-a", "gpu-b"))
    start_process, launched = fake_remote_launch(monkeypatch, rollout_runtime,
                                                 failures=2)
    monkeypatch.setattr(runtime_, "_start_shard_process", start_process)
    try:
        # Entries normalize to (destination, ssh port, identity, url host);
        # a plain string inherits its transport options from ssh config.
        assert runtime_._candidate_hosts() == (
            ("gpu-a", None, None, "gpu-a"), ("gpu-b", None, None, "gpu-b"))
        assert runtime_._candidate_hosts() == (
            ("gpu-b", None, None, "gpu-b"), ("gpu-a", None, None, "gpu-a"))
        lease(runtime_)
        # Both remote candidates time out: growth falls back to the manager
        # machine and records why.
        grown = lease(runtime_)
        assert grown.name == "gw1" and grown.spec.mode == "local"
        assert len(state.servers) == 2  # a local ephemeral port was bound
        record = json.loads(
            (runtime_.run_root / "gateway-spawn-fallback.json").read_text())
        assert record["fallback"] == "local"
        assert [attempt["host"] for attempt in record["attempts"]] == ["gpu-a", "gpu-b"]
        assert all("OSError" in attempt["error"] for attempt in record["attempts"])
    finally:
        runtime_.close()


def test_a_spawn_in_flight_does_not_double_grow(tmp_path, monkeypatch):
    """Concurrent leases at capacity grow exactly once.

    With the initial gateway full (n=2 held), two more leases arrive while
    a remote spawn is in flight: the second waiter re-checks capacity under
    the lease lock after that spawn finishes, so it fills the fresh shard
    instead of pulling up another one.
    """
    from harborrl.rollout import runtime as rollout_runtime

    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=2, concurrency=4,
        gateway_hosts=("gpu-a",))
    start_process, launched = fake_remote_launch(monkeypatch, rollout_runtime,
                                                 delay=0.05)
    monkeypatch.setattr(runtime_, "_start_shard_process", start_process)

    async def scenario():
        await runtime_.lease_shard()
        await runtime_.lease_shard()  # the initial gateway is at capacity
        third, fourth = await asyncio.gather(
            runtime_.lease_shard(), runtime_.lease_shard())
        return third, fourth

    try:
        third, fourth = asyncio.run(scenario())
        assert launched == [{"host": "gpu-a", "name": "gw1"}]
        assert third.name == fourth.name == "gw1"
        assert len(state.servers) == 1  # no local growth happened either
        assert runtime_._auto_inflight == {"gateway": 2, "gw1": 2}
    finally:
        runtime_.close()


def test_spawned_remote_shard_close_reaps_the_transport(tmp_path):
    """close() shuts down via the admin plane, then kills a hung transport."""
    from harborrl.rollout.remote_shard import RemoteShard
    from harborrl.rollout.runtime import GatewayShardSpec, SpawnedRemoteShard

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.killed = False

        async def wait(self):
            if self.killed:
                return self.returncode
            await asyncio.sleep(3600)

        def kill(self):
            self.killed = True
            self.returncode = 1

    spec = GatewayShardSpec(name="gw1", port=41001, url="http://gpu-a:41001",
                            mode="remote", admin_url="http://127.0.0.1:1")
    process = FakeProcess()
    shard = SpawnedRemoteShard(spec, token="tok", trace_dir=tmp_path,
                               process=process)
    shard.SHUTDOWN_GRACE = 0.05
    assert isinstance(shard, RemoteShard)  # presents the remote-shard surface

    async def scenario():
        shard.close()
        await asyncio.sleep(0.2)  # the reap task must run past its grace

    asyncio.run(scenario())
    assert process.killed and process.returncode == 1


def test_spawned_remote_shard_close_waits_for_a_clean_exit(tmp_path):
    """A transport that exits after the admin shutdown is reaped, never killed."""
    from harborrl.rollout.runtime import GatewayShardSpec, SpawnedRemoteShard

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.killed = False

        async def wait(self):
            await asyncio.sleep(0.02)
            self.returncode = 0
            return 0

        def kill(self):
            self.killed = True

    spec = GatewayShardSpec(name="gw1", port=41001, url="http://gpu-a:41001",
                            mode="remote", admin_url="http://127.0.0.1:1")
    process = FakeProcess()
    shard = SpawnedRemoteShard(spec, token="tok", trace_dir=tmp_path,
                               process=process)
    shard.SHUTDOWN_GRACE = 5.0

    async def scenario():
        shard.close()
        await asyncio.sleep(0.3)

    asyncio.run(scenario())
    assert not process.killed and process.returncode == 0


def test_remote_shard_spawn_environment_satisfies_the_service_contract(
        tmp_path, monkeypatch):
    """The inline ssh environment is exactly what ShardServiceConfig parses.

    A missing key would only surface as a remote process failing to start;
    parsing the built environment here pins the contract to compile time.
    """
    from harborrl.gateway.shard_service import ShardServiceConfig
    from harborrl.rollout.runtime import _shard_service_command, _shard_service_environment

    config = shard_config(
        tmp_path, monkeypatch,
        HARBORRL_NATIVE_GATEWAY_HOSTS=json.dumps(["root@gpu-a"]),
        HARBORRL_NATIVE_SHARD_TOKEN_FILE=str(tmp_path / "run-token"),
        HF_CKPT="/shared/ckpt")
    environment = _shard_service_environment(config, "gw1")
    parsed = ShardServiceConfig.from_environment(environment)
    # Both ports bind ephemeral; the manager reads the resolved ports from
    # the ready line and composes the URL itself.
    assert parsed.gateway_port == 0 and parsed.gateway_url == ""
    assert parsed.admin_port == 0
    assert parsed.run_id == config.run_id and parsed.shard_name == "gw1"
    assert parsed.run_root == config.run_root
    assert parsed.token_file == config.shard_token_file
    assert parsed.endpoint == config.endpoint
    assert parsed.profile_path == config.profile_path
    assert parsed.model_tag == config.model_tag
    assert parsed.max_output_tokens == config.max_output_tokens
    assert parsed.max_context == config.max_context
    assert parsed.expected_tokenizer_digest == config.expected_tokenizer_digest
    assert parsed.expected_template_digest == config.expected_template_digest
    assert parsed.audited_raw_logprobs == config.audited_raw_logprobs

    command = _shard_service_command(
        "root@gpu-a", environment, python="/train/python", project="/project")
    assert command[0] == "ssh" and command[-2] == "root@gpu-a"
    assert {"BatchMode=yes", "ClearAllForwardings=yes",
            "StrictHostKeyChecking=accept-new"} <= set(command)
    remote = command[-1]
    assert remote.startswith("cd /project && exec env ")
    assert remote.endswith("/train/python -m harborrl.gateway.shard_service")
    assert "RUN_ID=run-1" in remote and "HARBORRL_SHARD_NAME=gw1" in remote
    # ssh forwards no environment: every value crosses shell-quoted.
    quoted = _shard_service_command(
        "gpu-a", {"K": "a b"}, python="/p", project="/pr")[-1]
    assert "K='a b'" in quoted


# -- group dispatch -----------------------------------------------------------


def test_group_dispatch_leases_and_releases_gateways(tmp_path, monkeypatch):
    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, concurrency=2)
    from harborrl.rollout import runtime as rollout_runtime

    seen = []

    async def fake_attempt(identity, **kwargs):
        await asyncio.sleep(0)  # overlap the two slots so both leases are live
        seen.append(kwargs["gateway_url"])
        return {"identity": identity.to_dict()}

    monkeypatch.setattr(rollout_runtime, "GroupSlots", PermissiveSlots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", fake_attempt)
    try:
        result = asyncio.run(rollout_runtime.run_group(
            runtime_.config, runtime_, entry=group_entry(tmp_path), group_index=0,
            rollout_id=0, sampling_digest="s" * 64, policy_version="1",
            group_size=2))
        assert len(result) == 2
        # n=1 and two concurrent slots: each running sandbox gets its own
        # gateway, and the second one was pulled up on demand.
        assert len(set(seen)) == 2
        assert len(state.servers) == 2
        # Every lease was released once the group settled.
        assert runtime_._auto_inflight == {}
    finally:
        runtime_.close()


def test_failed_attempt_releases_its_gateway_lease(tmp_path, monkeypatch):
    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, concurrency=1)
    from harborrl.rollout import runtime as rollout_runtime

    calls = []

    async def flaky_attempt(identity, **kwargs):
        calls.append(identity.slot_id)
        if identity.slot_id == "0" and calls.count("0") == 1:
            raise ValueError("sandbox exploded")
        await asyncio.sleep(0)
        return {"identity": identity.to_dict()}

    monkeypatch.setattr(rollout_runtime, "GroupSlots", PermissiveSlots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", flaky_attempt)
    try:
        result = asyncio.run(rollout_runtime.run_group(
            runtime_.config, runtime_, entry=group_entry(tmp_path), group_index=0,
            rollout_id=0, sampling_digest="s" * 64, policy_version="1",
            group_size=2))
        assert len(result) == 2
        # Slot 0 failed once and its retry succeeded; the semaphore may wake
        # slot 1 between the two attempts, so the order stays unspecified.
        assert sorted(calls) == ["0", "0", "1"]
        # A released lease is reusable: no leaked occupancy ever spawns a
        # gateway that the running sandbox count does not justify.
        assert len(state.servers) == 1
        assert runtime_._auto_inflight == {}
    finally:
        runtime_.close()


def test_auto_group_below_capacity_stays_on_one_gateway(tmp_path, monkeypatch):
    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=30, concurrency=2)
    from harborrl.rollout import runtime as rollout_runtime

    async def fake_attempt(identity, **kwargs):
        await asyncio.sleep(0)
        return {"identity": identity.to_dict()}

    monkeypatch.setattr(rollout_runtime, "GroupSlots", PermissiveSlots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", fake_attempt)
    try:
        result = asyncio.run(rollout_runtime.run_group(
            runtime_.config, runtime_, entry=group_entry(tmp_path), group_index=0,
            rollout_id=0, sampling_digest="s" * 64, policy_version="1",
            group_size=2))
        assert len(result) == 2
        assert len(state.servers) == 1
        assert runtime_._auto_inflight == {}
    finally:
        runtime_.close()


# -- auto shrink ---------------------------------------------------------------


def test_settlement_retires_idle_spawned_gateways(tmp_path, monkeypatch):
    """Group settlement is the shrink trigger.

    n=1 with two concurrent slots pulls up gw1 mid-group; once the group
    settles and every lease is released, the running sandbox count no longer
    justifies gw1 — it detaches, closes, and the evidence record shrinks
    back to exactly the initial gateway.
    """
    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, concurrency=2)
    from harborrl.rollout import runtime as rollout_runtime

    async def fake_attempt(identity, **kwargs):
        await asyncio.sleep(0)  # overlap both slots so gw1 is pulled up
        return {"identity": identity.to_dict()}

    monkeypatch.setattr(rollout_runtime, "GroupSlots", PermissiveSlots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", fake_attempt)
    try:
        result = asyncio.run(rollout_runtime.run_group(
            runtime_.config, runtime_, entry=group_entry(tmp_path), group_index=0,
            rollout_id=0, sampling_digest="s" * 64, policy_version="1",
            group_size=2))
        assert len(result) == 2
        assert len(state.servers) == 2  # gw1 was pulled up mid-group
        # Settlement retired it: one live shard, one closed server, and the
        # evidence record shrank back with the live set.
        assert [shard.name for shard in runtime_.shards] == ["gateway"]
        assert len(state.closed) == 1
        evidence = json.loads((runtime_.run_root / "gateway-shards.json").read_text())
        assert [s["name"] for s in evidence["shards"]] == ["gateway"]
        assert runtime_._auto_spawned == set()
        assert runtime_._auto_inflight == {}
    finally:
        runtime_.close()
    assert len(state.closed) == 2  # the initial gateway closes with the runtime


def test_failed_group_settlement_still_retires_spawned_gateways(tmp_path, monkeypatch):
    """A failed group settles too: shrink runs on the failure path as well."""
    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, concurrency=2)
    from harborrl.rollout import runtime as rollout_runtime

    async def broken_attempt(identity, **kwargs):
        await asyncio.sleep(0)  # overlap so gw1 spawns before both fail
        raise ValueError("sandbox exploded")

    monkeypatch.setattr(rollout_runtime, "GroupSlots", PermissiveSlots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", broken_attempt)
    try:
        with pytest.raises(RuntimeError, match="exhausted"):
            asyncio.run(rollout_runtime.run_group(
                runtime_.config, runtime_, entry=group_entry(tmp_path), group_index=0,
                rollout_id=0, sampling_digest="s" * 64, policy_version="1",
                group_size=2))
        assert [shard.name for shard in runtime_.shards] == ["gateway"]
        assert runtime_._auto_inflight == {}
    finally:
        runtime_.close()


def test_retired_shard_names_are_never_reused(tmp_path, monkeypatch):
    """A retired gateway's trace subtree belongs to its own history.

    Retirement advances the name sequence, so regrowth is a fresh gw2 with a
    fresh trace subtree instead of colliding with retired gw1's evidence.
    """
    runtime_, state = build_auto_runtime(tmp_path, monkeypatch, sandboxes_per_shard=1)
    try:
        first = lease(runtime_)
        second = lease(runtime_)
        assert first.name == "gateway" and second.name == "gw1"
        runtime_.release_shard(first)
        runtime_.release_shard(second)
        runtime_.retire_idle_shards()
        assert [shard.name for shard in runtime_.shards] == ["gateway"]

        lease(runtime_)
        regrown = lease(runtime_)
        assert regrown.name == "gw2"
        assert regrown.trace_dir == runtime_.run_root / "native-traces" / "shard-gw2"
        assert regrown is not second
    finally:
        runtime_.close()


def test_busy_tail_blocks_retirement(tmp_path, monkeypatch):
    """Only idle shards retire, and only from the tail.

    Leases fill in order, so anything behind a busy shard stays: here gw1
    still runs a sandbox while gw0 is idle — the busy tail blocks, and the
    idle initial gateway is never a retirement candidate anyway.
    """
    runtime_, _ = build_auto_runtime(tmp_path, monkeypatch, sandboxes_per_shard=1)
    try:
        first = lease(runtime_)
        second = lease(runtime_)
        runtime_.release_shard(first)  # gw0 idle, gw1 still busy
        runtime_.retire_idle_shards()
        assert [shard.name for shard in runtime_.shards] == ["gateway", "gw1"]

        runtime_.release_shard(second)
        runtime_.retire_idle_shards()
        assert [shard.name for shard in runtime_.shards] == ["gateway"]
    finally:
        runtime_.close()


def test_leave_detaches_a_shard_from_the_admission_facade():
    """leave is join's mirror under the same lock, and idempotent."""
    from harborrl.rollout.runtime import ShardAdmission

    admission = ShardAdmission([])
    first, second = SimpleNamespace(name="gw0"), SimpleNamespace(name="gw1")
    admission.join(first)
    admission.join(second)
    assert admission.shards == (first, second)
    admission.leave(second)
    assert admission.shards == (first,)
    admission.leave(second)  # a settled retry must not raise on a gone shard
    assert admission.shards == (first,)


# -- deployment schema and env overlay ----------------------------------------


def test_deployment_accepts_auto_routing_with_capacity(tmp_path):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_routing": "auto", "sandboxes_per_shard": 20}))
    manager = config["manager"]
    assert manager["gateway_routing"] == "auto"
    assert manager["sandboxes_per_shard"] == 20


def test_deployment_auto_routing_accepts_a_shard_table(tmp_path):
    """Auto routing with a declared floor is now a supported topology.

    The declared table is fixed at startup; auto routing grows beyond it on
    the growth pool. Declaring the routing explicitly must simply stick.
    """
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_routing": "auto",
        "gateway_shards": [{"name": "gw0"}]}))
    assert config["manager"]["gateway_routing"] == "auto"
    assert [shard["name"] for shard in config["manager"]["gateway_shards"]] == ["gw0"]


@pytest.mark.parametrize("value", [0, -3, "many", 2.5])
def test_deployment_sandboxes_per_shard_fails_closed(tmp_path, value):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import DeploymentError, validate

    with pytest.raises(DeploymentError, match="sandboxes_per_shard"):
        validate(deployment_mapping(tmp_path, manager={"sandboxes_per_shard": value}))


def test_deployment_legacy_manager_shape_stays_unchanged(tmp_path):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    legacy = validate(deployment_mapping(tmp_path))["manager"]
    assert not ({"gateway_shards", "gateway_routing", "sandboxes_per_shard",
                 "worker_shards"} & set(legacy))


def _auto_overlay(tmp_path, manager_over):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment import controller as dispatch

    mapped = deployment_mapping(tmp_path, manager=manager_over)
    mapped["gpu_target"] = {"train_python": "/train/python"}
    mapped["rollout"] = {
        "provider": "ssh_docker",
        "workers": [{"id": "worker-a", "ssh": {"host": "worker-a"},
                     "runner_python": "/runner/python"}]}
    return dispatch._rollout_environment({"config": {"harbor": {}}}, mapped, {})


def test_rollout_environment_carries_auto_routing_and_capacity(tmp_path):
    overlay = _auto_overlay(
        tmp_path, {"gateway_routing": "auto", "sandboxes_per_shard": 20})
    assert overlay["HARBORRL_NATIVE_GATEWAY_ROUTING"] == "auto"
    assert overlay["HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD"] == "20"
    assert "HARBORRL_NATIVE_GATEWAY_SHARDS" not in overlay


def test_rollout_environment_leaves_undeclared_routing_to_the_default(tmp_path):
    overlay = _auto_overlay(tmp_path, {})
    assert "HARBORRL_NATIVE_GATEWAY_ROUTING" not in overlay
    assert "HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD" not in overlay


def test_rollout_environment_carries_an_explicit_static_opt_out(tmp_path):
    overlay = _auto_overlay(tmp_path, {"gateway_routing": "modulo"})
    assert overlay["HARBORRL_NATIVE_GATEWAY_ROUTING"] == "modulo"
    assert "HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD" not in overlay


# -- remote growth through the deployment and dispatch layers ------------------


def test_settlement_retires_a_remotely_spawned_gateway(tmp_path, monkeypatch):
    """Shrink is placement-agnostic: a remote shard retires like a local one."""
    from harborrl.rollout import runtime as rollout_runtime

    runtime_, state = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, concurrency=2,
        gateway_hosts=("gpu-a",))
    start_process, launched = fake_remote_launch(monkeypatch, rollout_runtime)
    monkeypatch.setattr(runtime_, "_start_shard_process", start_process)

    async def fake_attempt(identity, **kwargs):
        await asyncio.sleep(0)  # overlap both slots so gw1 is pulled up
        return {"identity": identity.to_dict()}

    monkeypatch.setattr(rollout_runtime, "GroupSlots", PermissiveSlots)
    monkeypatch.setattr(rollout_runtime, "run_attempt", fake_attempt)
    try:
        result = asyncio.run(rollout_runtime.run_group(
            runtime_.config, runtime_, entry=group_entry(tmp_path), group_index=0,
            rollout_id=0, sampling_digest="s" * 64, policy_version="1",
            group_size=2))
        assert len(result) == 2
        assert launched == [{"host": "gpu-a", "name": "gw1"}]
        assert len(state.servers) == 1  # growth stayed remote throughout
        # Settlement detached and closed the remote shard like a local one.
        assert [shard.name for shard in runtime_.shards] == ["gateway"]
        evidence = json.loads((runtime_.run_root / "gateway-shards.json").read_text())
        assert [s["name"] for s in evidence["shards"]] == ["gateway"]
        assert runtime_._auto_spawned == set()
        assert runtime_._auto_inflight == {}
    finally:
        runtime_.close()


def test_retire_detaches_and_closes_a_remotely_spawned_gateway(tmp_path, monkeypatch):
    """Retirement leaves the facade and closes the remote shard in place."""
    from harborrl.rollout import runtime as rollout_runtime

    runtime_, _ = build_auto_runtime(
        tmp_path, monkeypatch, sandboxes_per_shard=1, gateway_hosts=("gpu-a",))
    start_process, _ = fake_remote_launch(monkeypatch, rollout_runtime)
    monkeypatch.setattr(runtime_, "_start_shard_process", start_process)
    try:
        first = lease(runtime_)
        grown = lease(runtime_)
        assert grown.name == "gw1"
        runtime_.release_shard(first)
        runtime_.release_shard(grown)
        runtime_.retire_idle_shards()
        assert [shard.name for shard in runtime_.shards] == ["gateway"]
        assert grown.closed
        assert runtime_.registry.shards == (first,)
    finally:
        runtime_.close()


def test_deployment_accepts_gateway_hosts(tmp_path):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_hosts": ["root@gpu-a", "gpu-b"]}))
    assert config["manager"]["gateway_hosts"] == ["root@gpu-a", "gpu-b"]
    # Without a declared table the pool itself activates auto routing.
    assert config["manager"]["gateway_routing"] == "auto"
    assert "gateway_shards" not in config["manager"]


def test_deployment_accepts_a_static_table_with_the_growth_pool(tmp_path):
    """The tunnel floor plus elastic growth, declared in one deployment."""
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_hosts": ["root@gpu-a"],
        "gateway_shards": [
            {"name": "static-1", "mode": "remote",
             "url": "http://tunnel-host:5010", "admin_url": "http://tunnel-host:5011"}]}))
    assert [shard["name"] for shard in config["manager"]["gateway_shards"]] == ["static-1"]
    assert config["manager"]["gateway_hosts"] == ["root@gpu-a"]
    # Undeclared routing with a pool means growth beyond the declared floor.
    assert config["manager"]["gateway_routing"] == "auto"


def test_deployment_accepts_growth_host_mappings(tmp_path):
    """An ephemeral host carries its own key and port instead of ssh config."""
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_hosts": [
            {"host": "10.0.0.7", "user": "root", "port": 2222,
             "identity": str(tmp_path / "id_ed25519")},
            "root@gpu-b"]}))
    assert config["manager"]["gateway_hosts"] == [
        {"host": "10.0.0.7", "user": "root", "port": 2222,
         "identity": str(tmp_path / "id_ed25519")},
        "root@gpu-b"]


def test_deployment_accepts_shard_python_with_the_growth_pool(tmp_path):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import DeploymentError, validate

    config = validate(deployment_mapping(tmp_path, manager={
        "gateway_hosts": ["gpu-a"],
        "shard_python": "/serving/venv/bin/python"}))
    assert config["manager"]["shard_python"] == "/serving/venv/bin/python"
    # The interpreter exists only for growth; without a pool it is a typo.
    with pytest.raises(DeploymentError, match="gateway_hosts"):
        validate(deployment_mapping(tmp_path, manager={
            "shard_python": "/serving/venv/bin/python"}))


@pytest.mark.parametrize("manager_over, message", [
    ({"gateway_hosts": ["root@gpu-a", "root@gpu-a"]}, "unique"),
    ({"gateway_hosts": []}, "nonempty list"),
    ({"gateway_hosts": "root@gpu-a"}, "nonempty list"),
    ({"gateway_hosts": [""]}, "nonempty list"),
    ({"gateway_hosts": ["gpu a"]}, "nonempty list"),
    ({"gateway_hosts": ["gpu-a"], "gateway_routing": "modulo"},
     "auto routing"),
])
def test_deployment_gateway_hosts_fail_closed(tmp_path, manager_over, message):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import DeploymentError, validate

    with pytest.raises(DeploymentError, match=message):
        validate(deployment_mapping(tmp_path, manager=manager_over))


def test_deployment_gateway_hosts_require_a_service_mode_manager(tmp_path):
    from test_deployment_config import deployment_mapping

    from harborrl.deployment.schema import DeploymentError, validate

    values = deployment_mapping(tmp_path, manager={"gateway_hosts": ["gpu-a"]})
    values["manager"]["mode"] = "embedded"
    with pytest.raises(DeploymentError, match="service-mode"):
        validate(values)


def test_rollout_environment_carries_gateway_hosts(tmp_path):
    overlay = _auto_overlay(tmp_path, {"gateway_hosts": ["root@gpu-a"]})
    assert overlay["HARBORRL_NATIVE_GATEWAY_HOSTS"] == '["root@gpu-a"]'
    # Remotely spawned shards authenticate on the shared run token.
    assert overlay["HARBORRL_NATIVE_SHARD_TOKEN_FILE"]
    assert "HARBORRL_NATIVE_SHARD_PYTHON" not in overlay


def test_rollout_environment_carries_shard_python(tmp_path):
    overlay = _auto_overlay(tmp_path, {
        "gateway_hosts": ["root@gpu-a"],
        "shard_python": "/serving/venv/bin/python"})
    assert overlay["HARBORRL_NATIVE_SHARD_PYTHON"] == "/serving/venv/bin/python"


def test_rollout_environment_omits_gateway_hosts_when_undeclared(tmp_path):
    overlay = _auto_overlay(tmp_path, {})
    assert "HARBORRL_NATIVE_GATEWAY_HOSTS" not in overlay
