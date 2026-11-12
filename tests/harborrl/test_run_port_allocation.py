"""CPU contract evidence for per-run manager port allocation.

Distinct from live dispatch evidence: these tests exercise the allocator and
URL rewriting only, without SSH, GPU or a real rollout service.
"""
import json
import socket

import pytest

from harborrl.deployment import controller as dispatch


def deployment(**manager_over):
    manager = {
        "id": "manager", "mode": "service", "bind": "127.0.0.1",
        "control_port": 53900, "control_url": "http://127.0.0.1:53900",
        "gateway_port": 53910, "gateway_url": "http://127.0.0.1:53910",
        "run_token_file": "/run/token", "source_dir": "/repo",
        "controller_python": "/python",
    }
    manager.update(manager_over)
    return {"manager": manager, "gpu_target": {}, "storage": {},
            "rollout": {}, "connectivity": {}}


def hold(bind, port):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind((bind, port))
    except OSError:
        holder.close()
        return None
    return holder


@pytest.fixture(autouse=True)
def clean_allocation():
    dispatch._ALLOCATED.clear()
    yield
    dispatch._ALLOCATED.clear()


def test_first_run_keeps_declared_ports_and_urls():
    patched = dispatch._with_allocated_ports(deployment(), "run-a")
    manager = patched["manager"]
    assert manager["control_port"] == 53900
    assert manager["control_url"] == "http://127.0.0.1:53900"
    assert manager["gateway_port"] == 53910
    assert manager["gateway_url"] == "http://127.0.0.1:53910"
    dispatch._release_run_ports("run-a")


def test_busy_port_shifts_only_the_blocked_service():
    holder = hold("127.0.0.1", 53910)
    if holder is None:
        pytest.skip("another process already holds the declared gateway port")
    try:
        patched = dispatch._with_allocated_ports(deployment(), "run-b")
        manager = patched["manager"]
        assert manager["gateway_port"] == 53911
        assert manager["gateway_url"] == "http://127.0.0.1:53911"
        # The untouched control plane keeps its declared port and URL.
        assert manager["control_port"] == 53900
    finally:
        holder.close()
    dispatch._release_run_ports("run-b")


def test_release_returns_the_port_to_the_pool():
    dispatch._with_allocated_ports(deployment(), "run-c")
    dispatch._release_run_ports("run-c")
    assert not dispatch._ALLOCATED
    again = dispatch._with_allocated_ports(deployment(), "run-d")
    assert again["manager"]["gateway_port"] == 53910
    dispatch._release_run_ports("run-d")


def test_control_and_gateway_never_share_a_port(monkeypatch):
    monkeypatch.setattr(dispatch, "PORT_SLOTS", 4)
    # Overlapping ranges: control [53911..53914] inside gateway
    # [53910..53913]. Control allocates first; the gateway allocation must
    # skip both busy ports and the port the control plane already took.
    overlapping = deployment(control_port=53911)
    holders = [hold("127.0.0.1", port) for port in (53910, 53911)]
    if any(h is None for h in holders):
        for holder in holders:
            if holder is not None:
                holder.close()
        pytest.skip("another process already holds a candidate port")
    try:
        patched = dispatch._with_allocated_ports(overlapping, "run-e")
        manager = patched["manager"]
        assert manager["control_port"] == 53912
        assert manager["gateway_port"] == 53913
        assert manager["gateway_url"] == "http://127.0.0.1:53913"
    finally:
        for holder in holders:
            holder.close()
    dispatch._release_run_ports("run-e")


def test_exhausted_range_fails_loudly(monkeypatch):
    monkeypatch.setattr(dispatch, "PORT_SLOTS", 1)
    holder = hold("127.0.0.1", 53910)
    if holder is None:
        pytest.skip("another process already holds the declared gateway port")
    try:
        # Control allocates first and succeeds on its own base.
        with pytest.raises(RuntimeError, match="no free manager port"):
            dispatch._with_allocated_ports(deployment(), "run-f")
    finally:
        holder.close()
    dispatch._release_run_ports("run-f")


def test_gateway_shards_get_exclusive_ports():
    sharded = deployment(gateway_shards=[{"name": "gw0"}, {"name": "gw1"}])
    patched = dispatch._with_allocated_ports(sharded, "run-g")
    manager = patched["manager"]
    # Consecutive default slots after the declared gateway port.
    assert [spec["port"] for spec in manager["gateway_shards"]] == [53910, 53911]
    assert manager["gateway_port"] == 53910
    assert manager["gateway_url"] == "http://127.0.0.1:53910"
    dispatch._release_run_ports("run-g")


def test_busy_shard_port_shifts_only_that_shard():
    holder = hold("127.0.0.1", 53911)
    if holder is None:
        pytest.skip("another process already holds the candidate shard port")
    try:
        sharded = deployment(gateway_shards=[{"name": "gw0"}, {"name": "gw1"}])
        patched = dispatch._with_allocated_ports(sharded, "run-h")
        manager = patched["manager"]
        assert [spec["port"] for spec in manager["gateway_shards"]] == [53910, 53912]
        assert manager["gateway_port"] == 53910
        assert manager["gateway_url"] == "http://127.0.0.1:53910"
    finally:
        holder.close()
    dispatch._release_run_ports("run-h")


def test_rollout_environment_carries_the_shard_table():
    workers = [{"id": "worker-a", "ssh": {"host": "worker-a", "user": "root"},
                "runner_python": "/runner/python"},
               {"id": "worker-b", "ssh": {"host": "worker-b"},
                "runner_python": "/runner/python"}]
    sharded = deployment(
        gateway_shards=[{"name": "gw0", "port": 53910}, {"name": "gw1", "port": 53911}],
        gateway_routing="worker_affinity",
        worker_shards={"worker-a": "gw0", "worker-b": "gw1"})
    sharded["gpu_target"] = {"train_python": "/train/python"}
    sharded["rollout"] = {"workers": workers, "provider": "ssh_docker"}
    overlay = dispatch._rollout_environment(
        {"config": {"harbor": {}}}, sharded, {"BASE": "1"})
    assert json.loads(overlay["HARBORRL_NATIVE_GATEWAY_SHARDS"]) == [
        {"name": "gw0", "port": 53910}, {"name": "gw1", "port": 53911}]
    assert overlay["HARBORRL_NATIVE_GATEWAY_ROUTING"] == "worker_affinity"
    # Worker ids become the SSH destinations the runtime routes by.
    assert json.loads(overlay["HARBORRL_NATIVE_GATEWAY_WORKER_SHARDS"]) == {
        "root@worker-a": "gw0", "worker-b": "gw1"}
    # A single-gateway deployment renders no shard table at all.
    plain = deployment()
    plain["gpu_target"] = {"train_python": "/train/python"}
    plain["rollout"] = {"workers": workers, "provider": "ssh_docker"}
    assert "HARBORRL_NATIVE_GATEWAY_SHARDS" not in dispatch._rollout_environment(
        {"config": {"harbor": {}}}, plain, {"BASE": "1"})
