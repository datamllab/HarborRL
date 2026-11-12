"""Service-mode dispatch: the manager-side controller flow (P1 + P3).

An independent CPU manager runs the same preflight graph, materializes the
run on shared storage, starts the RolloutService control plane, launches the
existing Slime job on the GPU target over SSH, and waits for the shared
exit-code record. Rollout execution itself moves to the manager only through
the service; training stays on the GPU.
Hub mode extends service mode by allowing the manager to accept multiple runs
as a long-running daemon. Jobs are submitted via HTTP API and queued for
execution on available GPU resources.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

from harborrl.config import ROOT
from harborrl.deployment import remote
from harborrl.deployment.schema import destination
from harborrl.deployment.training_target import SshTrainingTarget
from harborrl.platform.evidence import (
    _materialize,
    _source_audit,
    effective_config,
    new_run_root,
    run_environment,
)
from harborrl.rollout.service import RolloutService, make_control_server
from harborrl.trajectories.native import publish

TOKEN_MODE = 0o600

PORT_SLOTS = 16
_ALLOC_LOCK = threading.Lock()
_ALLOCATED = {}  # port -> run_id, across control and gateway labels


def _allocate_port(bind, base, run_id):
    """Pick the first bindable port from ``base`` within PORT_SLOTS candidates.

    The probe closes the socket before the real server binds, so a race can
    still fail the later bind loudly; allocation never silently shares a port.
    Candidates already held by another label of the same run are skipped so
    control and gateway ranges may safely overlap.
    """
    with _ALLOC_LOCK:
        for candidate in range(base, base + PORT_SLOTS):
            if candidate in _ALLOCATED:
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind((bind, candidate))
            except OSError:
                continue
            _ALLOCATED[candidate] = run_id
            return candidate
    raise RuntimeError(f"no free manager port within {PORT_SLOTS} slots from {base}")


def _rewrite_port(url, port):
    parts = urlsplit(url)
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}:{port}"


def _with_allocated_ports(deployment, run_id):
    """Give one dispatch exclusive manager ports; concurrent runs then coexist.

    The deployment's declared ports act as the base of a small allocation
    range: every consumer (control server, rollout runtime, GPU environment
    overlay, evidence records) reads the patched manager mapping. Every
    LOCAL gateway shard claims its own exclusive port: the declared
    gateway_port is the first local shard's base and undeclared shard ports
    default to consecutive slots, so sibling runs may never land on the same
    gateway. Remote shards bind nothing here; their origins pass through.
    """
    manager = deployment["manager"]
    control = _allocate_port(manager["bind"], manager["control_port"], run_id)
    declared = manager.get("gateway_shards") or [{"name": "gateway"}]
    allocated = []
    for index, spec in enumerate(declared):
        if spec.get("mode", "local") == "remote":
            allocated.append(None)
            continue
        base = spec.get("port", manager["gateway_port"] + index)
        allocated.append(_allocate_port(manager["bind"], base, run_id))
    local_ports = [port for port in allocated if port is not None]
    patched = dict(manager)
    if local_ports:
        patched.update({
            "control_port": control, "control_url": _rewrite_port(manager["control_url"], control),
            "gateway_port": local_ports[0], "gateway_url": _rewrite_port(
                manager["gateway_url"], local_ports[0]),
        })
    else:
        patched.update({
            "control_port": control,
            "control_url": _rewrite_port(manager["control_url"], control),
        })
    if manager.get("gateway_shards"):
        patched["gateway_shards"] = [
            {**spec, "port": port} if port is not None else dict(spec)
            for spec, port in zip(manager["gateway_shards"], allocated)]
    return {**deployment, "manager": patched}


def _release_run_ports(run_id):
    with _ALLOC_LOCK:
        for port, owner in list(_ALLOCATED.items()):
            if owner == run_id:
                del _ALLOCATED[port]


def check_gpu_control(deployment):
    manager, gpu = deployment['manager'], deployment['gpu_target']
    timeout = deployment['connectivity']['http_timeout_s']
    result = remote.run_ssh(gpu['ssh'] or {'host': 'localhost'},
        f"curl --noproxy '*' -fsS --max-time {timeout} " + remote.quote(manager['control_url'] + '/healthz'),
        timeout=timeout + 15)
    if result.returncode != 0:
        raise RuntimeError('GPU cannot reach manager control service')
    payload = json.loads(result.stdout)
    if payload != {'ok': True, 'service': 'harborrl-rollout-service-v1'}:
        raise RuntimeError('GPU reached an unexpected control service')
    return {'gpu_to_controller': True, 'control_url': manager['control_url']}


def check_gateway_shards(deployment):
    """Preflight every remote shard before the GPU job can depend on it.

    Each remote process must answer as its declared shard name with the run
    token; a missing or misnamed shard fails dispatch loudly instead of
    failing attempts mid-training.
    """
    manager = deployment['manager']
    specs = [spec for spec in manager.get('gateway_shards') or []
             if spec.get('mode', 'local') == 'remote']
    if not specs:
        return None
    from harborrl.rollout.remote_shard import RemoteShard, read_shard_token
    from harborrl.rollout.runtime import GatewayShardSpec

    token = read_shard_token(manager['run_token_file'])
    checks = []
    for spec in specs:
        shard = RemoteShard(
            GatewayShardSpec(spec['name'], urlsplit(spec['url']).port, spec['url'],
                             mode='remote', admin_url=spec['admin_url']),
            token=token, trace_dir='unused')
        shard.start()  # Raises on unreachable or wrong shard identity.
        checks.append({'name': spec['name'], 'admin_url': spec['admin_url'],
                       'gateway_url': spec['url'], 'ok': True})
    return checks


def _ensure_token(token_file):
    """Load the run token, creating a fresh one on first use."""
    path = Path(token_file)
    if path.is_file():
        token = path.read_text().strip()
        if token:
            return token
    import secrets

    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, TOKEN_MODE)
    with os.fdopen(fd, "w") as stream:
        stream.write(token + "\n")
    return token


def _rollout_environment(plan, deployment, environment):
    """Environment overlay that points the GPU job at the manager service."""
    manager = deployment["manager"]
    workers = deployment["rollout"]["workers"]
    overlay = dict(environment)
    overlay.update({
        "HARBORRL_ROLLOUT_MODE": "service",
        "HARBORRL_ROLLOUT_SERVICE_URL": manager["control_url"],
        "HARBORRL_ROLLOUT_SERVICE_TOKEN_FILE": manager['run_token_file'],
        "TRAIN_PYTHON": deployment['gpu_target']['train_python'],
        "HARBORRL_NATIVE_GATEWAY_HOST": manager["bind"],
        "HARBORRL_NATIVE_GATEWAY_PORT": str(manager["gateway_port"]),
        "HARBORRL_NATIVE_GATEWAY_URL": manager["gateway_url"],
        "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(
            [destination(worker["ssh"]) for worker in workers]),
        "HARBORRL_NATIVE_RUNNER_PYTHON": (plan['config']['harbor']['python']
                                         if deployment['rollout']['provider'] == 'sandbox'
                                         else workers[0]['runner_python']),
    })
    if deployment['rollout']['provider'] == 'sandbox':
        if 'HARBORRL_NATIVE_SANDBOX' not in overlay:
            raise ValueError('sandbox deployment requires sandbox training config')
        transport = plan['config']['harbor'].get('runner_transport', 'ssh')
        if (transport == 'local' and workers) or (transport == 'ssh' and not workers):
            raise ValueError('sandbox runner transport and deployment workers disagree')
        overlay['HARBORRL_NATIVE_RUNNER_TRANSPORT'] = transport
    shards = manager.get("gateway_shards")
    if shards:
        entries = []
        for spec in shards:
            if spec.get("mode", "local") == "remote":
                entries.append({"name": spec["name"], "mode": "remote",
                                "url": spec["url"], "admin_url": spec["admin_url"]})
            else:
                entries.append({"name": spec["name"], "port": spec["port"]})
        overlay["HARBORRL_NATIVE_GATEWAY_SHARDS"] = json.dumps(entries)
        if any(spec.get("mode") == "remote" for spec in shards):
            # The remote processes read the same shared-storage token file.
            overlay["HARBORRL_NATIVE_SHARD_TOKEN_FILE"] = manager["run_token_file"]
        worker_shards = manager.get("worker_shards")
        if worker_shards:
            # The deployment speaks worker ids; the runtime routes by SSH
            # destination (or the 'local' transport marker).
            destinations = {worker["id"]: destination(worker["ssh"]) for worker in workers}
            overlay["HARBORRL_NATIVE_GATEWAY_WORKER_SHARDS"] = json.dumps(
                {destinations.get(key, key): shard
                 for key, shard in worker_shards.items()}, sort_keys=True)
    routing = manager.get("gateway_routing")
    if routing is not None:
        # Declared even without a shard table: an explicit routing opts in or
        # out of elastic growth; undeclared leaves the runtime's auto default.
        overlay["HARBORRL_NATIVE_GATEWAY_ROUTING"] = routing
    if manager.get("sandboxes_per_shard") is not None:
        overlay["HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD"] = str(
            manager["sandboxes_per_shard"])
    gateway_hosts = manager.get("gateway_hosts")
    if gateway_hosts:
        # Auto growth fills this pool first and only spawns locally when no
        # candidate answers; remotely spawned shards authenticate on the same
        # shared-storage run token as statically declared ones.
        overlay["HARBORRL_NATIVE_GATEWAY_HOSTS"] = json.dumps(gateway_hosts)
        overlay["HARBORRL_NATIVE_SHARD_TOKEN_FILE"] = manager["run_token_file"]
        if manager.get("shard_python"):
            # Growth hosts run the serving stack, which need not share the
            # runner's sandbox-framework interpreter.
            overlay["HARBORRL_NATIVE_SHARD_PYTHON"] = manager["shard_python"]
    return overlay


def _runtime_factory(deployment, environment, *, concurrency_limit, group_size, tokenizer=None):
    """Build the per-run rollout runtime when the GPU job announces serving."""

    def factory(run_id, router_url):
        from harborrl.rollout.runtime import RolloutRuntime, RolloutRuntimeConfig

        env = {**os.environ, **environment}
        config = RolloutRuntimeConfig.from_environment(
            env, endpoint=router_url, concurrency_limit=concurrency_limit,
            group_size=group_size,
            gateway_host=deployment["manager"]["bind"],
            gateway_port=deployment["manager"]["gateway_port"],
            gateway_url=deployment["manager"]["gateway_url"])
        resolved = tokenizer
        if resolved is None:
            from transformers import AutoTokenizer

            resolved = AutoTokenizer.from_pretrained(
                env["HF_CKPT"], trust_remote_code=True)
        return RolloutRuntime(config, resolved)

    return factory




def check_hub_readiness(deployment):
    """Preflight check: verify hub manager is reachable and healthy."""
    import requests
    
    manager = deployment["manager"]
    hub_url = manager.get("hub_url")
    timeout = deployment["connectivity"]["http_timeout_s"]
    
    if not hub_url:
        raise RuntimeError("hub mode requires manager.hub_url")
    
    token_file = manager.get("run_token_file")
    hub_token = ""
    if token_file:
        try:
            hub_token = open(token_file).read().strip()
        except OSError as exc:
            raise RuntimeError(f"hub mode cannot read manager.run_token_file {token_file}: {exc}")

    try:
        response = requests.get(
            f"{hub_url}/hub/status",
            timeout=timeout,
            headers={"X-Api-Key": hub_token},
        )
        response.raise_for_status()
        status = response.json()

        # The hub server identifies itself with schema_version, not a
        # service field; keep the check loose so both spellings pass.
        identity = status.get("service") or status.get("schema_version")
        if not str(identity or "").startswith("harborrl-hub"):
            raise RuntimeError(f"Unexpected service at {hub_url}: {status}")
        
        return {
            "hub_reachable": True,
            "hub_url": hub_url,
            "hub_status": status,
        }
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Hub manager not reachable at {hub_url}: {e}")


def dispatch_hub(plan, deployment, *, checks):
    """Hub mode: submit job to long-running manager, return immediately.
    
    In hub mode, the manager is a persistent daemon that accepts multiple
    training runs. This function submits the run to the hub HTTP API
    and returns control immediately rather than waiting for completion.
    """
    import requests
    
    manager = deployment["manager"]
    hub_url = manager.get("hub_url")
    
    if not hub_url:
        raise RuntimeError("hub mode requires manager.hub_url to be configured")
    
    try:
        token_file = manager.get("run_token_file")
        hub_token = open(token_file).read().strip() if token_file else ""
        response = requests.post(
            f"{hub_url}/hub/runs",
            json={"plan": plan, "deployment": deployment},
            timeout=deployment["connectivity"]["http_timeout_s"],
            headers={"Content-Type": "application/json", "X-Api-Key": hub_token},
        )
        response.raise_for_status()
        result = response.json()
        
        run_id = result.get("run_id")
        status_url = f"{hub_url}/hub/runs/{run_id}"
        
        print(json.dumps({
            "mode": "hub",
            "status": "submitted",
            "run_id": run_id,
            "status_url": status_url,
            "hub_url": hub_url,
        }), file=sys.stderr)
        
        return 0  # Success - job submitted
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to submit run to hub at {hub_url}: {e}")


def dispatch_service(plan, deployment, *, checks):
    manager, gpu, storage = deployment["manager"], deployment["gpu_target"], deployment["storage"]
    
    # Hub mode: submit to long-running manager instead of direct execution
    mode = manager.get("mode")
    if mode == "hub":
        return dispatch_hub(plan, deployment, checks=checks)
    
    # Standard service mode execution
    config = plan["config"]
    root = new_run_root(storage["run_root"])
    root.mkdir(parents=True, mode=0o700)
    run_id = root.name
    # Concurrent dispatches on one manager must not collide on the declared
    # ports; patch every consumer's view before the environment is rendered.
    deployment = _with_allocated_ports(deployment, run_id)
    manager = deployment["manager"]

    catalog_path, prompt_path = _materialize(plan, root)
    source = _source_audit(root)
    environment = run_environment(plan, root, catalog_path, prompt_path)
    token = _ensure_token(manager["run_token_file"])
    environment = _rollout_environment(plan, deployment, environment)

    sampling = config["sampling"]
    concurrency = sampling["group_size"] * sampling["groups_per_batch"]
    service = RolloutService(
        run_root=root,
        runtime_factory=_runtime_factory(
            deployment, environment,
            concurrency_limit=concurrency, group_size=sampling["group_size"]),
        token_file=manager["run_token_file"])
    control = make_control_server(service, host=manager["bind"],
                                  port=manager["control_port"])
    control_thread = None
    target = None
    exit_code = 1
    try:
        service.register_run(run_id, {
            "run_id": run_id, "deployment_source": plan["deployment"]["source"],
            "source_commit": source["commit"],
            "config_digest": plan["deployment"]["digest"],
            "group_size": sampling["group_size"],
            "workers": [worker["id"] for worker in deployment["rollout"]["workers"]],
        }, token=token)
        control_thread = threading.Thread(
            target=control.serve_forever, name="harborrl-control-plane", daemon=True)
        control_thread.start()
        publish(root / 'control-readiness.json', check_gpu_control(deployment))
        shard_checks = check_gateway_shards(deployment)
        if shard_checks:
            publish(root / 'gateway-shards-readiness.json', shard_checks)

        publish(root / "launch.json", {
            "plan": plan, "run_dir": str(root), "source": source,
            "materialized_catalog": str(catalog_path), "prompt_data": str(prompt_path),
            "control_url": manager["control_url"]})
        publish(root / "effective-config.json",
                effective_config(plan, environment, dict(environment), source, checks))

        target = SshTrainingTarget(
            gpu["ssh"] or {"host": "localhost"}, source_dir=gpu["source_dir"], run_dir=root,
            environment_file=gpu.get("environment_file"),
            command_timeout=deployment["connectivity"]["command_timeout_s"] + 30,
            ssh_timeout=deployment["connectivity"]["ssh_timeout_s"])
        # The launcher executes inside the GPU checkout; manager-side absolute
        # paths are rewritten so shared-storage artifacts keep one spelling.
        target.write_launcher(
            ["bash", str(ROOT / "harborrl/platform/slime_train.sh")], environment,
            path_rewrites=[(str(manager["source_dir"]), str(gpu["source_dir"]))])
        target.launch()

        def forward_signal(signum, frame):
            if target is not None:
                target.cancel()

        previous = {sig: signal.signal(sig, forward_signal)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            exit_code = target.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except BaseException:
        import traceback

        traceback.print_exc()
        if target is not None:
            # Best-effort teardown; the run record keeps the failure evidence.
            with contextlib.suppress(Exception):
                target.cancel()
        raise
    finally:
        with contextlib.suppress(Exception):
            service.drain_run(run_id)
        service.close()
        if control_thread is not None:
            control.shutdown()
            control_thread.join(5)
        control.server_close()
        _release_run_ports(run_id)
        print(json.dumps({"run_dir": str(root), "exit_code": exit_code}), file=sys.stderr)
    return exit_code
