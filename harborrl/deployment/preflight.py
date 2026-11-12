"""Role-layered preflight graph shared by train, doctor and quickstart.

Every check names the role and machine it ran against, the outcome, and a
user-executable fix. The graph never raises: failures are data, and any
failed mandatory check keeps a run from being created.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from harborrl.deployment import remote
from harborrl.deployment.schema import destination


def _check(role, machine, name, ok, detail="", hint=None, duration_ms=None):
    record = {"role": role, "machine": machine, "check": name, "ok": bool(ok),
              "detail": str(detail)}
    if hint:
        record["hint"] = str(hint)
    if duration_ms is not None:
        record["duration_ms"] = round(duration_ms, 3)
    return record


def _timed(record_factory):
    started = time.monotonic()
    record = record_factory()
    record["duration_ms"] = (time.monotonic() - started) * 1000
    return record


def _bindable(host, port):
    """Probe with the same address reuse semantics as our HTTP servers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen()
            return True, "port is free"
        except OSError as exc:
            return False, f"cannot bind manager listener on {host}:{port}: {exc}"


def _manager_checks(deployment):
    manager, storage = deployment["manager"], deployment["storage"]
    checks = []
    run_root = Path(storage["run_root"])
    exists = run_root.is_dir()
    writable = exists and _touch_ok(run_root) if storage["manager_writable"] else exists
    checks.append(_timed(lambda: _check(
        "manager", manager["id"], "storage.run_root", exists and writable, str(run_root),
        hint="create the shared run root and grant the manager write access")))

    if manager["mode"] == "service":
        token = Path(manager["run_token_file"])
        token_ok = token.is_file()
        detail = str(token)
        if token_ok:
            mode = token.stat().st_mode & 0o777
            token_ok = mode & 0o077 == 0
            detail = f"{token} mode {oct(mode)}"
        checks.append(_timed(lambda: _check(
            "manager", manager["id"], "manager.run_token_file", token_ok, detail,
            hint="install the run-scoped token with chmod 600")))

        for name, port in (("manager.control_port", manager["control_port"]),
                           ("manager.gateway_port", manager["gateway_port"])):
            started = time.monotonic()
            ok, detail = _bindable(manager["bind"], port)
            checks.append(_check(
                "manager", manager["id"], name, ok, detail,
                hint=f"free the port or stop the conflicting listener on {manager['bind']}:{port}",
                duration_ms=(time.monotonic() - started) * 1000))
    return checks


def _touch_ok(directory):
    probe = directory / ".harborrl-preflight-probe"
    try:
        probe.write_text("write probe\n")
        probe.unlink()
        return True
    except OSError:
        return False


def _local_runner():
    def run(command, *, timeout=30):
        try:
            return subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout, check=False, shell=isinstance(command, str))
        except (subprocess.TimeoutExpired, OSError) as exc:
            return subprocess.CompletedProcess(command, 255, "", f"{type(exc).__name__}: {exc}")

    def test(expression):
        result = run(f"test {expression} && echo ok")
        return result.returncode == 0 and result.stdout.strip() == "ok", result

    return run, test


def gpu_checks(plan, deployment):
    """Probes that need a live GPU machine; the dispatcher reruns them.

    Plan-time preflight cannot probe a machine that provisioning has not
    created yet, so it defers; the dispatcher calls this again once ``up``
    has produced one, before anything is launched.
    """
    gpu, connectivity = deployment["gpu_target"], deployment["connectivity"]
    config = plan["config"]
    machine = gpu["id"]
    checks = []
    if gpu["transport"] == "ssh" and gpu.get("ssh") is None:
        # A provisioned target has no machine until the dispatch obtains one:
        # probing here would test nothing, and a green check would be a lie.
        checks.append(_check(
            "gpu_target", machine, "gpu_target.provision", True,
            "deferred: gpu_target.provision obtains the machine at dispatch, "
            "which then reruns these checks before launching"))
        return checks
    if gpu["transport"] == "ssh":
        spec = gpu["ssh"]
        started = time.monotonic()
        probe = remote.run_ssh(spec, "true", timeout=connectivity["ssh_timeout_s"] + 5,
                               connect_timeout=connectivity["ssh_timeout_s"])
        checks.append(_check(
            "gpu_target", machine, "gpu_target.ssh", probe.returncode == 0,
            probe.stderr.strip()[-300:] or "non-interactive ssh ok",
            hint="the SSH config alias, agent or identity file must allow non-interactive login",
            duration_ms=(time.monotonic() - started) * 1000))
        if probe.returncode != 0:
            checks.append(_check(
                "gpu_target", machine, "gpu_target.further_probes", True,
                "skipped: fix gpu_target.ssh first, then rerun for path, model and GPU checks"))
            return checks

        def execute(command):
            return remote.run_ssh(spec, command, timeout=connectivity["command_timeout_s"],
                                  connect_timeout=connectivity["ssh_timeout_s"])

        def shell_test(expression):
            ok, result = remote.test_remote(spec, expression,
                                            timeout=connectivity["command_timeout_s"],
                                            connect_timeout=connectivity["ssh_timeout_s"])
            return ok, result
    else:
        run, test = _local_runner()

        def execute(command):
            return run(command, timeout=connectivity["command_timeout_s"])

        def shell_test(expression):
            return test(expression)

    def path_check(name, expression, target, hint):
        started = time.monotonic()
        ok, _result = shell_test(expression)
        checks.append(_check("gpu_target", machine, name, ok, target,
                             hint=hint, duration_ms=(time.monotonic() - started) * 1000))

    path_check("gpu_target.source_dir", f"-d {remote.quote(gpu['source_dir'])}", gpu["source_dir"],
               "mount or clone the HarborRL checkout at gpu_target.source_dir")
    path_check("gpu_target.train_python", f"-x {remote.quote(gpu['train_python'])}", gpu["train_python"],
               "provision the training interpreter at gpu_target.train_python")
    if gpu.get("environment_file"):
        path_check("gpu_target.environment_file",
                   f"-f {remote.quote(gpu['environment_file'])}", gpu["environment_file"],
                   "provide the controlled environment variable map")
    for key in ("checkpoint", "reference"):
        target = config["model"][key]
        path_check(f"model.{key}", f"-d {remote.quote(target)}", target,
                   "the model checkpoint must exist in GPU-machine path semantics")

    started = time.monotonic()
    gpu_list = execute("nvidia-smi --query-gpu=index --format=csv,noheader")
    visible = len([line for line in gpu_list.stdout.splitlines() if line.strip()])
    checks.append(_check(
        "gpu_target", machine, "gpu_target.expected_gpus",
        gpu_list.returncode == 0 and visible >= gpu["expected_gpus"],
        f"{visible} visible GPUs (returncode {gpu_list.returncode})",
        hint="check CUDA visibility and the GPU allocation before training",
        duration_ms=(time.monotonic() - started) * 1000))
    return checks


def _worker_checks(plan, deployment):
    workers, connectivity = deployment["rollout"]["workers"], deployment["connectivity"]
    expected_harbor = plan["config"]["harbor"]["version"]
    checks = []
    for worker in workers:
        spec, machine = worker["ssh"], worker["id"]
        started = time.monotonic()
        probe = remote.run_ssh(spec, "true", timeout=connectivity["ssh_timeout_s"] + 5,
                               connect_timeout=connectivity["ssh_timeout_s"])
        checks.append(_check(
            "rollout_worker", machine, "worker.ssh", probe.returncode == 0,
            probe.stderr.strip()[-300:] or "non-interactive ssh ok",
            hint=f"SSH access to {destination(spec)} must work without a password prompt",
            duration_ms=(time.monotonic() - started) * 1000))
        if probe.returncode != 0:
            continue

        def worker_test(spec, machine, name, expression, target, hint):
            started = time.monotonic()
            ok, _result = remote.test_remote(spec, expression,
                                             timeout=connectivity["command_timeout_s"],
                                             connect_timeout=connectivity["ssh_timeout_s"])
            checks.append(_check("rollout_worker", machine, name, ok, target,
                                 hint=hint, duration_ms=(time.monotonic() - started) * 1000))

        worker_test(spec, machine, "worker.source_dir",
                    f"-d {remote.quote(worker['source_dir'])}",
                    worker["source_dir"], "mount the HarborRL checkout on the worker")
        worker_test(spec, machine, "worker.runner_python",
                    f"-f {remote.quote(worker['runner_python'])}",
                    worker["runner_python"], "provision the runner interpreter on the worker")
        for index, task_root in enumerate(worker["task_roots"]):
            worker_test(spec, machine, f"worker.task_roots[{index}]",
                        f"-d {remote.quote(task_root)}",
                        task_root, "mount the task tree on the worker")

        started = time.monotonic()
        probe_command = (f"cd {remote.quote(worker['source_dir'])} && "
                         f"{remote.quote(worker['runner_python'])} -m "
                         "harborrl.rollout.harbor_job.runner --probe")
        result = remote.run_ssh(spec, probe_command, timeout=connectivity["command_timeout_s"],
                                connect_timeout=connectivity["ssh_timeout_s"])
        runtime_ok = harbor_ok = False
        detail = result.stderr.strip()[-300:] or result.stdout.strip()[-300:]
        try:
            payload = json.loads(result.stdout)
            parts = payload.get("python", "").split(".")[:2]
            runtime_ok = (result.returncode == 0
                          and len(parts) == 2 and all(p.isdigit() for p in parts)
                          and tuple(map(int, parts)) >= (3, 12))
            harbor_ok = payload.get("harbor_version") == expected_harbor
            if result.returncode == 0:
                detail = f"python {payload.get('python')}, harbor {payload.get('harbor_version')}"
        except (ValueError, AttributeError, TypeError):
            pass
        checks.append(_check(
            "rollout_worker", machine, "worker.runner_runtime", runtime_ok, detail,
            hint="runner Python must be >= 3.12 with the pinned Harbor installation",
            duration_ms=(time.monotonic() - started) * 1000))
        checks.append(_check(
            "rollout_worker", machine, "worker.harbor_version", harbor_ok, detail,
            hint=f"pin Harbor {expected_harbor} in the runner environment",
            duration_ms=(time.monotonic() - started) * 1000))

        started = time.monotonic()
        docker = remote.run_ssh(spec, "docker info --format '{{.ServerVersion}}'",
                                timeout=connectivity["command_timeout_s"],
                                connect_timeout=connectivity["ssh_timeout_s"])
        checks.append(_check(
            "rollout_worker", machine, "worker.docker", docker.returncode == 0,
            docker.stdout.strip()[:120] or docker.stderr.strip()[-200:],
            hint="start the Docker daemon; Harbor trials require container execution",
            duration_ms=(time.monotonic() - started) * 1000))
    return checks


def _control_loop_checks(deployment):
    manager, connectivity = deployment["manager"], deployment["connectivity"]
    checks = []
    if manager["mode"] != "service":
        return checks
    dials = (("connectivity.gpu_to_controller", manager["control_url"], connectivity["gpu_to_controller"]),
             ("connectivity.worker_to_gateway", manager["gateway_url"], connectivity["worker_to_gateway"]))
    for name, url, enabled in dials:
        if not enabled:
            continue
        parts = urlsplit(url)
        started = time.monotonic()
        try:
            socket.getaddrinfo(parts.hostname, parts.port)
            ok, detail = True, "address resolves from the manager"
        except socket.gaierror as exc:
            ok, detail = False, f"{exc}"
        checks.append(_check("control_loop", manager["id"], name, ok, detail,
                             hint="fix DNS or firewall so the address resolves for every role",
                             duration_ms=(time.monotonic() - started) * 1000))
        started = time.monotonic()
        serving = False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(min(connectivity["http_timeout_s"], 5))
                serving = client.connect_ex((parts.hostname, parts.port)) == 0
        except OSError:
            serving = False  # unreachable now; the post-start dial retries
        checks.append(_check(
            "control_loop", manager["id"], name + ".dial", True,
            "listener already serving" if serving
            else "no listener yet; the live dial reruns after components start",
            duration_ms=(time.monotonic() - started) * 1000))
    if connectivity["manager_to_gpu_inference"]:
        checks.append(_check(
            "control_loop", manager["id"], "connectivity.manager_to_gpu_inference", True,
            "deferred: the GPU job publishes its inference endpoint after serving is healthy"))
    return checks


def preflight(plan, deployment, *, live=True):
    """Run every deployment layer; remote probes are skipped when live=False."""
    if not live:
        return [_check("deployment", deployment["manager"]["id"], "deployment.config",
                       True, "dry-run: parsed and validated, remote probes skipped")]
    checks = []
    checks.extend(_manager_checks(deployment))
    checks.extend(gpu_checks(plan, deployment))
    if deployment["rollout"]["provider"] == "ssh_docker":
        checks.extend(_worker_checks(plan, deployment))
    checks.extend(_control_loop_checks(deployment))
    return checks


def readiness_checks(plan, deployment, *, from_specs=()):
    """Post-start dials: cross-machine health after components are listening.

    ``from_specs`` maps a role to the ssh spec that should perform the dial;
    callers pass the GPU target and rollout workers of a live deployment.
    """
    manager, connectivity = deployment["manager"], deployment["connectivity"]
    checks = []
    dials = (("gpu_target", "gpu_to_controller.ready", manager["control_url"],
              connectivity["gpu_to_controller"]),
             ("rollout_worker", "worker_to_gateway.ready", manager["gateway_url"],
              connectivity["worker_to_gateway"]))
    specs = dict(from_specs)
    for role, name, url, enabled in dials:
        if not enabled:
            continue
        spec = specs.get(role)
        if spec is None:
            checks.append(_check("control_loop", manager["id"], name, True,
                                 "deferred: no machine mapped for this dial"))
            continue
        path = "/healthz" if "control" in name else "/readyz"
        command = (f"python3 -c "
                   f"\"import urllib.request;urllib.request.urlopen('{url}{path}',"
                   f"timeout={connectivity['http_timeout_s']})\"")
        started = time.monotonic()
        result = remote.run_ssh(spec, command, timeout=connectivity["command_timeout_s"],
                                connect_timeout=connectivity["ssh_timeout_s"])
        checks.append(_check("control_loop", destination(spec), name,
                             result.returncode == 0,
                             result.stderr.strip()[-300:] or "health dial ok",
                             hint=f"open the route from {destination(spec)} to {url}{path}",
                             duration_ms=(time.monotonic() - started) * 1000))
    return checks
