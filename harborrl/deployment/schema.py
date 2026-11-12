"""Deployment schema v1: machine roles and access, never training semantics.

A deployment config describes WHERE the manager, GPU trainer and rollout
workers live and HOW they reach each other. It deliberately excludes every
training-side field (learning rate, sampling, task catalog) and never stores
secrets: SSH targets must resolve through the invoking user's SSH config,
agent or a restricted identity file.
"""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from harborrl.config import ROOT

DEPLOYMENT_SCHEMA_VERSION = 1
DEPLOYMENT_SECTIONS = {"manager", "gpu_target", "storage", "rollout", "connectivity"}

SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}

MANAGER_REQUIRED = {"id", "mode", "bind", "control_port", "control_url", "gateway_port",
                    "gateway_url", "run_token_file", "source_dir", "controller_python"}
MANAGER_OPTIONAL = {"gateway_shards", "gateway_routing", "sandboxes_per_shard",
                    "worker_shards", "gateway_hosts", "shard_python"}
MANAGER_FIELDS = MANAGER_REQUIRED | MANAGER_OPTIONAL
SHARD_FIELDS = {"name", "port", "mode", "url", "admin_url"}
SHARD_MODES = ("local", "remote")
SHARD_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
GATEWAY_ROUTINGS = ("modulo", "worker_affinity", "auto")
DEFAULT_SANDBOXES_PER_SHARD = 30
SSH_FIELDS = {"host", "user", "port", "identity"}
GPU_REQUIRED = {"id", "transport", "ssh", "source_dir", "train_python", "environment_file",
                "expected_gpus"}
GPU_OPTIONAL = {"provision"}
GPU_FIELDS = GPU_REQUIRED | GPU_OPTIONAL
PROVISION_FIELDS = {"up", "down", "timeout_s"}
DEFAULT_PROVISION_TIMEOUT_S = 900
STORAGE_FIELDS = {"mode", "run_root", "manager_writable", "gpu_writable", "worker_writable"}
WORKER_FIELDS = {"id", "ssh", "source_dir", "runner_python", "docker", "task_roots"}
ROLLOUT_FIELDS = {"provider", "workers"}
CONNECTIVITY_FIELDS = {"command_timeout_s", "ssh_timeout_s", "http_timeout_s",
                       "gpu_to_controller", "worker_to_gateway", "manager_to_gpu_inference"}
TRANSPORTS = ("local", "ssh")
ROLLOUT_PROVIDERS = ("ssh_docker", "embedded", "sandbox")
STORAGE_MODES = ("shared_fs",)


class DeploymentError(ValueError):
    """Raised for any malformed or unsafe deployment configuration."""


def _slug(value, label):
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise DeploymentError(f"{label} must be a short machine identifier")
    return value


def _positive_int(value, label):
    if type(value) is not int or value <= 0 or type(value) is bool:
        raise DeploymentError(f"{label} must be a positive integer")
    return value


def _nonempty_str(value, label):
    if not isinstance(value, str) or not value.strip():
        raise DeploymentError(f"{label} must be a nonempty string")
    return value


def _bool(value, label):
    if type(value) is not bool:
        raise DeploymentError(f"{label} must be a boolean")
    return value


def _absolute_path(value, label):
    value = _nonempty_str(value, label)
    if not Path(value).is_absolute():
        raise DeploymentError(f"{label} must be an absolute path in target-machine semantics")
    return value


def _plain_origin(url, label, *, port=None, allow_loopback=False):
    parts = urlsplit(_nonempty_str(url, label))
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in ("", "/")
            or (port is not None and parts.port != port)):
        raise DeploymentError(f"{label} must be a plain http(s) origin matching its declared port")
    if not allow_loopback and parts.hostname.lower() in LOOPBACK_HOSTS:
        raise DeploymentError(f"{label} must be reachable from other machines; loopback is forbidden")
    return f"{parts.scheme}://{parts.netloc}".rstrip("/")


def _ssh_spec(value, label, *, require_host=True):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - SSH_FIELDS:
        raise DeploymentError(
            f"{label} supports only {sorted(SSH_FIELDS)}; passwords and private keys are never stored")
    host = value.get("host")
    if host is not None:
        host = _nonempty_str(host, f"{label}.host")
        if host.startswith("-") or host.count("@") > 1:
            raise DeploymentError(f"{label}.host must be an SSH destination, not a flag")
    if require_host and not host:
        raise DeploymentError(f"{label}.host is required")
    user = value.get("user")
    if user is not None:
        user = _nonempty_str(user, f"{label}.user")
        if user.startswith("-"):
            raise DeploymentError(f"{label}.user must not look like a flag")
    port = value.get("port")
    if port is not None:
        port = _positive_int(port, f"{label}.port")
    identity = value.get("identity")
    if identity is not None:
        identity = _absolute_path(identity, f"{label}.identity")
    spec = {"host": host}
    if user:
        spec["user"] = user
    if port:
        spec["port"] = port
    if identity:
        spec["identity"] = identity
    return spec


def destination(spec):
    """Render an ssh destination from a normalized spec (user@host)."""
    return f"{spec['user']}@{spec['host']}" if spec.get("user") else spec["host"]


def _growth_host(value, label):
    """Validate one growth-pool entry, preserving its declared spelling.

    A plain string is an ssh destination that inherits every transport option
    from the invoking user's ssh config — the historical form, kept byte for
    byte so legacy digests do not move. A mapping narrows those options for
    an ephemeral machine whose port and identity are known per run rather
    than in the config file.
    """
    if isinstance(value, str):
        if (not value.strip() or any(character.isspace() for character in value)
                or value.startswith("-") or value.count("@") > 1):
            raise DeploymentError(f"{label} must be an ssh destination (user@host)")
        return value
    if not isinstance(value, dict) or set(value) - SSH_FIELDS or "host" not in value:
        raise DeploymentError(
            f"{label} requires at most {sorted(SSH_FIELDS)} and a host")
    return _ssh_spec(value, label)


def _growth_key(entry):
    """The identity used for duplicate detection across both entry forms."""
    return destination(entry) if isinstance(entry, dict) else entry


def _gateway_shards(value, control_port, *, service):
    """Normalize manager.gateway_shards; absent keeps the legacy single gateway."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise DeploymentError("manager.gateway_shards must be a nonempty list")
    shards, names, ports = [], set(), set()
    for entry in value:
        if not isinstance(entry, dict) or set(entry) - SHARD_FIELDS or "name" not in entry:
            raise DeploymentError(
                f"manager.gateway_shards entries require at most {sorted(SHARD_FIELDS)} and a name")
        name = entry["name"]
        if not isinstance(name, str) or not SHARD_NAME.fullmatch(name):
            raise DeploymentError("manager.gateway_shards names are lowercase slugs")
        mode = entry.get("mode", "local")
        if mode not in SHARD_MODES:
            raise DeploymentError(f"manager.gateway_shards mode must be one of {SHARD_MODES}")
        shard = {"name": name}
        if mode == "remote":
            if not service:
                raise DeploymentError(
                    "remote gateway shards require a service-mode manager")
            if "port" in entry:
                raise DeploymentError("a remote gateway shard takes its port from url")
            shard["mode"] = "remote"
            # A remote shard may bind on a hub loopback when reached through a
            # reverse tunnel: the forward onto 127.0.0.1:N is the real route.
            shard["url"] = _plain_origin(entry.get("url"), "manager.gateway_shards url",
                                         allow_loopback=True)
            shard["admin_url"] = _plain_origin(entry.get("admin_url"),
                                               "manager.gateway_shards admin_url",
                                               allow_loopback=True)
        else:
            if "url" in entry or "admin_url" in entry:
                raise DeploymentError("a local gateway shard binds a port and derives its url")
            port = entry.get("port")
            if port is not None:
                port = _positive_int(port, "manager.gateway_shards port")
                if port == control_port:
                    raise DeploymentError("manager.gateway_shards port collides with control_port")
                if port in ports:
                    raise DeploymentError(f"duplicate gateway shard port: {port}")
                ports.add(port)
                shard["port"] = port
        if name in names:
            raise DeploymentError(f"duplicate gateway shard name: {name}")
        names.add(name)
        shards.append(shard)
    return shards


def _manager(config):
    if not isinstance(config, dict) or set(config) - MANAGER_FIELDS:
        raise DeploymentError(f"manager supports only {sorted(MANAGER_FIELDS)}")
    missing = MANAGER_REQUIRED - set(config)
    if missing:
        raise DeploymentError(f"manager requires {sorted(missing)}")
    mode = config["mode"]
    if mode not in ("embedded", "service"):
        raise DeploymentError("manager.mode must be embedded or service")
    service = mode == "service"
    control_port = _positive_int(config["control_port"], "manager.control_port")
    gateway_port = _positive_int(config["gateway_port"], "manager.gateway_port")
    if control_port == gateway_port:
        raise DeploymentError("manager.control_port and manager.gateway_port must differ")
    if not 0 < control_port < 65536 or not 0 < gateway_port < 65536:
        raise DeploymentError("manager ports must be in 1..65535")
    control_url = _plain_origin(config["control_url"], "manager.control_url",
                                port=control_port, allow_loopback=not service)
    gateway_url = _plain_origin(config["gateway_url"], "manager.gateway_url",
                                port=gateway_port, allow_loopback=not service)
    routing = config.get("gateway_routing", "modulo")
    if routing not in GATEWAY_ROUTINGS:
        raise DeploymentError(f"manager.gateway_routing must be one of {GATEWAY_ROUTINGS}")
    shards = _gateway_shards(config.get("gateway_shards"), control_port, service=service)
    # A static table and elastic growth may coexist: declared shards exist
    # from startup (a long-lived host served through a tunnel), and auto
    # routing grows beyond them when concurrency passes the per-shard budget.
    sandboxes_per_shard = config.get("sandboxes_per_shard", DEFAULT_SANDBOXES_PER_SHARD)
    if type(sandboxes_per_shard) is not int or sandboxes_per_shard < 1:
        raise DeploymentError(
            "manager.sandboxes_per_shard must be a positive integer (default "
            f"{DEFAULT_SANDBOXES_PER_SHARD} sandboxes per gateway)")
    worker_shards = config.get("worker_shards")
    if worker_shards is not None:
        if shards is None:
            raise DeploymentError("manager.worker_shards requires manager.gateway_shards")
        if (not isinstance(worker_shards, dict)
                or not all(isinstance(key, str) and key.strip()
                           and isinstance(value, str) and value.strip()
                           for key, value in worker_shards.items())):
            raise DeploymentError(
                "manager.worker_shards maps worker ids (or 'local') to gateway shard names")
    gateway_hosts = config.get("gateway_hosts")
    if gateway_hosts is not None:
        if not service:
            raise DeploymentError(
                "manager.gateway_hosts grows remote shards on a service-mode manager")
        if not isinstance(gateway_hosts, list) or not gateway_hosts:
            raise DeploymentError(
                "manager.gateway_hosts is a nonempty list of ssh destinations (user@host)")
        normalized_hosts = []
        for index, entry in enumerate(gateway_hosts):
            if (isinstance(entry, str) and entry.strip()
                    and not any(character.isspace() for character in entry)):
                normalized_hosts.append(entry)
            elif isinstance(entry, dict):
                normalized_hosts.append(
                    _growth_host(entry, f"manager.gateway_hosts[{index}]"))
            else:
                # Historical wording: a malformed string entry is still a
                # missing destination, not a new option form.
                raise DeploymentError(
                    "manager.gateway_hosts is a nonempty list of ssh destinations (user@host)")
        gateway_hosts = normalized_hosts
        if len({_growth_key(entry) for entry in gateway_hosts}) != len(gateway_hosts):
            raise DeploymentError("manager.gateway_hosts entries must be unique")
        if config.get("gateway_routing") in ("modulo", "worker_affinity"):
            # An undeclared routing keeps the runtime's elastic auto default;
            # only an explicit static routing refuses the growth pool.
            raise DeploymentError(
                "manager.gateway_hosts grows gateways under auto routing")
    shard_python = config.get("shard_python")
    if shard_python is not None:
        if gateway_hosts is None:
            raise DeploymentError(
                "manager.shard_python interprets gateway_hosts growth; "
                "declare the pool first")
        shard_python = _absolute_path(shard_python, "manager.shard_python")
    manager = {
        "id": _slug(config["id"], "manager.id"), "mode": mode,
        "bind": _nonempty_str(config["bind"], "manager.bind"),
        "control_port": control_port, "control_url": control_url,
        "gateway_port": gateway_port, "gateway_url": gateway_url,
        "run_token_file": _absolute_path(config["run_token_file"], "manager.run_token_file"),
        "source_dir": _absolute_path(config["source_dir"], "manager.source_dir"),
        "controller_python": _absolute_path(config["controller_python"], "manager.controller_python"),
    }
    # Shard keys appear only when sharding is declared, so legacy deployments
    # keep their exact historical shape (and digest). An explicitly declared
    # routing without shards opts in or out of auto growth; undeclared means
    # the runtime's elastic default (one gateway until sandboxes_per_shard).
    if shards is not None:
        manager["gateway_shards"] = shards
        # A declared table pins the historical static default, unless the
        # growth pool asks auto routing to expand beyond it.
        manager["gateway_routing"] = (
            "auto" if gateway_hosts is not None and "gateway_routing" not in config
            else routing)
        if worker_shards is not None:
            manager["worker_shards"] = dict(worker_shards)
    elif gateway_hosts is not None:
        # The pool is meaningless without growth; never let the runtime see a
        # static routing that it would reject after the GPU job is submitted.
        manager["gateway_routing"] = "auto"
    elif "gateway_routing" in config:
        manager["gateway_routing"] = routing
    if "sandboxes_per_shard" in config:
        manager["sandboxes_per_shard"] = sandboxes_per_shard
    if gateway_hosts is not None:
        # Auto growth fills this pool before falling back to the manager
        # machine; the entry spelling is the URL host runners will use.
        manager["gateway_hosts"] = list(gateway_hosts)
        if shard_python is not None:
            manager["shard_python"] = shard_python
    return manager


def _command(value, label):
    """A provision command: an argv list, or a string split without a shell."""
    if isinstance(value, str):
        parts = shlex.split(value)
        if not parts:
            raise DeploymentError(f"{label} must be a nonempty command")
        return parts
    if (not isinstance(value, list) or not value
            or not all(isinstance(part, str) and part for part in value)):
        raise DeploymentError(f"{label} must be a nonempty argv list of strings")
    return list(value)


def _provision(value):
    """Normalize gpu_target.provision: how a dispatch obtains its GPU machine.

    The commands are opaque to the core: ``up`` prints one JSON object naming
    the machine it produced, ``down`` releases it. Nothing site-specific is
    interpreted here, and no secret belongs in either command.
    """
    if (not isinstance(value, dict) or set(value) - PROVISION_FIELDS
            or "up" not in value):
        raise DeploymentError(
            f"gpu_target.provision supports {sorted(PROVISION_FIELDS)} and requires up")
    provision = {"up": _command(value["up"], "gpu_target.provision.up")}
    down = value.get("down")
    if down is not None:
        provision["down"] = _command(down, "gpu_target.provision.down")
    provision["timeout_s"] = _positive_int(
        value.get("timeout_s", DEFAULT_PROVISION_TIMEOUT_S),
        "gpu_target.provision.timeout_s")
    return provision


def _gpu_target(config):
    if (not isinstance(config, dict) or set(config) - GPU_FIELDS
            or GPU_REQUIRED - set(config)):
        raise DeploymentError(
            f"gpu_target requires {sorted(GPU_REQUIRED)} and supports at most "
            f"{sorted(GPU_FIELDS)}")
    transport = config["transport"]
    if transport not in TRANSPORTS:
        raise DeploymentError(f"gpu_target.transport must be one of {TRANSPORTS}")
    provision = config.get("provision")
    if provision is not None:
        # A provisioned machine is reached over ssh; a local target already
        # exists and has nothing to obtain or release.
        if transport != "ssh":
            raise DeploymentError("gpu_target.provision requires ssh transport")
        if config["ssh"] is not None:
            raise DeploymentError(
                "gpu_target.provision supplies the machine; declare no gpu_target.ssh")
        provision = _provision(provision)
    ssh = config["ssh"]
    if transport == "ssh":
        if ssh is None and provision is None:
            raise DeploymentError(
                "gpu_target.ssh is required unless gpu_target.provision supplies the machine")
        if ssh is not None:
            ssh = _ssh_spec(ssh, "gpu_target.ssh")
    elif ssh is not None:
        raise DeploymentError("gpu_target.ssh must be null for local transport")
    target = {
        "id": _slug(config["id"], "gpu_target.id"), "transport": transport, "ssh": ssh,
        "source_dir": _absolute_path(config["source_dir"], "gpu_target.source_dir"),
        "train_python": _absolute_path(config["train_python"], "gpu_target.train_python"),
        "expected_gpus": _positive_int(config["expected_gpus"], "gpu_target.expected_gpus"),
    }
    environment_file = config["environment_file"]
    if environment_file is not None:
        target["environment_file"] = _absolute_path(environment_file, "gpu_target.environment_file")
    if provision is not None:
        target["provision"] = provision
    return target


def _storage(config):
    if not isinstance(config, dict) or set(config) != STORAGE_FIELDS:
        raise DeploymentError(f"storage requires exactly {sorted(STORAGE_FIELDS)}")
    if config["mode"] not in STORAGE_MODES:
        raise DeploymentError(f"storage.mode must be one of {STORAGE_MODES}")
    return {
        "mode": config["mode"],
        "run_root": _absolute_path(config["run_root"], "storage.run_root"),
        "manager_writable": _bool(config["manager_writable"], "storage.manager_writable"),
        "gpu_writable": _bool(config["gpu_writable"], "storage.gpu_writable"),
        "worker_writable": _bool(config["worker_writable"], "storage.worker_writable"),
    }


def _rollout(config):
    if not isinstance(config, dict) or set(config) != ROLLOUT_FIELDS:
        raise DeploymentError(f"rollout requires exactly {sorted(ROLLOUT_FIELDS)}")
    if config["provider"] not in ROLLOUT_PROVIDERS:
        raise DeploymentError(f"rollout.provider must be one of {ROLLOUT_PROVIDERS}")
    workers = config["workers"]
    # ssh_docker needs explicit machines; the embedded provider falls back to
    # the training config's worker list, so it may start empty.
    if (not isinstance(workers, list)
            or (not workers and config["provider"] not in ("embedded", "sandbox"))):
        raise DeploymentError("rollout.workers must be a nonempty list")
    normalized, seen = [], set()
    for worker in workers:
        if not isinstance(worker, dict) or set(worker) != WORKER_FIELDS:
            raise DeploymentError(f"rollout worker requires exactly {sorted(WORKER_FIELDS)}")
        worker_id = _slug(worker["id"], "rollout worker id")
        if worker_id in seen:
            raise DeploymentError(f"duplicate rollout worker id: {worker_id}")
        seen.add(worker_id)
        task_roots = worker["task_roots"]
        if (not isinstance(task_roots, list) or not task_roots
                or any(not isinstance(root, str) or not root.strip() for root in task_roots)):
            raise DeploymentError(f"worker {worker_id} requires nonempty task_roots")
        normalized.append({
            "id": worker_id,
            "ssh": _ssh_spec(worker["ssh"], f"rollout worker {worker_id} ssh"),
            "source_dir": _absolute_path(worker["source_dir"], f"worker {worker_id} source_dir"),
            "runner_python": _absolute_path(worker["runner_python"], f"worker {worker_id} runner_python"),
            "docker": _bool(worker["docker"], f"worker {worker_id} docker"),
            "task_roots": [_absolute_path(root, f"worker {worker_id} task_root") for root in task_roots],
        })
    return {"provider": config["provider"], "workers": normalized}


def _connectivity(config):
    if not isinstance(config, dict) or set(config) != CONNECTIVITY_FIELDS:
        raise DeploymentError(f"connectivity requires exactly {sorted(CONNECTIVITY_FIELDS)}")
    return {
        "command_timeout_s": _positive_int(config["command_timeout_s"], "connectivity.command_timeout_s"),
        "ssh_timeout_s": _positive_int(config["ssh_timeout_s"], "connectivity.ssh_timeout_s"),
        "http_timeout_s": _positive_int(config["http_timeout_s"], "connectivity.http_timeout_s"),
        "gpu_to_controller": _bool(config["gpu_to_controller"], "connectivity.gpu_to_controller"),
        "worker_to_gateway": _bool(config["worker_to_gateway"], "connectivity.worker_to_gateway"),
        "manager_to_gpu_inference": _bool(config["manager_to_gpu_inference"], "connectivity.manager_to_gpu_inference"),
    }


def _check_gateway_shard_routing(manager, raw_manager, rollout):
    """Static routing must be total before dispatch, not discovered mid-run."""
    declared_routing = raw_manager.get("gateway_routing", "modulo")
    shards = manager.get("gateway_shards")
    worker_shards = manager.get("worker_shards") or {}
    if declared_routing == "worker_affinity" and not shards:
        raise DeploymentError("worker_affinity routing requires manager.gateway_shards")
    worker_ids = {worker["id"] for worker in rollout["workers"]}
    if declared_routing == "worker_affinity":
        missing = sorted(worker_ids - set(worker_shards))
        if missing:
            raise DeploymentError(
                f"worker_affinity requires a shard for every rollout worker; missing {missing}")
    if not worker_shards:
        return
    unknown_keys = sorted(set(worker_shards) - worker_ids - {"local"})
    if unknown_keys:
        raise DeploymentError(f"manager.worker_shards names unknown workers {unknown_keys}")
    unknown_shards = sorted(set(worker_shards.values()) - {shard["name"] for shard in shards})
    if unknown_shards:
        raise DeploymentError(f"manager.worker_shards names unknown shards {unknown_shards}")


def validate(config):
    """Strictly validate and normalize a deployment mapping."""
    if not isinstance(config, dict):
        raise DeploymentError("deployment config must be a mapping")
    if set(config) != DEPLOYMENT_SECTIONS | {"schema_version"}:
        raise DeploymentError(
            "deployment config requires exactly schema_version and "
            + ", ".join(sorted(DEPLOYMENT_SECTIONS)))
    if config["schema_version"] != DEPLOYMENT_SCHEMA_VERSION or type(config["schema_version"]) is not int:
        raise DeploymentError(f"deployment schema_version must be {DEPLOYMENT_SCHEMA_VERSION}")
    manager = _manager(config["manager"])
    rollout = _rollout(config["rollout"])
    _check_gateway_shard_routing(manager, config["manager"], rollout)
    return {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "manager": manager,
        "gpu_target": _gpu_target(config["gpu_target"]),
        "storage": _storage(config["storage"]),
        "rollout": rollout,
        "connectivity": _connectivity(config["connectivity"]),
    }


def embedded_local(*, run_root=None, workers=(), expected_gpus=1):
    """Default same-machine deployment; behavior matches the legacy pipeline."""
    return validate({
        "schema_version": 1,
        "manager": {"id": "embedded-local", "mode": "embedded", "bind": "127.0.0.1",
                    "control_port": 49200, "control_url": "http://127.0.0.1:49200",
                    "gateway_port": 49184, "gateway_url": "http://127.0.0.1:49184",
                    "run_token_file": str(ROOT / "runs" / "run-token"),
                    "source_dir": str(ROOT), "controller_python": sys.executable},
        "gpu_target": {"id": "local", "transport": "local", "ssh": None,
                       "source_dir": str(ROOT), "train_python": sys.executable,
                       "environment_file": None, "expected_gpus": expected_gpus},
        "storage": {"mode": "shared_fs", "run_root": run_root or str(ROOT / "runs"),
                    "manager_writable": True, "gpu_writable": True, "worker_writable": True},
        "rollout": {"provider": "embedded", "workers": [
            {"id": f"worker-{index}", "ssh": {"host": worker}, "source_dir": str(ROOT),
             "runner_python": sys.executable, "docker": True, "task_roots": [str(ROOT)]}
            for index, worker in enumerate(workers)]},
        "connectivity": {"command_timeout_s": 30, "ssh_timeout_s": 15, "http_timeout_s": 10,
                         "gpu_to_controller": True, "worker_to_gateway": True,
                         "manager_to_gpu_inference": True},
    })


DEFAULT_FILENAME = "deployment.yaml"


def load_deployment(path=None, *, default_directory="."):
    """Load an explicit, default-discovered or embedded-local deployment config.

    Returns ``(config, source)`` where source describes where the config came
    from; absent any file the embedded-local topology keeps legacy behavior.
    """
    if path is not None:
        source = str(Path(path).expanduser())
        candidates = [source]
    else:
        candidates = [str(Path(default_directory) / DEFAULT_FILENAME), DEFAULT_FILENAME]
        source = candidates[0]
    for candidate in candidates:
        file = Path(candidate)
        if not file.is_file():
            continue
        try:
            text = file.read_text()
        except OSError as exc:
            raise DeploymentError(f"cannot read deployment config {candidate}: {exc}") from exc
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise DeploymentError(f"deployment YAML {candidate}: {exc}") from exc
        return validate(raw), str(file.resolve())
    if path is not None:
        raise DeploymentError(f"deployment config not found: {path}")
    return embedded_local(), "embedded-local"


def summary(config, source):
    """Plan-embedded deployment digest; the full config travels with it."""
    from harborrl.trajectories.native import digest
    return {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "source": source,
        "digest": digest(config),
        "mode": config["manager"]["mode"],
        "manager_id": config["manager"]["id"],
        "control_url": config["manager"]["control_url"],
        "gateway_url": config["manager"]["gateway_url"],
        "gpu": {"id": config["gpu_target"]["id"], "transport": config["gpu_target"]["transport"]},
        "worker_ids": [worker["id"] for worker in config["rollout"]["workers"]],
        "run_root": config["storage"]["run_root"],
        "config": config,
    }
