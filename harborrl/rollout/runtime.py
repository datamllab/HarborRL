"""Extracted rollout core (P2): gateway, traces and complete groups.

This module owns rollout execution and must not import Slime or Ray: training
frameworks live in ``harborrl.backends.*`` adapters that build a
``RolloutRuntimeConfig`` and hand over a tokenizer. Group semantics, worker
scheduling, retries and the RL_READY contract stay exactly as the embedded
pipeline enforced them.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import json
import re
import secrets
import shlex
import threading
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from harborrl.models.qwen import normalize_chat_template_ids
from harborrl.rollout.harbor_job.coordinator import GroupSlots, run_attempt
from harborrl.rollout.remote_shard import RemoteShard, read_shard_token
from harborrl.trajectories.native import (
    Identity,
    RewardProfile,
    digest,
    encode,
    publish,
    read_json,
)

TASK_ENTRY_KEYS = {"id", "revision", "path", "task_digest", "reward_profile"}
SHARD_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
GATEWAY_ROUTINGS = ("modulo", "worker_affinity", "auto")
DEFAULT_SANDBOXES_PER_SHARD = 30
LOCAL_RUNNER_KEY = "local"
SHARD_SERVICE_MODULE = "harborrl.gateway.shard_service"
# One remote shard must report its ready line within this budget: ssh connect
# plus tokenizer load on the remote host. A candidate that misses it is
# abandoned so growth can try the next host instead of hanging a slot.
REMOTE_SHARD_START_TIMEOUT = 120.0


@dataclass(frozen=True)
class GatewayShardSpec:
    """One gateway's static placement: name, exclusive port, advertised URL.

    ``mode`` is "local" (this process binds the port) or "remote" (an
    independent shard process serves ``url``; the manager reaches its
    registry through ``admin_url``).
    """

    name: str
    port: int
    url: str
    mode: str = "local"
    admin_url: str = None


def _port_url(url, port):
    """Rewrite only the port of a plain origin."""
    parts = urlsplit(url)
    host = f"[{parts.hostname}]" if ":" in (parts.hostname or "") else parts.hostname
    return f"{parts.scheme}://{host}:{port}"


GROWTH_HOST_FIELDS = {"host", "user", "port", "identity"}


def _growth_host(entry):
    """Normalize one growth-pool entry into (destination, port, identity, url_host).

    A plain string is an ssh destination (``user@host``) that inherits every
    transport option from the invoking user's SSH config. An object narrows
    those options for this host only — an ephemeral machine whose key and
    port are known per run, not in ``~/.ssh/config``. The url host is the
    bare host: the SSH port is the transport's, never the shard's gateway
    port, which the shard process binds and reports on its ready line.
    """
    if isinstance(entry, str):
        if not entry.strip() or any(character.isspace() for character in entry):
            raise ValueError('gateway_hosts lists ssh destinations (user@host)')
        return entry, None, None, entry.split("@")[-1]
    if not isinstance(entry, dict) or set(entry) - GROWTH_HOST_FIELDS:
        raise ValueError(
            f'gateway_hosts entries are ssh destinations or {sorted(GROWTH_HOST_FIELDS)}')
    host = entry.get("host")
    if (not isinstance(host, str) or not host.strip()
            or any(character.isspace() for character in host)
            or host.startswith("-") or host.count("@") > 1):
        raise ValueError('gateway_hosts entries require an ssh destination host')
    user, port, identity = entry.get("user"), entry.get("port"), entry.get("identity")
    if user is not None and (not isinstance(user, str) or not user.strip()
                             or user.startswith("-")):
        raise ValueError('gateway_hosts entry user must not look like a flag')
    if port is not None and (type(port) is not int or not 0 < port < 65536):
        raise ValueError('gateway_hosts entry port must be an integer in 1..65535')
    if identity is not None and (not isinstance(identity, str) or not identity.strip()):
        raise ValueError('gateway_hosts entry identity must be a nonempty path')
    destination = f"{user}@{host}" if user else host
    return destination, port, identity, host.split("@")[-1]


def _plain_origin(url, label):
    parts = urlsplit(url if isinstance(url, str) else "")
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in ("", "/")):
        raise ValueError(f"{label} must be a plain http(s) origin")
    return f"{parts.scheme}://{parts.netloc}".rstrip("/")


class RuntimeLifecycleError(RuntimeError):
    """Embedded rollout loop/thread-affinity or lifecycle misuse."""


class CleanupRequiredError(RuntimeError):
    """Harbor resources must be inspected before admitting more attempts."""


class CleanupAdmission:
    def __init__(self, root):
        self.markers = set(Path(root).rglob("cleanup-required.json"))

    def observe(self, root):
        self.markers.update(Path(root).rglob("cleanup-required.json"))

    def check(self):
        self.markers = {path for path in self.markers if path.exists()}
        if self.markers:
            paths = ", ".join(str(path) for path in sorted(self.markers))
            raise CleanupRequiredError(
                "rollout cleanup unresolved; automatic retry blocked. "
                "Inspect Harbor resources, then remove resolved cleanup-required.json "
                f"markers before resuming: {paths}")


@dataclass(frozen=True)
class RolloutRuntimeConfig:
    """Deployment-independent description of one run's rollout machinery."""

    run_id: str
    run_root: str
    catalog_path: str
    profile_path: str
    gateway_host: str
    gateway_port: int
    gateway_url: str
    runner_python: str
    workers: tuple
    task_worker_overrides: dict
    max_attempts: int
    max_output_tokens: int
    max_context: int
    concurrency_limit: int
    group_size: int
    expected_tokenizer_digest: str
    expected_template_digest: str
    audited_raw_logprobs: bool
    drain_timeout: float
    model_tag: str
    endpoint: str
    sandbox: dict = None
    runner_transport: str = 'ssh'
    sampling_seed: int | None = None
    shards: tuple = ()
    gateway_routing: str = 'modulo'
    sandboxes_per_shard: int = DEFAULT_SANDBOXES_PER_SHARD
    worker_shards: dict = field(default_factory=dict)
    shard_token_file: str = None
    gateway_hosts: tuple = ()
    hf_ckpt: str = None
    shard_python: str = None

    def __post_init__(self):
        if self.sampling_seed is not None and (
                type(self.sampling_seed) is not int or not 0 <= self.sampling_seed < 2**31):
            raise ValueError('sampling_seed must be null or an integer in 0..2147483647')
        if self.gateway_routing not in GATEWAY_ROUTINGS:
            raise ValueError(f'gateway_routing must be one of {GATEWAY_ROUTINGS}')
        if (type(self.sandboxes_per_shard) is not int
                or self.sandboxes_per_shard < 1):
            raise ValueError('sandboxes_per_shard must be a positive integer')
        if self.gateway_routing == 'auto' and self.gateway_hosts:
            # Growth needs the shared run token and a tokenizer checkpoint the
            # spawned process loads itself.
            if not self.shard_token_file:
                raise ValueError(
                    'gateway_hosts requires HARBORRL_NATIVE_SHARD_TOKEN_FILE')
            if not self.hf_ckpt:
                raise ValueError(
                    'gateway_hosts requires HF_CKPT so remote shards load the '
                    'run tokenizer')
        if self.gateway_hosts and self.gateway_routing != 'auto':
            # Remote growth is an auto-routing concern: static routing has no
            # mid-run growth for the pool to serve.
            raise ValueError('gateway_hosts grows gateways under auto routing only')
        if not isinstance(self.gateway_hosts, tuple):
            raise ValueError('gateway_hosts lists ssh destinations (user@host)')
        # Validated here so a malformed pool fails at startup; the runtime
        # normalizes and rotates the same entries when it grows a shard.
        for entry in self.gateway_hosts:
            _growth_host(entry)
        if (not isinstance(self.worker_shards, dict)
                or not all(isinstance(key, str) and isinstance(value, str)
                           for key, value in self.worker_shards.items())):
            raise ValueError('worker_shards maps runner destinations to gateway shard names')
        if self.worker_shards and self.gateway_routing == 'auto':
            # auto leases from the live shard set and never consults the
            # affinity map, so accepting one here would silently route by
            # lease order while the operator believes affinity is in force.
            raise ValueError(
                'worker_shards is static affinity routing; auto routing ignores it. '
                'Declare gateway_routing: worker_affinity, or drop the mapping')
        if self.shards:
            names = [spec.name for spec in self.shards]
            ports = [spec.port for spec in self.shards]
            if any(not SHARD_NAME.fullmatch(name) for name in names):
                raise ValueError('gateway shard names are lowercase slugs')
            if len(set(names)) != len(names):
                raise ValueError('gateway shard names must be unique')
            if (any(type(port) is not int or not 0 < port < 65536 for port in ports)
                    or len(set(ports)) != len(ports)):
                raise ValueError('gateway shard ports must be unique integers in 1..65535')
            if any(spec.mode not in ('local', 'remote') for spec in self.shards):
                raise ValueError('gateway shard mode is local or remote')
            if any(spec.mode == 'remote' and not spec.admin_url for spec in self.shards):
                raise ValueError('a remote gateway shard requires its admin_url')
            if any(spec.mode == 'remote' for spec in self.shards) and not self.shard_token_file:
                raise ValueError('remote gateway shards require HARBORRL_NATIVE_SHARD_TOKEN_FILE')
            if self.gateway_routing == 'worker_affinity':
                # Static routing must be total before dispatch, never
                # discovered mid-run: every runner names exactly one shard.
                required = set(self.workers)
                if self.runner_transport == 'local':
                    required.add(LOCAL_RUNNER_KEY)
                allowed = set(self.workers) | {LOCAL_RUNNER_KEY}
                unknown = sorted(set(self.worker_shards) - allowed)
                if unknown:
                    raise ValueError(f'worker_shards names unknown runners {unknown}')
                missing = sorted(required - set(self.worker_shards))
                if missing:
                    raise ValueError('worker_affinity routes every runner to a declared '
                                     f'gateway shard; missing {missing}')
                unsharded = sorted(set(self.worker_shards.values()) - set(names))
                if unsharded:
                    raise ValueError(f'worker_shards names unknown gateway shards {unsharded}')

    @staticmethod
    def _required(env, name):
        value = env.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"rollout runtime requires {name}")
        return value

    @classmethod
    def from_environment(cls, env, *, endpoint, concurrency_limit, group_size,
                         gateway_host=None, gateway_port=None, gateway_url=None,
                         workers=None, runner_python=None):
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rollout runtime requires a serving endpoint")
        if type(group_size) is not int or group_size < 2:
            raise ValueError("rollout runtime requires a group size of at least two")
        if workers is None:
            workers = json.loads(cls._required(env, "HARBORRL_NATIVE_RUNNER_WORKERS"))
        if runner_python is None:
            runner_python = cls._required(env, "HARBORRL_NATIVE_RUNNER_PYTHON")
        try:
            overrides = json.loads(env.get("HARBORRL_NATIVE_TASK_WORKER_OVERRIDES") or "{}")
        except ValueError as exc:
            raise ValueError("invalid HARBORRL_NATIVE_TASK_WORKER_OVERRIDES") from exc
        try:
            worker_shards = json.loads(env.get("HARBORRL_NATIVE_GATEWAY_WORKER_SHARDS") or "{}")
        except ValueError as exc:
            raise ValueError("invalid HARBORRL_NATIVE_GATEWAY_WORKER_SHARDS") from exc
        try:
            gateway_hosts = json.loads(env.get("HARBORRL_NATIVE_GATEWAY_HOSTS") or "[]")
        except ValueError as exc:
            raise ValueError("invalid HARBORRL_NATIVE_GATEWAY_HOSTS") from exc
        if (not isinstance(gateway_hosts, list)
                or not all(isinstance(host, (str, dict)) for host in gateway_hosts)):
            # Entries are ssh destinations or option mappings; __post_init__
            # validates each one's fields.
            raise ValueError(
                "HARBORRL_NATIVE_GATEWAY_HOSTS is a list of ssh destinations or host mappings")
        gateway_port = int(gateway_port or cls._required(env, "HARBORRL_NATIVE_GATEWAY_PORT"))
        gateway_url = gateway_url or cls._required(env, "HARBORRL_NATIVE_GATEWAY_URL")
        shards = cls._gateway_shards(env, gateway_port, gateway_url)
        try:
            sandboxes_per_shard = int(env.get(
                "HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD",
                str(DEFAULT_SANDBOXES_PER_SHARD)))
        except ValueError as exc:
            raise ValueError("invalid HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD") from exc
        # Without a static table the default is elastic: exactly one gateway
        # until the live sandbox count passes sandboxes_per_shard, then one
        # more. A declared table keeps the historical static default.
        declared_routing = env.get("HARBORRL_NATIVE_GATEWAY_ROUTING")
        declared_routing = declared_routing.strip() if isinstance(declared_routing, str) else ""
        return cls(
            run_id=cls._required(env, "RUN_ID"),
            run_root=cls._required(env, "RUN_DIR"),
            catalog_path=cls._required(env, "HARBORRL_NATIVE_CATALOG"),
            profile_path=cls._required(env, "HARBORRL_NATIVE_PROFILE"),
            gateway_host=gateway_host or env.get("HARBORRL_NATIVE_GATEWAY_HOST", "0.0.0.0"),
            gateway_port=gateway_port,
            gateway_url=gateway_url,
            shards=shards,
            gateway_routing=declared_routing or ("auto" if not shards else "modulo"),
            sandboxes_per_shard=sandboxes_per_shard,
            worker_shards=dict(worker_shards),
            shard_token_file=env.get("HARBORRL_NATIVE_SHARD_TOKEN_FILE") or None,
            gateway_hosts=tuple(gateway_hosts),
            hf_ckpt=env.get("HF_CKPT") or None,
            shard_python=env.get("HARBORRL_NATIVE_SHARD_PYTHON") or None,
            runner_python=runner_python,
            workers=tuple(workers),
            task_worker_overrides=dict(overrides),
            max_attempts=int(env.get("HARBORRL_NATIVE_MAX_ATTEMPTS", "2")),
            max_output_tokens=int(env.get("ROLLOUT_MAX_RESPONSE_LEN", "8192")),
            max_context=int(env.get("ROLLOUT_MAX_CONTEXT_LEN", "32768")),
            concurrency_limit=max(1, int(concurrency_limit)),
            group_size=int(group_size),
            expected_tokenizer_digest=cls._required(env, "HARBORRL_NATIVE_TOKENIZER_DIGEST"),
            expected_template_digest=cls._required(env, "HARBORRL_NATIVE_TEMPLATE_DIGEST"),
            audited_raw_logprobs=env.get("HARBORRL_NATIVE_AUDITED_LOGPROBS") == "1",
            drain_timeout=float(env.get("HARBORRL_NATIVE_DRAIN_TIMEOUT", "30")),
            model_tag=env.get("MODEL_TAG", env.get("MODEL_ARGS_FILE", "qwen3-8B")),
            endpoint=endpoint.rstrip("/"),
            sandbox=json.loads(env.get('HARBORRL_NATIVE_SANDBOX', 'null')),
            runner_transport=env.get('HARBORRL_NATIVE_RUNNER_TRANSPORT', 'ssh'),
            sampling_seed=(int(env['HARBORRL_TRAINING_SEED'])
                           if 'HARBORRL_TRAINING_SEED' in env else None),
        )

    @staticmethod
    def _gateway_shards(env, gateway_port, gateway_url):
        """Parse the static shard table; an absent table means one elastic gateway.

        Undeclared local shard ports default to consecutive slots after the
        declared gateway port; every local shard URL is the gateway URL with
        its port rewritten, so shard placement stays one spelling everywhere.
        A remote shard names the origins of its independent process instead.
        An absent table routes "auto": the implicit first gateway serves until
        the live sandbox count passes sandboxes_per_shard, then the runtime
        spawns the next local shard itself.
        """
        raw = env.get("HARBORRL_NATIVE_GATEWAY_SHARDS")
        if raw is None or not raw.strip():
            return ()
        try:
            entries = json.loads(raw)
        except ValueError as exc:
            raise ValueError("invalid HARBORRL_NATIVE_GATEWAY_SHARDS") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("HARBORRL_NATIVE_GATEWAY_SHARDS must be a nonempty shard list")
        specs = []
        for index, entry in enumerate(entries):
            if (not isinstance(entry, dict)
                    or set(entry) - {"name", "port", "mode", "url", "admin_url"}
                    or not isinstance(entry.get("name"), str)):
                raise ValueError("each gateway shard is {name, port?, mode?, url?, admin_url?}")
            mode = entry.get("mode", "local")
            if mode not in ("local", "remote"):
                raise ValueError("gateway shard mode is local or remote")
            if mode == "remote":
                if "port" in entry:
                    raise ValueError("a remote gateway shard takes its port from url")
                url = _plain_origin(entry.get("url"), "remote gateway shard url")
                admin_url = _plain_origin(entry.get("admin_url"),
                                          "remote gateway shard admin_url")
                specs.append(GatewayShardSpec(name=entry["name"], port=urlsplit(url).port,
                                              url=url, mode="remote", admin_url=admin_url))
                continue
            if "url" in entry or "admin_url" in entry:
                raise ValueError("a local gateway shard binds a port and derives its url")
            port = entry.get("port")
            port = gateway_port + index if port is None else int(port)
            specs.append(GatewayShardSpec(name=entry["name"], port=port,
                                          url=_port_url(gateway_url, port)))
        return tuple(specs)

    def environment(self):
        """Render the environment mapping this config was built from."""
        return {
            "RUN_ID": self.run_id, "RUN_DIR": self.run_root,
            "HARBORRL_NATIVE_CATALOG": self.catalog_path,
            "HARBORRL_NATIVE_PROFILE": self.profile_path,
            "HARBORRL_NATIVE_GATEWAY_HOST": self.gateway_host,
            "HARBORRL_NATIVE_GATEWAY_PORT": str(self.gateway_port),
            "HARBORRL_NATIVE_GATEWAY_URL": self.gateway_url,
            "HARBORRL_NATIVE_RUNNER_PYTHON": self.runner_python,
            "HARBORRL_NATIVE_RUNNER_WORKERS": json.dumps(list(self.workers)),
            "HARBORRL_NATIVE_TASK_WORKER_OVERRIDES": json.dumps(
                self.task_worker_overrides, sort_keys=True),
            "HARBORRL_NATIVE_MAX_ATTEMPTS": str(self.max_attempts),
            "ROLLOUT_MAX_RESPONSE_LEN": str(self.max_output_tokens),
            "ROLLOUT_MAX_CONTEXT_LEN": str(self.max_context),
            "HARBORRL_NATIVE_TOKENIZER_DIGEST": self.expected_tokenizer_digest,
            "HARBORRL_NATIVE_TEMPLATE_DIGEST": self.expected_template_digest,
            "HARBORRL_NATIVE_AUDITED_LOGPROBS": str(int(self.audited_raw_logprobs)),
            "HARBORRL_NATIVE_DRAIN_TIMEOUT": str(self.drain_timeout),
            "MODEL_TAG": self.model_tag,
            "HARBORRL_NATIVE_SANDBOX": json.dumps(self.sandbox),
            "HARBORRL_NATIVE_RUNNER_TRANSPORT": self.runner_transport,
            "HARBORRL_NATIVE_GATEWAY_ROUTING": self.gateway_routing,
            "HARBORRL_NATIVE_GATEWAY_SANDBOXES_PER_SHARD": str(self.sandboxes_per_shard),
            "HARBORRL_NATIVE_GATEWAY_WORKER_SHARDS": json.dumps(
                self.worker_shards, sort_keys=True),
            **({'HARBORRL_NATIVE_GATEWAY_SHARDS': json.dumps([
                {'name': spec.name, 'port': spec.port} if spec.mode == 'local' else {
                    'name': spec.name, 'mode': spec.mode, 'url': spec.url,
                    'admin_url': spec.admin_url}
                for spec in self.shards])}
               if self.shards else {}),
            **({'HARBORRL_NATIVE_SHARD_TOKEN_FILE': self.shard_token_file}
               if self.shard_token_file else {}),
            **({'HARBORRL_NATIVE_GATEWAY_HOSTS': json.dumps(list(self.gateway_hosts))}
               if self.gateway_hosts else {}),
            **({'HF_CKPT': self.hf_ckpt} if self.hf_ckpt else {}),
            **({'HARBORRL_NATIVE_SHARD_PYTHON': self.shard_python}
               if self.shard_python else {}),
            **({'HARBORRL_TRAINING_SEED': str(self.sampling_seed)}
               if self.sampling_seed is not None else {}),
        }


def tokenizer_fingerprint(tokenizer):
    messages = [{"role": "user", "content": "HarborRL native tokenizer/template fingerprint"}]
    kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    rendered_ids = normalize_chat_template_ids(
        tokenizer.apply_chat_template(messages, **kwargs))
    vocabulary = tokenizer.get_vocab()
    tokenizer_digest = digest({
        "class": type(tokenizer).__name__,
        "vocab": sorted((str(key), int(value)) for key, value in vocabulary.items()),
    })
    template = digest({"messages": messages, "kwargs": kwargs, "token_ids": rendered_ids})
    return tokenizer_digest, template


def _shard_service_environment(config, name):
    """The environment one remotely spawned shard process starts under.

    Everything the shard service validates comes from the manager's own
    config, so a spawned shard is locked to the same run, digests and
    serving pool as the gateways that started locally. Both ports bind
    ephemeral — the OS owns exclusivity across every machine that might
    host a shard — and the manager reads the resolved ports from the
    process' ready line before advertising the shard anywhere.
    """
    return {
        "RUN_ID": config.run_id,
        "RUN_DIR": str(config.run_root),
        "HARBORRL_SHARD_NAME": name,
        "HARBORRL_SHARD_GATEWAY_HOST": "0.0.0.0",
        "HARBORRL_SHARD_GATEWAY_PORT": "0",
        "HARBORRL_SHARD_ADMIN_HOST": "0.0.0.0",
        "HARBORRL_SHARD_ADMIN_PORT": "0",
        "HARBORRL_SHARD_TOKEN_FILE": config.shard_token_file,
        "HARBORRL_SHARD_ENDPOINT": config.endpoint,
        "HARBORRL_NATIVE_PROFILE": config.profile_path,
        "MODEL_TAG": config.model_tag,
        "ROLLOUT_MAX_RESPONSE_LEN": str(config.max_output_tokens),
        "ROLLOUT_MAX_CONTEXT_LEN": str(config.max_context),
        "HARBORRL_NATIVE_TOKENIZER_DIGEST": config.expected_tokenizer_digest,
        "HARBORRL_NATIVE_TEMPLATE_DIGEST": config.expected_template_digest,
        "HARBORRL_NATIVE_AUDITED_LOGPROBS": str(int(config.audited_raw_logprobs)),
        "HF_CKPT": config.hf_ckpt,
    }


def _shard_service_command(host, environment, *, python, project, port=None, identity=None):
    """The ssh command that starts one shard process on a remote host.

    ssh forwards no environment, so the shard service's env contract crosses
    as explicit ``env`` assignments ahead of the interpreter; the same
    BatchMode/ClearAllForwardings options the runner transport uses keep the
    launch non-interactive and proxy-free. ``port`` and ``identity`` come
    from a growth-host mapping: an ephemeral machine often listens on a
    non-standard ssh port behind a key that is not in the invoking user's
    ssh config.
    """
    assignments = " ".join(f"{key}={shlex.quote(value)}"
                           for key, value in sorted(environment.items()))
    remote = (f"cd {shlex.quote(project)} && exec env {assignments} "
              f"{shlex.quote(python)} -m {SHARD_SERVICE_MODULE}")
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
            "-o", "StrictHostKeyChecking=accept-new"]
    if port is not None:
        argv += ["-p", str(port)]
    if identity is not None:
        argv += ["-i", str(identity)]
    argv += [host, remote]
    return argv


class GatewayShard:
    """One local gateway instance: exclusive port, trace registry and trace tree.

    A remote shard (a separate gateway process, Phase 2) presents the same
    surface — ``name``, ``url``, per-attempt admission, ``start``/``close`` —
    so scheduling and the version barrier never learn which form a shard
    takes. Within one process the sandbox admission budget stays shared:
    shards split admission and tracing, not the worker quota.
    """

    def __init__(self, spec, *, registry, gateway, server, trace_dir):
        self.spec = spec
        self.registry = registry
        self.gateway = gateway
        self.server = server
        self.trace_dir = Path(trace_dir)
        self.thread = threading.Thread(
            target=server.serve_forever, name=f"harborrl-gateway-{spec.name}", daemon=True)
        self._started = False

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
        self._started = True
        self.thread.start()

    def close(self):
        # shutdown() before serve_forever() starts would deadlock the thread.
        if self._started:
            self.server.shutdown()
            if self.thread.is_alive():
                self.thread.join(5)
        self.server.server_close()


class SpawnedRemoteShard(RemoteShard):
    """A remote shard this runtime launched, plus the ssh transport owning it.

    ``close`` asks the process to stop through the admin plane — it unwinds
    cleanly and flushes its traces — and only kills the ssh transport when it
    outlives a grace period. Killing first would hang up the remote
    mid-shutdown; the transport is nevertheless always reaped so the manager
    never leaks child processes.
    """

    SHUTDOWN_GRACE = 15.0

    def __init__(self, spec, *, token, trace_dir, process, timeout=30.0):
        super().__init__(spec, token=token, trace_dir=trace_dir, timeout=timeout)
        self.process = process

    def close(self):
        super().close()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # No loop left to reap on; the shutdown request went out.
        loop.create_task(self._reap())

    async def _reap(self):
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.process.wait(), self.SHUTDOWN_GRACE)
        if self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.kill()
            with suppress(Exception):
                await self.process.wait()


class ShardAdmission:
    """The one policy epoch shared by every gateway shard.

    ``drain`` closes admission on ALL shards before waiting on any of them,
    and ``publish_version`` opens the next epoch only after every shard
    drained. Per-attempt operations (register, begin, finish, seal) stay on
    the shard that owns the attempt; this facade is the only surface the
    rollout service and the version barrier ever touch. Shard disagreement
    (a partially completed activation) reads as a closed admission until the
    next activation republishes a consistent epoch.

    Auto routing grows the shard set mid-run: a runtime passes its live
    shard list here, and ``join`` attaches a freshly spawned shard onto the
    current epoch atomically so the facade never observes disagreement.
    """

    def __init__(self, shards):
        # Hold a caller's list by reference so mid-run growth stays visible;
        # any other iterable is copied into a private list.
        self._shards = shards if isinstance(shards, list) else list(shards)
        self._lock = threading.Lock()
        self._epoch = None  # (states, version) of the last published epoch

    @property
    def shards(self):
        with self._lock:
            return tuple(self._shards)

    @property
    def version(self):
        versions = {shard.registry.version for shard in self.shards}
        return versions.pop() if len(versions) == 1 else None

    @property
    def accepting(self):
        return all(shard.registry.accepting for shard in self.shards)

    def drain(self, timeout):
        deadline = time.monotonic() + timeout
        shards = self.shards
        for shard in shards:
            shard.registry.close_admission()
        for shard in shards:
            shard.registry.drain(max(0.0, deadline - time.monotonic()))

    def publish_version(self, states, expected):
        with self._lock:
            self._epoch = (states, str(expected))
        for shard in self.shards:
            shard.registry.publish_version(states, expected)

    def join(self, shard):
        """Attach a mid-run shard onto the current epoch, then admit it.

        A fresh registry starts (version None, accepting True); joining with
        that version would read as shard disagreement and fail-close group
        admission until the next activation. Replaying the last published
        epoch inside the lock keeps one version across every shard; before
        the first activation there is no epoch yet and every shard
        legitimately starts unversioned.

        The replay is the facade's own close-then-publish sequence scoped to
        the new shard: ``publish_version`` refuses to open a barrier that was
        never closed, and a fresh registry opens accepting, so its admission
        closes first exactly like a drain would.
        """
        with self._lock:
            epoch = self._epoch
            if epoch is not None:
                states, version = epoch
                shard.registry.close_admission()
                shard.registry.publish_version(states, version)
            self._shards.append(shard)

    def leave(self, shard):
        """Detach a retiring shard from the epoch under the same lock.

        The mirror of ``join``: the facade stops spanning the shard, so a
        concurrent drain or publish no longer waits on or versions a gateway
        that is closing. Retirement happens only after the shard's attempts
        all settled, so its registry state is already sealed history.
        """
        with self._lock:
            if shard in self._shards:
                self._shards.remove(shard)


class RolloutRuntime:
    """One rollout root and one or more local gateway shards per run."""

    def __init__(self, config, tokenizer, *, backend=None):
        from harborrl.gateway.server import Gateway, make_server
        from harborrl.gateway.sglang_backend import SGLangBackend
        from harborrl.gateway.trace import TraceRegistry
        from harborrl.math_registry import MATH_REGISTRY

        self.config = config
        self.run_root = Path(config.run_root).resolve()
        self.catalog = read_json(config.catalog_path)
        self.profile = read_json(config.profile_path)
        self.rollout_root = self.run_root / "native-rollouts"
        self.rollout_root.mkdir(parents=True, exist_ok=True)
        self._loop = None
        self.cleanup = CleanupAdmission(self.rollout_root)
        self.cleanup.check()
        self.semaphore = asyncio.Semaphore(config.concurrency_limit)
        if config.sandbox:
            from harborrl.rollout.harbor_job.sandbox_config import SandboxAdmission
            self.sandbox_admission = SandboxAdmission(config.sandbox)
        tokenizer_digest, template_digest = tokenizer_fingerprint(tokenizer)
        if (tokenizer_digest != config.expected_tokenizer_digest
                or template_digest != config.expected_template_digest):
            raise ValueError("live tokenizer/template digest differs from the native lock")
        model = MATH_REGISTRY.model_for_preset(config.model_tag)
        # One shared serving backend: shards split admission and tracing,
        # never the inference pool.
        self.backend = backend or SGLangBackend(
            config.endpoint, tokenizer, model.adapter,
            tokenizer_digest=tokenizer_digest, template_digest=template_digest,
            max_context=config.max_context,
            audited_raw_logprobs=config.audited_raw_logprobs,
            sampling_seed=config.sampling_seed,
        )
        specs = config.shards or (GatewayShardSpec(
            name="gateway", port=config.gateway_port, url=config.gateway_url),)
        trace_root = self.run_root / "native-traces"
        # An explicit shard table gives every shard its own immutable trace
        # subtree; the implicit first gateway keeps the historical layout.
        shards = []
        for spec in specs:
            trace_dir = trace_root / f"shard-{spec.name}" if config.shards else trace_root
            if spec.mode == "remote":
                # An independent process owns the registry and the gateway;
                # this side only holds the admin-channel proxy. Resolved
                # through the module attribute so deployments can substitute
                # the surface (tests pin it this way).
                from harborrl.rollout.remote_shard import RemoteShard
                shards.append(RemoteShard(spec, token=read_shard_token(config.shard_token_file),
                                          trace_dir=trace_dir))
                continue
            registry = TraceRegistry(trace_dir)
            gateway = Gateway(registry, self.backend, model=self.profile["model"],
                              max_output_tokens=config.max_output_tokens)
            # Each attempt holds at most one policy generation plus occasional
            # count_tokens calls; double the scheduler budget with a floor of 4.
            server = make_server(gateway, host=config.gateway_host, port=spec.port,
                                 max_inflight=max(4, 2 * config.concurrency_limit))
            shards.append(GatewayShard(spec, registry=registry, gateway=gateway,
                                       server=server, trace_dir=trace_dir))
        # The live shard set: auto routing appends spawned gateways here and
        # the admission facade holds the same list, so the one epoch always
        # spans every shard that exists.
        self._shards = shards
        # Auto routing occupancy lives on the event loop that dispatches
        # attempts; nothing outside the loop touches it.
        self._auto_inflight = {}
        # Auto routing names spawned gateways monotonically so a retired
        # name's trace subtree is never reused by a later regrowth. A static
        # table may coexist with the growth pool, so the counter also skips
        # every name the table already claims.
        self._static_names = {spec.name for spec in config.shards}
        self._shard_seq = 0
        self._auto_spawned = set()
        # Growth serializes under this lock: concurrent slots at capacity
        # must grow exactly once, and the fill re-check happens under it.
        self._lease_lock = asyncio.Lock()
        # Remote growth rotates through the declared pool so consecutive
        # spawns spread across machines instead of piling onto the first.
        self._host_seq = 0
        # Remotely spawned shards share the run token with statically
        # declared ones; loading it here fails loudly at startup instead of
        # on the first mid-run growth attempt.
        self._shard_token = (read_shard_token(config.shard_token_file)
                             if config.gateway_hosts else None)
        # The service-facing admission epoch spans every shard.
        self.registry = ShardAdmission(self._shards)
        if config.gateway_routing == "auto":
            # Growth legitimately advances the record, so auto routing keeps
            # it replaceable; static tables keep the immutable one-shot form.
            self._write_shards_evidence()
        else:
            publish(self.run_root / "gateway-shards.json", self._shards_payload())
        try:
            for shard in self._shards:
                shard.start()
        except BaseException:
            # Never leak local servers because a later shard failed to start.
            for shard in self._shards:
                with suppress(Exception):
                    shard.close()
            raise

    def check_event_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeLifecycleError("rollout runtime belongs to a different event loop")

    @property
    def shards(self):
        """The live shard set; auto routing appends spawned gateways mid-run."""
        return tuple(self._shards)

    def _shards_payload(self):
        return {
            "gateway_routing": self.config.gateway_routing,
            **({"sandboxes_per_shard": self.config.sandboxes_per_shard}
               if self.config.gateway_routing == "auto" else {}),
            "shards": [
                {"name": shard.name, "port": shard.port, "url": shard.url,
                 "mode": shard.spec.mode,
                 **({"admin_url": shard.spec.admin_url}
                    if shard.spec.mode == "remote" else {}),
                 "trace_dir": str(shard.trace_dir)}
                for shard in self.shards]}

    def _write_shards_evidence(self):
        """Rewrite the shard table evidence after auto routing grows it."""
        path = self.run_root / "gateway-shards.json"
        temp = self.run_root / "gateway-shards.tmp"
        temp.write_bytes(encode(self._shards_payload()) + b"\n")
        temp.replace(path)

    async def lease_shard(self):
        """Pick the gateway for the next running sandbox attempt.

        Auto routing keeps at most ``sandboxes_per_shard`` concurrent sandbox
        attempts per gateway: the (n+1)-th concurrent attempt grows the fleet
        instead of piling onto a full one — on a declared remote host first,
        locally only when no candidate answers. Fail-closed by construction:
        a spawn failure fails this attempt loudly, never a silent fallback
        onto an overloaded gateway.
        """
        for shard in self.shards:
            if self._auto_inflight.get(shard.name, 0) < self.config.sandboxes_per_shard:
                break
        else:
            async with self._lease_lock:
                # Slots that waited on the lock while a sibling spawned must
                # re-check: capacity may have appeared, and growth that
                # started under the lock must not repeat.
                for shard in self.shards:
                    if (self._auto_inflight.get(shard.name, 0)
                            < self.config.sandboxes_per_shard):
                        break
                else:
                    shard = await self._spawn_shard()
        self._auto_inflight[shard.name] = self._auto_inflight.get(shard.name, 0) + 1
        return shard

    def release_shard(self, shard):
        remaining = self._auto_inflight.get(shard.name, 0) - 1
        if remaining > 0:
            self._auto_inflight[shard.name] = remaining
        else:
            self._auto_inflight.pop(shard.name, None)

    def _candidate_hosts(self):
        """The declared growth pool, rotated so spawns spread across it.

        Returns normalized ``(destination, port, identity, url_host)``
        tuples; a plain-string entry inherits its transport options from the
        invoking user's ssh config.
        """
        hosts = self.config.gateway_hosts
        if not hosts:
            return ()
        rotation = self._host_seq % len(hosts)
        self._host_seq += 1
        rotated = hosts[rotation:] + hosts[:rotation]
        return tuple(_growth_host(entry) for entry in rotated)

    async def _spawn_shard(self):
        """Grow by one gateway: declared remote hosts first, local last."""
        # One growth consumes exactly one name: candidates that never
        # materialized leave no trace subtree, so they reuse this name
        # instead of burning the sequence. A declared static table may name
        # some gateways already, so the sequence skips those names.
        self._shard_seq += 1
        name = f"gw{self._shard_seq}"
        while name in self._static_names:
            self._shard_seq += 1
            name = f"gw{self._shard_seq}"
        attempts = []
        for destination, port, identity, url_host in self._candidate_hosts():
            try:
                return await self._spawn_remote_shard(
                    destination, name, port=port, identity=identity, url_host=url_host)
            except Exception as exc:  # noqa: BLE001 — next candidate, then local
                attempts.append({
                    "host": destination,
                    **({"port": port} if port is not None else {}),
                    **({"identity": identity} if identity is not None else {}),
                    "error": f"{type(exc).__name__}: {exc}"})
        if attempts:
            # Observable, not silent: every remote candidate failed, so this
            # gateway grows on the manager machine instead.
            publish(self.run_root / "gateway-spawn-fallback.json",
                    {"fallback": "local", "attempts": attempts})
        return self._spawn_local_shard(name)

    def _shard_launch_python(self):
        """The interpreter growth hosts run the shard service under.

        The shard process needs the serving stack (transformers, the harborrl
        gateway) rather than the runner's sandbox framework, so deployments
        with split environments point this at their serving interpreter;
        undeclared keeps the runner's, which uniform deployments share.
        """
        return self.config.shard_python or self.config.runner_python

    async def _start_shard_process(self, host, environment, *, port=None, identity=None):
        """Launch the shard process over ssh and read its ready line.

        Returns the parsed ready record — shard name, resolved gateway and
        admin ports — or raises; the caller abandons the candidate on any
        failure and tries the next host.
        """
        command = _shard_service_command(
            host, environment, python=self._shard_launch_python(),
            project=str(Path(__file__).resolve().parents[2]),
            port=port, identity=identity)
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        try:
            line = await asyncio.wait_for(
                process.stdout.readline(), REMOTE_SHARD_START_TIMEOUT)
            record = json.loads(line)
            if record.get("shard") is None or "gateway_port" not in record:
                raise ValueError("no shard service ready line")
            return process, record
        except BaseException:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(Exception):
                    await process.wait()
            raise

    async def _spawn_remote_shard(self, host, name, *, port=None, identity=None,
                                  url_host=None):
        """Start one shard process on a remote host and join it to the epoch."""
        from harborrl.gateway.shard_service import SHARD_SERVICE

        trace_dir = self.run_root / "native-traces" / f"shard-{name}"
        process, ready = await self._start_shard_process(
            host, _shard_service_environment(self.config, name),
            port=port, identity=identity)
        shard = None
        try:
            if ready.get("shard") != name or ready.get("service") != SHARD_SERVICE:
                raise RuntimeError(f"gateway host {host} answered as another service")
            # The shard is advertised at the host both planes reach it by,
            # which is the destination's host part (user@ stripped) unless
            # the caller names another spelling — e.g. a tunnel endpoint that
            # the manager's side forwards to the same process.
            if url_host is None:
                url_host = host.split("@")[-1]
            gateway_port, admin_port = int(ready["gateway_port"]), int(ready["admin_port"])
            spec = GatewayShardSpec(
                name=name, port=gateway_port, url=f"http://{url_host}:{gateway_port}",
                mode="remote", admin_url=f"http://{url_host}:{admin_port}")
            shard = SpawnedRemoteShard(spec, token=self._shard_token,
                                       trace_dir=trace_dir, process=process)
            # start() proves the process answers as this shard over the admin
            # plane — ssh alone only proved the process launched.
            shard.start()
            self.registry.join(shard)
        except BaseException:
            if shard is not None:
                with suppress(Exception):
                    shard.close()
            elif process.returncode is None:
                # The shard object never formed; kill the transport directly
                # so a rejected candidate leaves no process behind.
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(Exception):
                    await process.wait()
            raise
        self._auto_spawned.add(name)
        self._write_shards_evidence()
        return shard

    def _spawn_local_shard(self, name):
        """Start one more local gateway and attach it to the live epoch."""
        from harborrl.gateway.server import Gateway, make_server
        from harborrl.gateway.trace import TraceRegistry

        trace_dir = self.run_root / "native-traces" / f"shard-{name}"
        registry = TraceRegistry(trace_dir)
        gateway = Gateway(registry, self.backend, model=self.profile["model"],
                          max_output_tokens=self.config.max_output_tokens)
        # The OS owns exclusivity: an ephemeral bind cannot collide with a
        # sibling run's port allocation, and the advertised URL carries the
        # port the socket actually took.
        server = make_server(gateway, host=self.config.gateway_host, port=0,
                             max_inflight=max(4, 2 * self.config.concurrency_limit))
        port = server.server_address[1]
        spec = GatewayShardSpec(name=name, port=port,
                                url=_port_url(self.config.gateway_url, port))
        shard = GatewayShard(spec, registry=registry, gateway=gateway,
                             server=server, trace_dir=trace_dir)
        try:
            shard.start()
            self.registry.join(shard)
        except BaseException:
            with suppress(Exception):
                shard.close()
            raise
        self._auto_spawned.add(name)
        self._write_shards_evidence()
        return shard

    def retire_idle_shards(self):
        """Retire auto-spawned gateways once their sandboxes all settled.

        The mirror of growth: when a group settles and the running sandbox
        count no longer justifies a gateway, it detaches from the admission
        epoch, closes, and the evidence record shrinks back. Only idle
        auto-spawned shards retire — the implicit first gateway never does,
        and a busy shard (or anything behind it, since leases fill in order)
        blocks the tail. The group's own outcome is already decided by the
        time this runs, so a close failure is contained instead of replacing
        the settlement result.
        """
        retired = False
        while len(self._shards) > 1:
            shard = self._shards[-1]
            if (shard.name not in self._auto_spawned
                    or self._auto_inflight.get(shard.name, 0)):
                break
            self._shards.pop()
            self._auto_spawned.discard(shard.name)
            self._auto_inflight.pop(shard.name, None)
            self.registry.leave(shard)
            with suppress(Exception):
                shard.close()
            retired = True
        if retired:
            self._write_shards_evidence()

    def close(self):
        for shard in self.shards:
            shard.close()

    def manifest_path(self, identity):
        """Deterministic shared-storage path of one trajectory manifest."""
        if isinstance(identity, dict):
            identity = Identity(**identity)
        return self.rollout_root / identity.group_id / identity.attempt_id / "trajectory.v2.json"


# Historical name kept for existing hooks and evidence readers.
NativeRolloutRuntime = RolloutRuntime

_RUNTIME = None
_RUNTIME_LOCK = threading.RLock()
_EMBEDDED_LOCK = threading.RLock()
_EMBEDDED_LOOP = None
_EMBEDDED_THREAD = None


async def runtime(config, *, tokenizer=None, tokenizer_loader=None):
    """Process-wide runtime cache for the embedded GPU-side rollout path."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            if tokenizer is None:
                if tokenizer_loader is None:
                    raise ValueError("runtime requires a tokenizer or a loader")
                tokenizer = tokenizer_loader()
            _RUNTIME = RolloutRuntime(config, tokenizer)
        elif _RUNTIME.config != config:
            raise RuntimeLifecycleError("close the cached rollout runtime before changing its configuration")
        _RUNTIME.check_event_loop()
        return _RUNTIME


def run_embedded(coroutine):
    """Run successive synchronous rollout calls on one Python 3.10-compatible loop."""
    global _EMBEDDED_LOOP, _EMBEDDED_THREAD
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coroutine.close()
        raise RuntimeLifecycleError("synchronous rollout cannot run inside an active event loop")
    with _EMBEDDED_LOCK:
        if _EMBEDDED_LOOP is None:
            _EMBEDDED_LOOP = asyncio.new_event_loop()
            _EMBEDDED_THREAD = threading.get_ident()
        elif _EMBEDDED_THREAD != threading.get_ident():
            coroutine.close()
            raise RuntimeLifecycleError("embedded rollout must stay on its owning thread")
        try:
            return _EMBEDDED_LOOP.run_until_complete(coroutine)
        except BaseException:
            try:
                close_runtime()
            except Exception:
                logging.getLogger(__name__).exception("embedded rollout shutdown failed")
            raise


def publish_native_version(runtime_, states, expected):
    """Open the next admission epoch only after the complete pool is healthy."""
    runtime_.registry.drain(runtime_.config.drain_timeout)
    runtime_.registry.publish_version(states, str(expected))


def current_policy_version(runtime_):
    """Publish and lock the version currently served by the rollout pool."""
    states = runtime_.backend.pool_states()
    from harborrl.platform.policy_pool import publish_pool
    version = publish_pool(states)
    runtime_.registry.drain(runtime_.config.drain_timeout)
    runtime_.registry.publish_version(states, version)
    return version


def close_runtime():
    """Drain embedded tasks, close the gateway, then release the owning loop."""
    global _RUNTIME, _EMBEDDED_LOOP, _EMBEDDED_THREAD
    with _EMBEDDED_LOCK:
        loop = _EMBEDDED_LOOP
        if loop is not None and loop.is_running():
            raise RuntimeLifecycleError("close embedded rollout between calls")
        if loop is not None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeLifecycleError("close embedded rollout outside an active event loop")

        async def cancel_pending():
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        try:
            if loop is not None:
                loop.run_until_complete(cancel_pending())
            with _RUNTIME_LOCK:
                cached, _RUNTIME = _RUNTIME, None
                try:
                    if cached is not None:
                        cached.close()
                finally:
                    if loop is not None:
                        loop.run_until_complete(loop.shutdown_asyncgens())
                        loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            if loop is not None:
                loop.close()
            _EMBEDDED_LOOP = None
            _EMBEDDED_THREAD = None


atexit.register(close_runtime)


def task_entry(sample):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    entry = metadata.get("native_task")
    if not isinstance(entry, dict) or set(entry) != TASK_ENTRY_KEYS:
        raise ValueError("native rollout requires a locked task catalog entry")
    return entry


def slot_identity(config, group_index, slot, rollout_id, entry,
                  reward_digest, sampling_digest, policy_version, *, harness_digest=None):
    return Identity(
        run_id=config.run_id,
        batch_id=f"rollout-{rollout_id}",
        group_id=f"group-{group_index}",
        slot_id=str(slot),
        attempt_id=secrets.token_hex(16),
        trajectory_id=secrets.token_hex(16),
        task_digest=entry["task_digest"],
        policy_version=str(policy_version),
        harness_digest=harness_digest if harness_digest is not None else digest(read_json(config.profile_path)),
        reward_digest=reward_digest,
        sampling_digest=sampling_digest,
    )


def _group_slots(config, runtime_, group_index, rollout_id, entry,
                 sampling_digest, policy_version, size):
    reward_profile = RewardProfile(**entry["reward_profile"])
    harness_digest = digest(runtime_.profile)
    identities = [
        slot_identity(config, group_index, index, rollout_id, entry,
                      reward_profile.digest, sampling_digest, policy_version,
                      harness_digest=harness_digest)
        for index in range(size)
    ]
    return GroupSlots(identities, max_attempts=config.max_attempts), reward_profile


async def run_slot(config, runtime_, group_index, slot, rollout_id, entry,
                   sampling_digest, policy_version):
    """Compatibility entry point for executing an individual group slot."""
    slots, reward_profile = _group_slots(
        config, runtime_, group_index, rollout_id, entry,
        sampling_digest, policy_version, config.group_size)
    return await _run_slot(config, runtime_, slots, slot, entry, reward_profile)


def route_shard(config, shards, *, runner_host, slot, attempt_index):
    """Static per-attempt shard routing: fixed at dispatch, never balanced.

    Fail-closed by construction — an unmapped runner or an unknown shard
    raises instead of falling back to another shard's admission; a dead
    shard fails exactly its own attempts.
    """
    if len(shards) == 1:
        return shards[0]
    if config.gateway_routing == "worker_affinity":
        key = runner_host if runner_host is not None else LOCAL_RUNNER_KEY
        name = config.worker_shards.get(key)
        if name is None:
            raise ValueError(f"worker_affinity has no gateway shard for runner {key!r}")
        for shard in shards:
            if shard.name == name:
                return shard
        raise ValueError(f"worker_affinity names unknown gateway shard {name!r}")
    return shards[(slot + attempt_index) % len(shards)]


async def _run_slot(config, runtime_, slots, slot, entry, reward_profile):
    """Execute one slot with the group's shared identity and retry bookkeeping."""
    workers = config.workers
    overrides = config.task_worker_overrides
    if not hasattr(runtime_, "cleanup"):
        runtime_.cleanup = CleanupAdmission(runtime_.rollout_root)
    cleanup = runtime_.cleanup
    cleanup.check()
    for attempt_index in range(slots.max_attempts):
        identity = slots.start(str(slot))
        runner_host = overrides.get(entry["id"])
        if config.runner_transport == 'local':
            runner_host = None
        elif runner_host is None:
            if not workers:
                raise ValueError("rollout runtime requires at least one worker destination")
            runner_host = workers[(slot + attempt_index) % len(workers)]
        auto = config.gateway_routing == "auto"
        shard = None
        if not auto:
            shard = route_shard(config, runtime_.shards, runner_host=runner_host,
                                slot=slot, attempt_index=attempt_index)
        root = runtime_.rollout_root / identity.group_id / identity.attempt_id
        admission = nullcontext()
        if config.sandbox:
            from harborrl.rollout.harbor_job.sandbox_config import task_binding
            admission = runtime_.sandbox_admission.lease(task_binding(config.sandbox, entry['path']), root)
        try:
            async with admission, runtime_.semaphore:
                cleanup.check()
                if auto:
                    # Auto routing leases a gateway once this attempt actually
                    # runs: the (n+1)-th concurrent sandbox, not a queued one,
                    # is what pulls up the next gateway. Remote growth waits
                    # on an ssh subprocess, so the lease itself awaits.
                    shard = await runtime_.lease_shard()
                try:
                    ir = await run_attempt(
                        identity,
                        task_path=entry["path"],
                        profile=runtime_.profile,
                        gateway_url=shard.url,
                        runner_python=config.runner_python,
                        runner_host=runner_host,
                        root=root,
                        registry=shard.registry,
                        reward_profile=reward_profile,
                        max_output_tokens=config.max_output_tokens,
                        **({'sandbox': config.sandbox} if config.sandbox else {}),
                    )
                finally:
                    if auto:
                        runtime_.release_shard(shard)
                    cleanup.observe(root)
            cleanup.check()
        except CleanupRequiredError as exc:
            slots.finish(str(slot), error=f"{type(exc).__name__}: {exc}")
            raise
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
            slots.finish(str(slot), error=f"{type(exc).__name__}: {exc}")
            cleanup.check()
            if config.sandbox and ((root / 'cleanup-required.json').exists() or runtime_.sandbox_admission.blocked):
                raise RuntimeError('sandbox cleanup unresolved; automatic retry blocked') from exc
            continue
        slots.finish(str(slot), ir)
        return ir
    raise RuntimeError(f"native slot {slot} exhausted its retry budget")


async def cancel_and_wait(tasks, result):
    """Cancel owned tasks once and finish cleanup despite further cancellation."""
    for task in tasks:
        task.cancel()
    settled = asyncio.gather(result, *tasks, return_exceptions=True)
    while True:
        try:
            await asyncio.shield(settled)
            return
        except asyncio.CancelledError:
            # The caller re-raises its original failure after cleanup completes.
            continue


async def run_group(config, runtime_, *, entry, group_index, rollout_id,
                    sampling_digest, policy_version, group_size=None):
    """Execute one complete group and return per-slot IRs in slot order."""
    size = config.group_size if group_size is None else int(group_size)
    slots, reward_profile = _group_slots(
        config, runtime_, group_index, rollout_id, entry,
        sampling_digest, policy_version, size)
    tasks = [asyncio.create_task(
        _run_slot(config, runtime_, slots, slot, entry, reward_profile))
        for slot in range(size)
    ]
    result = asyncio.gather(*tasks)
    try:
        # Cancel children once here, so gather cannot interrupt slower cleanup.
        return list(await asyncio.shield(result))
    except BaseException:
        await cancel_and_wait(tasks, result)
        raise
    finally:
        # Settlement (success or failure) is the shrink trigger: every lease
        # of this group is released by now, so an auto-spawned gateway the
        # running sandbox count no longer justifies goes away before the
        # next group can lease it.
        if getattr(config, "gateway_routing", "auto") == "auto":
            retire = getattr(runtime_, "retire_idle_shards", None)
            if retire is not None:
                retire()
