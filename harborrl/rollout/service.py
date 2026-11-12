"""RolloutService (P3): the manager-side HTTP control plane for rollout groups.

First-phase contract: HTTP JSON, run-scoped tokens, idempotent group
submission. Bulky payloads (trajectory manifests, token ids, logprobs, loss
masks) live on the shared run directory; HTTP responses carry paths plus
SHA-256 digests, and the SlimeAdapter recomputes every digest before
converting anything into training samples.

Rules that keep the lifecycle safe:
- same group id and same content -> the existing state, never a second run;
- same group id and different content -> rejected;
- after a timeout the client must query, never resubmit;
- a new policy version is admitted only after the whole serving pool is
  healthy on it (drain first, publish second).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from harborrl.rollout.exporters.native import export_group
from harborrl.rollout.runtime import run_group
from harborrl.trajectories.native import (
    canonical_sampling,
    digest,
    finite,
    publish,
    read_json,
)

SERVICE_SCHEMA = "harborrl-rollout-service-v1"
GROUP_REQUEST_KEYS = {"group_id", "rollout_id", "group_index", "policy_version",
                      "task", "sampling", "slots"}
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "max_new_tokens", "stop"}
TASK_ENTRY_KEYS = {"id", "revision", "path", "task_digest", "reward_profile"}
GROUP_DIR = "rollout-service/groups"


class ServiceError(Exception):
    """Maps onto one HTTP status; carries a machine-readable code."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def validate_group_request(request):
    if not isinstance(request, dict) or set(request) != GROUP_REQUEST_KEYS:
        raise ServiceError(400, "invalid_request",
                           f"group request requires exactly {sorted(GROUP_REQUEST_KEYS)}")
    for key in ("group_id", "policy_version"):
        if not isinstance(request[key], str) or not request[key].strip():
            raise ServiceError(400, "invalid_request", f"{key} must be a nonempty string")
    if request["group_id"] in (".", "..") or any(c in request["group_id"] for c in ("/", "\\")):
        raise ServiceError(400, "invalid_request", "group_id must be one directory component")
    for key in ("rollout_id", "group_index"):
        if type(request[key]) is not int or request[key] < 0:
            raise ServiceError(400, "invalid_request", f"{key} must be a nonnegative integer")
    task = request["task"]
    if not isinstance(task, dict) or set(task) != TASK_ENTRY_KEYS:
        raise ServiceError(400, "invalid_request",
                           f"task requires exactly {sorted(TASK_ENTRY_KEYS)}")
    if not isinstance(task["reward_profile"], dict):
        raise ServiceError(400, "invalid_request", "task.reward_profile must be a mapping")
    sampling = request["sampling"]
    if not isinstance(sampling, dict) or set(sampling) != SAMPLING_KEYS:
        raise ServiceError(400, "invalid_request",
                           f"sampling requires exactly {sorted(SAMPLING_KEYS)}")
    if not finite(sampling["temperature"]) or not finite(sampling["top_p"]):
        raise ServiceError(400, "invalid_request", "sampling probabilities must be finite")
    if type(sampling["top_k"]) is not int:
        raise ServiceError(400, "invalid_request", "sampling.top_k must be an integer")
    if type(sampling["max_new_tokens"]) is not int or sampling["max_new_tokens"] < 1:
        raise ServiceError(400, "invalid_request", "sampling.max_new_tokens must be positive")
    if (not isinstance(sampling["stop"], list)
            or any(not isinstance(stop, str) or not stop for stop in sampling["stop"])):
        raise ServiceError(400, "invalid_request", "sampling.stop must be a string list")
    slots = request["slots"]
    if (not isinstance(slots, list) or len(slots) < 2
            or any(not isinstance(slot, dict) or set(slot) != {"slot_id"}
                   or not isinstance(slot["slot_id"], str) or not slot["slot_id"].strip()
                   for slot in slots)):
        raise ServiceError(400, "invalid_request",
                           "slots must describe at least two unique slot ids")
    if len({slot["slot_id"] for slot in slots}) != len(slots):
        raise ServiceError(400, "invalid_request", "slot ids must be unique")
    return request


def request_digest(request):
    """Canonical digest of the full request content, group id included."""
    return digest(request)


def manifest_path_from(run_root, identity):
    """Deterministic trajectory manifest path for one identity."""
    return Path(run_root) / "native-rollouts" / identity["group_id"] / \
        identity["attempt_id"] / "trajectory.v2.json"


def build_group_result(run_id, request, trajectories, *, runtime, export=None):
    if len(trajectories) != len(request["slots"]):
        raise ServiceError(500, "incomplete_group",
                           "executor returned a different slot count")
    slots = []
    for descriptor, ir in zip(request["slots"], trajectories):
        identity = ir["identity"]
        path = runtime.manifest_path(identity)
        payload = read_json(path)
        seal = ir["trace_seal"]
        slots.append({
            "slot_id": descriptor["slot_id"],
            "trajectory_id": identity["trajectory_id"],
            "attempt_id": identity["attempt_id"],
            "manifest_path": str(path),
            "manifest_sha256": digest(payload),
            "finish_reason": ir["termination"],
            "reward": ir["evaluation"]["training_reward"],
            "policy_version": identity["policy_version"],
            "task_digest": identity["task_digest"],
            "harness_digest": identity["harness_digest"],
            "sampling_digest": identity["sampling_digest"],
            "readiness": ir["readiness"],
            "integrity_errors": ir["integrity_errors"],
            "trace_seal": {"status": seal["status"], "errors": seal["errors"],
                           "digest": digest(seal)},
            "verifier_digest": ir["evaluation"].get("source_result_digest"),
        })
    result = {
        "schema_version": SERVICE_SCHEMA,
        "run_id": run_id,
        "group_id": request["group_id"],
        "rollout_id": request["rollout_id"],
        "request_digest": request_digest(request),
        "policy_version": request["policy_version"],
        "status": "complete",
        "slots": slots,
        "completed_at": time.time(),
    }
    if export is not None:
        result["group_export"] = export
    return result


async def default_executor(config, runtime_, request):
    """The production executor: the framework-free rollout core."""
    return await run_group(
        config, runtime_, entry=request["task"], group_index=request["group_index"],
        rollout_id=request["rollout_id"], sampling_digest=digest(canonical_sampling(request["sampling"])),
        policy_version=request["policy_version"], group_size=len(request["slots"]))


@dataclass
class GroupState:
    request: dict
    request_sha: str
    status: str = "queued"
    result: dict = None
    error: str = None
    created_at: float = field(default_factory=time.time)
    future: object = None
    task: object = None
    cancel_requested: bool = False

    def summary(self, *, include_result=True):
        payload = {"group_id": self.request["group_id"], "status": self.status,
                   "request_digest": self.request_sha,
                   "created_at": self.created_at}
        if self.error:
            payload["error"] = self.error
        if include_result and self.result is not None:
            payload["result"] = self.result
        return payload


@dataclass
class RunState:
    run_id: str
    manifest: dict
    manifest_sha: str
    token_digest: str
    config: object = None
    runtime: object = None
    drained: bool = False
    serving: str = None
    groups: dict = field(default_factory=dict)


class RolloutService:
    """Run registry plus a single-threaded asyncio executor for group work."""

    def __init__(self, *, run_root, executor=None, runtime_factory=None,
                 token_file=None, group_wait_timeout=3600.0, poll_interval=0.05,
                 allow_synthetic=False, cleanup_timeout=None):
        if cleanup_timeout is not None and (not finite(cleanup_timeout) or cleanup_timeout <= 0):
            raise ValueError("cleanup_timeout must be positive and finite")
        self.run_root = Path(run_root).resolve()
        self.executor = executor or default_executor
        self.runtime_factory = runtime_factory
        self.allow_synthetic = allow_synthetic
        self.token = None
        if token_file is not None:
            token = Path(token_file).read_text().strip()
            if not token:
                raise ServiceError(500, "invalid_token_file", "run token file is empty")
            self.token = token
        self.group_wait_timeout = group_wait_timeout
        self.poll_interval = poll_interval
        self.runs = {}
        self.lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._closing = self._closed = False
        self.cleanup_timeout = cleanup_timeout
        self._store_lock = self._lock_run_directory()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve_loop,
                                       name="harborrl-rollout-service", daemon=True)
        try:
            self.thread.start()
        except RuntimeError:
            self.loop.close()
            self._store_lock.close()
            raise

    def _lock_run_directory(self):
        directory = self.run_root / "rollout-service"
        directory.mkdir(parents=True, exist_ok=True)
        stream = (directory / "service.lock").open("a+b")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                stream.seek(0)
                if not stream.read(1):
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            stream.close()
            raise ServiceError(409, "run_directory_busy", "another service owns this run directory") from exc
        return stream

    def _serve_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _cleanup_budget(self):
        if self.cleanup_timeout is not None:
            return self.cleanup_timeout
        # Runner cancellation allows 120s, or the configured Sandbox deletion
        # deadline plus 15s. The service must not cut that lifecycle to 30s.
        budgets = [120.0]
        with self.lock:
            states = list(self.runs.values())
        for state in states:
            sandbox = getattr(state.config, "sandbox", None)
            if sandbox:
                budgets.append(float(sandbox["cleanup_timeout_s"]) + 15)
        return max(budgets) + 30

    def _ensure_open(self):
        if self._closing:
            raise ServiceError(503, "service_closing", "rollout service is closing")

    def _shutdown_failed(self, error):
        marker = self.run_root / "native-rollouts" / "cleanup-required.json"
        if not marker.exists():
            publish(marker, {"reason": "RolloutService shutdown incomplete",
                             "error": str(error),
                             "groups": {state.run_id: {key: group.status for key, group in state.groups.items()}
                                        for state in self.runs.values()}})

    async def _shutdown_loop_resources(self):
        await self.loop.shutdown_asyncgens()
        await self.loop.shutdown_default_executor()

    def close(self):
        if threading.current_thread() is self.thread:
            raise RuntimeError("close the rollout service outside its executor loop")
        with self._close_lock:
            if self._closed:
                return
            with self.lock:
                self._closing = True
                states = list(self.runs.values())
                for state in states:
                    state.drained = True
                groups = [group for state in states for group in state.groups.values()]
            try:
                self._cancel_groups(groups)
                for state in states:
                    if state.runtime is None:
                        continue
                    registry = getattr(state.runtime, "registry", None)
                    if registry is not None:
                        registry.drain(float(getattr(state.config, "drain_timeout", 30)))
                    close = getattr(state.runtime, "close", None)
                    if close is not None:
                        close()
                future = asyncio.run_coroutine_threadsafe(self._shutdown_loop_resources(), self.loop)
                future.result(timeout=self._cleanup_budget())
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(5)
                if self.thread.is_alive():
                    raise ServiceError(409, "shutdown_timeout", "rollout executor thread has not stopped")
                self.loop.close()
                self._store_lock.close()
                self._closed = True
            except Exception as exc:
                # Keep the executor alive to finish pending cleanup; never claim
                # that a deadline or a cancelled wrapper Future proves deletion.
                self._shutdown_failed(exc)
                raise

    # -- run lifecycle ----------------------------------------------------

    def _token_digest(self, token):
        return hashlib.sha256(token.encode()).hexdigest()

    def register_run(self, run_id, manifest, *, token=None):
        if (not isinstance(run_id, str) or not run_id.strip() or run_id in (".", "..")
                or any(c in run_id for c in ("/", "\\"))):
            raise ServiceError(400, "invalid_request", "run id must be one directory component")
        if not isinstance(manifest, dict) or not manifest:
            raise ServiceError(400, "invalid_request", "manifest must be a nonempty mapping")
        if manifest.get("run_id", run_id) != run_id:
            raise ServiceError(400, "invalid_request", "manifest run id must match the path")
        manifest_sha = digest(manifest)
        with self.lock:
            self._ensure_open()
            existing = self.runs.get(run_id)
            if existing is not None:
                if existing.manifest_sha != manifest_sha:
                    raise ServiceError(409, "conflict",
                                       "run id already registered with a different manifest")
                return {"run_id": run_id, "state": "registered", "existed": True}
            if self.token is not None and token != self.token:
                raise ServiceError(401, "unauthorized", "invalid manager run token")
            manifest_path = self.run_root / "rollout-service/runs" / run_id / "manifest.json"
            existed = manifest_path.exists()
            if existed and digest(read_json(manifest_path)) != manifest_sha:
                raise ServiceError(409, "conflict", "persisted run manifest differs from registration")
            groups = self._restore_groups(run_id)
            publish(manifest_path, manifest)
            generated = token is None
            if generated:
                token = secrets.token_urlsafe(32)
            self.runs[run_id] = RunState(run_id=run_id, manifest=manifest,
                                         manifest_sha=manifest_sha,
                                         token_digest=self._token_digest(token), groups=groups,
                                         drained=any(group.status == "unknown" for group in groups.values()))
        return {"run_id": run_id, "state": "registered", "existed": existed,
                "recovered_groups": len(groups), **({"run_token": token} if generated else {})}

    def authenticate(self, run_id, token):
        with self.lock:
            state = self.runs.get(run_id)
        if state is None:
            raise ServiceError(404, "not_found", f"unknown run: {run_id}")
        if not isinstance(token, str) or not token or not hmac.compare_digest(
                state.token_digest, self._token_digest(token)):
            raise ServiceError(401, "unauthorized", "invalid run token")
        return state

    def _run(self, run_id):
        state = self.runs.get(run_id)
        if state is None:
            raise ServiceError(404, "not_found", f"unknown run: {run_id}")
        return state

    def attach_runtime(self, run_id, config, runtime):
        """Bind the rollout runtime that executes this run's groups."""
        with self.lock:
            self._ensure_open()
            state = self._run(run_id)
            state.config, state.runtime = config, runtime
        return {"run_id": run_id, "state": "attached"}

    def announce_serving(self, run_id, router_url):
        parts = urlsplit(router_url or "")
        if (parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username or parts.password or parts.query
                or parts.fragment or parts.path not in ("", "/")):
            raise ServiceError(400, "invalid_request", "router_url must be a plain origin")
        with self.lock:
            self._ensure_open()
            state = self._run(run_id)
            state.serving = f"{parts.scheme}://{parts.netloc}".rstrip("/")
            if state.runtime is None and self.runtime_factory is not None:
                runtime = self.runtime_factory(run_id, state.serving)
                state.runtime = runtime
                if state.config is None:
                    # The factory-built runtime carries its own config; the
                    # executor needs both for group admission.
                    state.config = getattr(runtime, "config", None)
        return {"run_id": run_id, "router_url": state.serving,
                "runtime": "attached" if state.runtime is not None else "pending"}

    # -- policy lifecycle (§7) ---------------------------------------------

    def activate_policy(self, run_id, states, expected):
        from harborrl.platform.policy_pool import validate_pool

        with self.lock:
            self._ensure_open()
            state = self._run(run_id)
        if state.runtime is None:
            raise ServiceError(409, "admission_closed", "run has no serving runtime yet")
        try:
            version = validate_pool(states, str(expected))
            backend = getattr(state.runtime, 'backend', None)
            if backend is not None:
                live_states = backend.pool_states()
                validate_pool(live_states, version)
                if {item['endpoint'] for item in live_states} != {item['endpoint'] for item in states}:
                    raise ValueError('manager and GPU disagree on serving engine membership')
                states = live_states
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ServiceError(409, "pool_invalid", str(exc)) from exc
        with self.lock:
            # Pool probes above can race a shutdown. Do not reopen generation
            # admission after close has frozen the service.
            self._ensure_open()
            registry = state.runtime.registry
            registry.drain(float(getattr(state.config, "drain_timeout", 30)))
            registry.publish_version(states, version)
            # Mirror the admitted version into shared storage.
            payload = {"weight_version": version, "engines": states, "timestamp": time.time()}
            temp = self.run_root / "policy-pool.tmp"
            temp.write_text(json.dumps(payload) + "\n")
            temp.replace(self.run_root / "policy-pool.json")
        return {"run_id": run_id, "policy_version": version, "admission": "open"}

    # -- group lifecycle ----------------------------------------------------

    def _group_dir(self, run_id, group_id):
        return self.run_root / GROUP_DIR / run_id / group_id

    def _restore_groups(self, run_id):
        """Restore durable outcomes, never resume an unowned Runner or resample."""
        groups = {}
        directory = self.run_root / GROUP_DIR / run_id
        for root in sorted(directory.glob("*")):
            if not root.is_dir():
                continue
            request_path = root / "request.json"
            if not request_path.exists():
                if any((root / name).exists() for name in ("running.json", "result.json", "failed.json", "cancelled.json", "unknown.json")):
                    raise ServiceError(500, "invalid_persisted_group", f"missing request: {root.name}")
                continue
            try:
                request = validate_group_request(read_json(request_path))
                if request["group_id"] != root.name:
                    raise ValueError("group directory differs from request")
                sha = request_digest(request)
                terminal = [(name, status) for name, status in (
                    ("result.json", "complete"), ("failed.json", "failed"), ("cancelled.json", "cancelled"))
                    if (root / name).exists()]
                if len(terminal) > 1:
                    raise ValueError("conflicting terminal group records")
                if terminal:
                    name, status = terminal[0]
                    record = read_json(root / name)
                else:
                    status = "unknown"
                    record_path = root / "unknown.json" if (root / "unknown.json").exists() else root / "running.json"
                    record = read_json(record_path) if record_path.exists() else {
                        "group_id": root.name, "request_digest": sha, "created_at": time.time()}
                if not isinstance(record, dict):
                    raise TypeError("persisted group record must be a mapping")
                if record.get("group_id") != root.name or record.get("request_digest") != sha:
                    raise ValueError("persisted group does not match its request")
                if terminal and record.get("status") != status:
                    raise ValueError("terminal status differs from its record")
                if status == "complete" and (
                    record.get("run_id") != run_id or record.get("schema_version") != SERVICE_SCHEMA
                    or record.get("policy_version") != request["policy_version"]
                    or [slot["slot_id"] for slot in record["slots"]] != [slot["slot_id"] for slot in request["slots"]]
                ):
                    raise ValueError("persisted result identity differs from its request")
                created_at = record.get("created_at")
                if created_at is None and (root / "running.json").exists():
                    running = read_json(root / "running.json")
                    if (not isinstance(running, dict) or running.get("request_digest") != sha
                            or running.get("group_id") != root.name):
                        raise ValueError("running record differs from its request")
                    created_at = running["created_at"]
                group = GroupState(request, sha, status=status,
                                   result=record if status == "complete" else None,
                                   error=record.get("error"), created_at=created_at or 0)
                if status == "unknown":
                    group.error = "Service restarted before durable group completion; confirm Runner cleanup and use a new run"
                    publish(root / "unknown.json", group.summary(include_result=False))
                    marker = self.run_root / "native-rollouts/cleanup-required.json"
                    if not marker.exists():
                        publish(marker, {"run_id": run_id, "group_id": root.name,
                                         "reason": "Interrupted group has unknown Runner cleanup status"})
                groups[root.name] = group
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ServiceError(500, "invalid_persisted_group", f"cannot restore {root.name}: {exc}") from exc
        return groups

    def submit_group(self, run_id, request, *, wait=True):
        request = validate_group_request(request)
        sha = request_digest(request)
        with self.lock:
            state = self._run(run_id)
            existing = state.groups.get(request["group_id"])
            if existing is not None:
                if existing.request_sha != sha:
                    raise ServiceError(409, "conflict",
                                       "group id already submitted with different content")
                group = existing
            else:
                self._ensure_open()
                if state.drained:
                    raise ServiceError(409, "admission_closed", "run is draining")
                if state.runtime is None or state.config is None:
                    raise ServiceError(409, "admission_closed",
                                       "run has no rollout runtime attached")
                registry = getattr(state.runtime, "registry", None)
                version = getattr(registry, "version", None) if registry is not None else None
                accepting = getattr(registry, "accepting", True) if registry is not None else False
                if not accepting or version != request["policy_version"]:
                    raise ServiceError(
                        409, "stale_policy",
                        f"policy {request['policy_version']!r} is not the admitted version "
                        f"({version!r}, accepting={accepting}); activate first")
                group = GroupState(request=request, request_sha=sha)
                state.groups[request["group_id"]] = group
                future = asyncio.run_coroutine_threadsafe(
                    self._execute(state, group), self.loop)
                group.future = future
        if wait:
            return self._wait_group(run_id, group)
        return group.summary(include_result=False)

    def _wait_group(self, run_id, group):
        deadline = time.monotonic() + self.group_wait_timeout
        while group.status in ("queued", "running"):
            if time.monotonic() >= deadline:
                return {**group.summary(include_result=False),
                        "hint": "query GET /groups/{group_id}; do not resubmit"}
            time.sleep(self.poll_interval)
        return group.summary()

    async def _execute(self, state, group):
        # Immutable state records: each transition gets its own file, and
        # publish() refuses to rewrite history.
        group.task = asyncio.current_task()
        request = group.request
        directory = self._group_dir(state.run_id, request["group_id"])
        try:
            publish(directory / "request.json", request)
            if group.cancel_requested:
                raise asyncio.CancelledError
            group.status = "running"
            publish(directory / "running.json", group.summary(include_result=False))
            trajectories = await self.executor(state.config, state.runtime, request)
            export = export_group(
                trajectories, len(trajectories),
                allow_synthetic=self.allow_synthetic)
            export_path = directory / "group-export.json"
            publish(export_path, export)
            result = build_group_result(
                state.run_id, request, trajectories, runtime=state.runtime,
                export={"path": str(export_path), "sha256": digest(export)})
            publish(directory / "result.json", result)
            group.result = result
            group.status = "complete"
        except asyncio.CancelledError:
            group.status = "cancelled"
            publish(directory / "cancelled.json", group.summary(include_result=False))
            raise
        except Exception as exc:  # noqa: BLE001 — any executor failure is
            # recorded as a failed group instead of killing the service
            group.status = "failed"
            group.error = f"{type(exc).__name__}: {exc}"
            publish(directory / "failed.json", group.summary(include_result=False))

    def group_status(self, run_id, group_id):
        with self.lock:
            state = self._run(run_id)
            group = state.groups.get(group_id)
        if group is None:
            raise ServiceError(404, "not_found", f"unknown group: {group_id}")
        return group.summary()

    def cancel_group(self, run_id, group_id):
        with self.lock:
            state = self._run(run_id)
            group = state.groups.get(group_id)
        if group is None:
            raise ServiceError(404, "not_found", f"unknown group: {group_id}")
        if group.status == "unknown":
            raise ServiceError(409, "cleanup_required", "interrupted group has no owned Task; inspect Runner cleanup")
        self._cancel_groups([group])
        return group.summary()

    def drain_run(self, run_id):
        with self.lock:
            state = self._run(run_id)
            state.drained = True
            runtime = state.runtime
        if runtime is not None:
            registry = getattr(runtime, "registry", None)
            if registry is not None:
                timeout = float(getattr(state.config, "drain_timeout", 30))
                try:
                    registry.drain(timeout)
                except TimeoutError as exc:
                    raise ServiceError(409, "drain_timeout",
                                       f"generation still in flight: {exc}") from exc
        return {"run_id": run_id, "state": "drained"}

    def cancel_run(self, run_id):
        with self.lock:
            state = self._run(run_id)
            state.drained = True
            groups = list(state.groups.values())
        if any(group.status == "unknown" for group in groups):
            raise ServiceError(409, "cleanup_required", "interrupted groups require external cleanup confirmation")
        self._cancel_groups(groups)
        return {"run_id": run_id, "state": "cancelled",
                "groups": {group.request["group_id"]: group.status for group in groups}}

    async def _cancel_and_wait(self, groups):
        for group in groups:
            if not group.cancel_requested and group.future is not None and not group.future.done():
                group.cancel_requested = True
                if group.task is not None:
                    group.task.cancel()
        # Do not cancel the run_coroutine_threadsafe Future: that wrapper can
        # become done before the actual Task's asynchronous finally completes.
        pending = [asyncio.wrap_future(group.future) for group in groups if group.future is not None]
        results = await asyncio.gather(*pending, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise ServiceError(500, "cleanup_failed", f"group completion failed: {result}")

    def _cancel_groups(self, groups):
        if not any(group.future is not None for group in groups):
            return
        if threading.current_thread() is self.thread:
            raise RuntimeError("cancel rollout groups outside their executor loop")
        joined = asyncio.run_coroutine_threadsafe(self._cancel_and_wait(groups), self.loop)
        try:
            joined.result(timeout=self._cleanup_budget())
        except concurrent.futures.TimeoutError as exc:
            # Leave the waiter and group cleanup running; a second cancel would
            # interrupt their finally blocks. Retrying close may join them later.
            raise ServiceError(409, "cleanup_timeout", "rollout groups are still cleaning up") from exc


def make_control_server(service, host="127.0.0.1", port=0, *, max_body_bytes=8 * 1024 * 1024):
    """Bounded stdlib HTTP transport for the control plane."""

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def token(self):
            token = self.headers.get("x-api-key", "")
            auth = self.headers.get("Authorization", "")
            if auth:
                if not auth.startswith("Bearer ") or token:
                    raise ServiceError(400, "invalid_request", "ambiguous authentication")
                token = auth[7:]
            return token

        def read_json(self):
            if self.headers.get("Transfer-Encoding"):
                raise ServiceError(400, "invalid_request", "chunked bodies are not supported")
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > max_body_bytes:
                raise ServiceError(400, "invalid_request", "invalid request body size")
            return json.loads(self.rfile.read(size))

        def dispatch(self, method):
            parts = [part for part in urlsplit(self.path).path.split("/") if part]
            try:
                payload = self.route(method, parts)
            except ServiceError as exc:
                self.reply(exc.status, {"error": {"type": exc.code, "message": str(exc)}})
            except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": {"type": "invalid_request",
                                           "message": f"{type(exc).__name__}: {exc}"}})
            else:
                self.reply(200, payload)

        def route(self, method, parts):
            if method == "GET" and parts == ["healthz"]:
                return {"ok": True, "service": SERVICE_SCHEMA}
            if method == "GET" and parts == ["readyz"]:
                with service.lock:
                    attached = sum(1 for state in service.runs.values()
                                   if state.runtime is not None)
                return {"ok": True, "runs": len(service.runs), "attached_runs": attached}
            if len(parts) < 2 or parts[0] != "runs":
                raise ServiceError(404, "not_found", "unsupported endpoint")
            run_id = parts[1]
            if len(parts) == 3 and method == "POST" and parts[2] == "register":
                body = self.read_json()
                if not isinstance(body, dict) or set(body) != {"manifest"}:
                    raise ServiceError(400, "invalid_request", "body must be {manifest}")
                return service.register_run(run_id, body["manifest"], token=self.token())
            service.authenticate(run_id, self.token())
            if len(parts) == 3 and method == "POST" and parts[2] == "groups":
                return service.submit_group(run_id, self.read_json())
            if (len(parts) == 4 and parts[2] == "groups"
                    and method == "GET"):
                return service.group_status(run_id, parts[3])
            if (len(parts) == 5 and parts[2] == "groups" and parts[4] == "cancel"
                    and method == "POST"):
                return service.cancel_group(run_id, parts[3])
            if len(parts) == 4 and method == "POST" and parts[2] == "policy" \
                    and parts[3] == "activate":
                body = self.read_json()
                if not isinstance(body, dict) or set(body) != {"states", "expected"}:
                    raise ServiceError(400, "invalid_request",
                                       "body must be {states, expected}")
                return service.activate_policy(run_id, body["states"], body["expected"])
            if len(parts) == 4 and method == "POST" and parts[2] == "serving" \
                    and parts[3] == "announce":
                body = self.read_json()
                if not isinstance(body, dict) or set(body) != {"router_url"}:
                    raise ServiceError(400, "invalid_request", "body must be {router_url}")
                return service.announce_serving(run_id, body["router_url"])
            if len(parts) == 3 and method == "POST" and parts[2] == "drain":
                return service.drain_run(run_id)
            if len(parts) == 3 and method == "POST" and parts[2] == "cancel":
                return service.cancel_run(run_id)
            raise ServiceError(404, "not_found", "unsupported endpoint")

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
