"""Training-side client for hub-mode rollout dispatch (hub_pull provider).

The GPU job sends one attempt request per (group, slot) to the hub server;
Registered workers pull them, execute the full trial (worker-hosted gateway +
sandbox container + verifier) and report {success, ir_path, ir_sha256,
reward}. Trajectory IRs live on shared storage, so this client only needs
paths and digests over HTTP — mirroring the rollout-service contract where
heavy payloads never cross the wire.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import request, error


class HubRolloutError(RuntimeError):
    def __init__(self, message, *, status=None, code=None):
        super().__init__(message)
        self.status, self.code = status, code


class HubRolloutClient:
    """Synchronous urllib client; safe to call from asyncio.to_thread."""

    def __init__(self, hub_url, run_id, token, *, timeout=120,
                 poll_interval=2.0, opener=None):
        self.base_url = hub_url.rstrip("/")
        self.run_id = run_id
        self.token = token
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.opener = opener or request.build_opener(request.ProxyHandler({}))

    def _request(self, method, path, body=None):
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Api-Key"] = self.token
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return json.load(response)
        except error.HTTPError as exc:
            try:
                payload = json.load(exc)
            except (ValueError, OSError):
                payload = {"error": {"type": "transport", "message": str(exc)}}
            error_payload = payload.get("error", payload)
            raise HubRolloutError(
                f"{method} {path}: {error_payload.get('type')}: "
                f"{error_payload.get('message')}",
                status=exc.code, code=error_payload.get("type")) from exc
        except (error.URLError, OSError, ValueError) as exc:
            raise HubRolloutError(f"{method} {path}: {exc}", code="transport") from exc

    def submit_attempt(self, attempt_request):
        """Idempotent per (run, rollout, group, slot); returns task_id."""
        payload = self._request("POST", "/hub/rollout_tasks",
                                {"run_id": self.run_id, "request": attempt_request})
        task_id = payload.get("task_id")
        if not task_id:
            raise HubRolloutError(f"hub submit returned no task_id: {payload}")
        return task_id

    def task_status(self, task_ids):
        return self._request("POST", "/hub/rollout_tasks/status",
                             {"task_ids": list(task_ids)})


def load_attempt_ir(task_id, result, *, expected_policy_version=None,
                    expected_task_digest=None):
    """Read the trajectory IR from shared storage and verify its seal."""
    if not isinstance(result, dict):
        raise HubRolloutError(f"task {task_id} reported no result body")
    # RolloutPuller reports an outcome envelope; the hub preserves it verbatim.
    if "result" in result:
        if result.get("success") is not True or not isinstance(result["result"], dict):
            raise HubRolloutError(f"task {task_id} reported an unsuccessful result envelope")
        result = result["result"]
    if result.get("success") is False:
        raise HubRolloutError(f"task {task_id} reported a failed attempt")
    ir_path = result.get("ir_path")
    ir_sha = result.get("ir_sha256")
    if not ir_path or not ir_sha:
        raise HubRolloutError(
            f"task {task_id} completed without ir_path/ir_sha256")
    path = Path(ir_path)
    if not path.is_file():
        raise HubRolloutError(f"task {task_id} ir_path missing on shared storage: {ir_path}")
    from harborrl.trajectories.native import digest, read_json

    try:
        ir = read_json(path)
        actual_digest = digest(ir)
    except (OSError, ValueError, TypeError) as exc:
        raise HubRolloutError(f"task {task_id} invalid IR at {ir_path}: {exc}") from exc
    if actual_digest != ir_sha:
        raise HubRolloutError(f"task {task_id} IR digest mismatch at {ir_path}")
    identity = ir.get("identity", {})
    if expected_policy_version is not None and \
            str(identity.get("policy_version")) != str(expected_policy_version):
        raise HubRolloutError(
            f"task {task_id} policy version {identity.get('policy_version')!r} "
            f"!= locked {expected_policy_version!r}")
    if expected_task_digest and identity.get("task_digest") != expected_task_digest:
        raise HubRolloutError(f"task {task_id} task digest mismatch")
    if ir.get("readiness") != "RL_READY":
        raise HubRolloutError(
            f"task {task_id} IR readiness {ir.get('readiness')!r} is not RL_READY")
    if ir.get("integrity_errors"):
        raise HubRolloutError(f"task {task_id} IR integrity errors: {ir['integrity_errors']}")
    return ir


def _rewrite_task_path(path):
    """Map a trainer-visible task path onto the worker's local filesystem.

    Task trees live twice: the shared-storage copy the GPU job inspects for
    digest preflight, and the worker-local copy the sandbox actually mounts
    (sshfs paths are invisible to the worker's docker daemon). The mapping
    comes from HARBORRL_HUB_TASK_PATH_MAP="shared=local[,shared2=local2]".
    """
    mapping = os.environ.get("HARBORRL_HUB_TASK_PATH_MAP", "")
    if not mapping:
        import json as _json
        try:
            env_file = os.environ.get(
                "HARBORRL_HUB_ENV_FILE",
                "")
            mapping = _json.loads(Path(env_file).read_text(encoding="utf-8")).get(
                "HARBORRL_HUB_TASK_PATH_MAP", "")
        except (OSError, ValueError):
            mapping = ""
    for pair in filter(None, mapping.split(",")):
        if "=" not in pair:
            continue
        shared, local = pair.split("=", 1)
        if path.startswith(shared):
            return path.replace(shared, local, 1)
    return path


def hub_attempt_request(group_request, slot, *, serving_endpoint, hf_ckpt,
                        tokenizer_digest, template_digest, model_tag,
                        max_output_tokens, max_context, task_path,
                        reward_profile=None, batch_id=None, audited_raw_logprobs=False):
    """Expand one service-style GroupRequest into a per-slot attempt request.

    Field superset consumed by deployment.hub_rollout_executor.execute().
    """
    task = group_request["task"]
    sampling = group_request.get("sampling", {})
    if type(audited_raw_logprobs) is not bool:
        raise ValueError("audited_raw_logprobs must be boolean")
    if 'rollout_generation' in group_request:
        from harborrl.rollout.generations import generation_batch_id
        locked_batch = generation_batch_id(
            group_request['rollout_id'], group_request['rollout_generation'])
        if batch_id is not None and batch_id != locked_batch:
            raise ValueError('hub batch_id conflicts with rollout generation')
        batch_id = locked_batch
    elif batch_id is None:
        batch_id = 'batch'
    return {
        # slot identity — also the hub's idempotency key part
        "rollout_id": group_request["rollout_id"],
        "group_id": group_request["group_id"],
        "group_index": group_request.get("group_index"),
        "slot_id": str(slot["slot_id"]),
        "batch_id": batch_id,
        "trajectory_id": f"{group_request['group_id']}-s{slot['slot_id']}",
        "policy_version": group_request["policy_version"],
        # task lock (worker executes from its local mirror of the tree)
        "task_path": _rewrite_task_path(task["path"]),
        "task_digest": task.get("task_digest", ""),
        "reward_profile": reward_profile or {"key": "reward", "scale": 1.0,
                                             "offset": 0.0, "raw_range": [0.0, 1.0]},
        # generation
        "model_tag": model_tag,
        "serving_endpoint": serving_endpoint,
        "hf_ckpt": hf_ckpt,
        "audited_raw_logprobs": audited_raw_logprobs,
        "tokenizer_digest": tokenizer_digest,
        "template_digest": template_digest,
        "max_output_tokens": max_output_tokens,
        "max_context": max_context,
        "stop": list(sampling.get("stop") or []),
        "sampling": sampling,
    }


def client_from_environment():
    """Build a HubRolloutClient from HARBORRL_HUB_* variables or a JSON file.

    Ray workers inherit the raylet's startup environment, which predates the
    trainer shell's exports on platform-bootstrapped clusters; a shared JSON
    file (HARBORRL_HUB_ENV_FILE) reaches both the trainer and the workers.
    The file maps variable names to values and overrides nothing already set.
    """
    env_file = os.environ.get(
        "HARBORRL_HUB_ENV_FILE",
        "")
    extra = {}
    if env_file:
        path = Path(env_file)
        if path.is_file():
            extra = json.loads(path.read_text(encoding="utf-8"))

    def setting(name, default=None):
        return os.environ.get(name) or extra.get(name) or default

    hub_url = setting("HARBORRL_HUB_URL")
    if not hub_url:
        raise HubRolloutError(
            "HARBORRL_HUB_URL is not set (env or HARBORRL_HUB_ENV_FILE); hub rollout requires it")
    token_file = setting("HARBORRL_HUB_TOKEN_FILE")
    token = ""
    if token_file:
        token = Path(token_file).read_text().strip()
    run_id = os.environ.get("RUN_ID") or extra.get("RUN_ID") or "adhoc"
    return HubRolloutClient(hub_url, run_id, token)
