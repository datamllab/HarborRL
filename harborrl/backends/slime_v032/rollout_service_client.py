"""SlimeAdapter HTTP client (P3): complete groups from the RolloutService.

The GPU job never trusts a GroupResult: every manifest digest is recomputed,
identities are checked against the request, the group must be complete with
at least two RL_READY slots, and the policy version must be the locked one.
After a transport timeout the client queries status; it never resubmits.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import error, parse, request

from harborrl.rollout.runtime import task_entry
from harborrl.trajectories.native import (
    canonical_sampling,
    digest,
    read_json,
    require_ready,
)

SERVICE_SCHEMA = "harborrl-rollout-service-v1"


def fetch_pool_states(endpoint, *, timeout=120, opener=None):
    """Read the serving pool states directly from the SGLang router.

    Mirrors SGLangBackend.pool_states without a tokenizer: the GPU job calls
    this before asking the manager to admit the version it currently serves.
    """
    import json as json_module
    from urllib import request as url_request

    opener = opener or url_request.build_opener(url_request.ProxyHandler({}))
    endpoint = endpoint.rstrip("/")

    def get(url):
        with opener.open(url_request.Request(url), timeout=timeout) as response:
            return json_module.load(response)

    workers = get(endpoint + "/workers").get("workers")
    if not isinstance(workers, list) or not workers:
        raise ServiceRolloutError("serving router returned no workers")
    states = []
    for worker in workers:
        engine = worker.get("url") if isinstance(worker, dict) else None
        if not isinstance(engine, str) or not engine.strip():
            raise ServiceRolloutError("serving router returned an invalid worker URL")
        engine = engine.rstrip("/")
        version = get(engine + "/get_weight_version").get("weight_version")
        if not isinstance(version, str) or not version.strip():
            raise ServiceRolloutError("missing serving weight version")
        states.append({"endpoint": engine, "healthy": True,
                       "weight_version": version})
    versions = {state["weight_version"] for state in states}
    if len(versions) != 1:
        raise ServiceRolloutError(
            f"serving pool is mid-update; retry after all engines agree: {sorted(versions)}")
    return states


class ServiceRolloutError(RuntimeError):
    def __init__(self, message, *, status=None, code=None):
        super().__init__(message)
        self.status, self.code = status, code


def service_group_request(group, sampling_params, rollout_id, policy_version):
    """Build one GroupRequest from a Slime sample group."""
    entries = [task_entry(sample) for sample in group]
    if any(entry != entries[0] for entry in entries[1:]):
        raise ValueError("native rollout group mixes task locks")
    indexes = {getattr(sample, "group_index", None) for sample in group}
    if len(indexes) != 1 or type(next(iter(indexes), None)) is not int:
        raise ValueError("Slime must provide one group identity before native generation")
    index = indexes.pop()
    return {
        "group_id": f"group-{index}",
        "rollout_id": int(rollout_id),
        "group_index": index,
        "policy_version": str(policy_version),
        "task": entries[0],
        "sampling": {key: sampling_params.get(key) for key in
                     ("temperature", "top_p", "top_k", "max_new_tokens", "stop")},
        "slots": [{"slot_id": str(slot)} for slot in range(len(group))],
    }


class RolloutServiceClient:
    """Synchronous urllib client; safe to call from asyncio.to_thread."""

    def __init__(self, base_url, run_id, token, *, timeout=120, poll_interval=1.0,
                 opener=None):
        self.base_url = base_url.rstrip("/")
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
            headers["Authorization"] = f"Bearer {self.token}"
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
            raise ServiceRolloutError(
                f"{method} {path}: {error_payload.get('type')}: "
                f"{error_payload.get('message')}",
                status=exc.code, code=error_payload.get("type")) from exc
        except (error.URLError, OSError, ValueError) as exc:
            raise ServiceRolloutError(f"{method} {path}: {exc}",
                                      code="transport") from exc

    def health(self):
        return self._request("GET", "/healthz")

    def register(self, manifest):
        return self._request("POST", f"/runs/{parse.quote(self.run_id)}/register",
                             {"manifest": manifest})

    def submit_group(self, group_request, *, max_attempts=None):
        """Submit idempotently; on timeout poll status, never regenerate."""
        path = f"/runs/{parse.quote(self.run_id)}/groups"
        try:
            status = self._request('POST', path, group_request)
        except ServiceRolloutError as exc:
            if exc.code != 'transport':
                raise
            status = self.group_status(group_request['group_id'])
        polls = 0
        while status['status'] in ('queued', 'running'):
            if max_attempts is not None and polls >= max_attempts:
                raise ServiceRolloutError('group polling deadline reached', code='transport')
            polls += 1
            time.sleep(self.poll_interval)
            status = self.group_status(group_request["group_id"])
        return status

    def group_status(self, group_id):
        return self._request(
            "GET", f"/runs/{parse.quote(self.run_id)}/groups/{parse.quote(group_id)}")

    def cancel_group(self, group_id):
        return self._request(
            "POST",
            f"/runs/{parse.quote(self.run_id)}/groups/{parse.quote(group_id)}/cancel")

    def activate_policy(self, states, expected):
        return self._request(
            "POST", f"/runs/{parse.quote(self.run_id)}/policy/activate",
            {"states": states, "expected": str(expected)})

    def announce_serving(self, router_url):
        return self._request(
            "POST", f"/runs/{parse.quote(self.run_id)}/serving/announce",
            {"router_url": router_url})

    def drain(self):
        return self._request("POST", f"/runs/{parse.quote(self.run_id)}/drain")

    def cancel_run(self):
        return self._request("POST", f"/runs/{parse.quote(self.run_id)}/cancel")


def load_group_result(run_id, group_request, result, *, allow_synthetic=False):
    """Validate a GroupResult and load its trajectory manifests in slot order."""
    if not isinstance(result, dict):
        raise ServiceRolloutError("group result must be a mapping")
    if result.get("status") != "complete":
        raise ServiceRolloutError(
            f"group is {result.get('status')!r}, not complete: {result.get('error', '')}")
    if result.get("run_id") != run_id or result.get("group_id") != group_request["group_id"]:
        raise ServiceRolloutError("group result identity does not match the request")
    if result.get("request_digest") != digest(group_request):
        raise ServiceRolloutError("group result digest does not match the request")
    slots = result.get("slots")
    expected = [slot["slot_id"] for slot in group_request["slots"]]
    if (not isinstance(slots, list) or len(slots) != len(expected) or len(slots) < 2
            or [slot.get("slot_id") for slot in slots] != expected):
        raise ServiceRolloutError("group result is incomplete or reordered")
    sampling_sha = digest(canonical_sampling(group_request["sampling"]))
    trajectories = []
    for descriptor, slot in zip(group_request["slots"], slots):
        if slot.get("sampling_digest") != sampling_sha:
            raise ServiceRolloutError(f"slot {descriptor['slot_id']} sampling digest differs from the request")
        path = Path(slot["manifest_path"])
        try:
            payload = read_json(path)
        except (OSError, ValueError) as exc:
            raise ServiceRolloutError(f"cannot read slot manifest: {exc}") from exc
        if digest(payload) != slot.get("manifest_sha256"):
            raise ServiceRolloutError(f"slot {descriptor['slot_id']} manifest digest mismatch")
        identity = payload.get("identity", {})
        if identity.get("sampling_digest") != sampling_sha:
            raise ServiceRolloutError(f"slot {descriptor['slot_id']} manifest sampling differs from the request")
        if (identity.get("policy_version") != group_request["policy_version"]
                or identity.get("task_digest") != group_request["task"]["task_digest"]
                or identity.get("run_id") != run_id
                or identity.get("group_id") != group_request["group_id"]
                or identity.get("slot_id") != descriptor["slot_id"]):
            raise ServiceRolloutError(
                f"slot {descriptor['slot_id']} identity differs from the request")
        try:
            require_ready(payload, allow_synthetic=allow_synthetic)
        except ValueError as exc:
            raise ServiceRolloutError(f"slot {descriptor['slot_id']}: {exc}") from exc
        trajectories.append(payload)
    return trajectories
