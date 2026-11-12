"""Remote gateway shards: the local shard surface, served over admin HTTP.

A remote shard is an independent process (usually on another machine) that
owns one TraceRegistry and one gateway server. Generation traffic never
crosses this module: runners talk to the shard's gateway URL directly with
their attempt credentials, exactly as with a local shard. Only the
manager-side registry surface — admission epochs, registration, delivery
waits, session audits, sealing — crosses the admin channel, authenticated
by the run token.

Fail-closed throughout: a transport failure surfaces as OSError so the
scheduler fails exactly this shard's attempts, and an unreachable shard
reads as a closed admission (version unknown, not accepting) until the
next activation republishes a consistent epoch. A restarted-but-reachable
shard instead reports no version — the facade surfaces that disagreement
as an unknown epoch, so the service rejects new groups until the next
activation — while its fresh registry still accepts and enforces
settle-before-republish for the attempts it owns.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from urllib import error as urlerror
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

# Manager<->shard traffic is direct LAN HTTP; never route it through a proxy.
_OPENER = build_opener(ProxyHandler({}))
_REMOTE_ERRORS = {"ValueError": ValueError, "TimeoutError": TimeoutError,
                  "PermissionError": PermissionError, "KeyError": KeyError,
                  "TypeError": TypeError, "RuntimeError": RuntimeError}


class RemoteShardError(OSError):
    """The remote shard answered with an unmapped failure."""


def plain_origin(url, label):
    """Validate and normalize a plain http(s) origin."""
    parts = urlsplit(url if isinstance(url, str) else "")
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in ("", "/")):
        raise ValueError(f"{label} must be a plain http(s) origin")
    return f"{parts.scheme}://{parts.netloc}".rstrip("/")


def read_shard_token(path):
    """Load the run token a remote shard and its manager must share."""
    token = Path(path).read_text().strip()
    if not token:
        raise ValueError(f"gateway shard token file is empty: {path}")
    return token


class RemoteRegistry:
    """The manager-side TraceRegistry surface of one remote shard.

    ``version`` and ``accepting`` are live reads, not a cache: admission
    decisions never trust manager-local memory about a process that can
    crash and restart between two group submissions.
    """

    def __init__(self, admin_url, token, *, timeout=30.0, status_timeout=5.0):
        self.admin_url = plain_origin(admin_url, "shard admin_url")
        if not isinstance(token, str) or not token:
            raise ValueError("remote shard registry requires the run token")
        self.token = token
        self.timeout = float(timeout)
        self.status_timeout = float(status_timeout)

    # -- transport ---------------------------------------------------------

    def request(self, method, path, payload=None, *, timeout=None):
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"x-api-key": self.token}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.admin_url}{path}", data=data,
                          headers=headers, method=method)
        try:
            with _OPENER.open(request, timeout=timeout or self.timeout) as response:
                body = response.read()
        except urlerror.HTTPError as exc:
            if exc.code == 401:
                raise PermissionError("remote shard rejected the run token") from exc
            raise
        return json.loads(body) if body else None

    def _call(self, action, payload=None, *, timeout=None):
        body = self.request("POST", f"/registry/{action}", payload, timeout=timeout)
        if isinstance(body, dict) and body.get("ok"):
            return body.get("result")
        error = (body or {}).get("error") or {}
        raise _REMOTE_ERRORS.get(error.get("type"), RemoteShardError)(
            error.get("message") or "remote shard call failed")

    def status(self):
        return self.request("GET", "/status", timeout=self.status_timeout) or {}

    # -- admission epoch ---------------------------------------------------

    @property
    def version(self):
        try:
            return self.status().get("version")
        except (OSError, ValueError):
            return None

    @property
    def accepting(self):
        try:
            return bool(self.status().get("accepting"))
        except (OSError, ValueError):
            return False

    def close_admission(self):
        self._call("close_admission")

    def drain(self, timeout):
        timeout = float(timeout)
        self._call("drain", {"timeout": timeout}, timeout=timeout + 15)

    def publish_version(self, states, expected):
        self._call("publish_version", {"states": states, "expected": str(expected)})

    # -- per-attempt surface -----------------------------------------------

    def register(self, identity, trial_id, *, credential=None):
        result = self._call("register", {"identity": identity.to_dict(),
                                         "trial_id": trial_id, "credential": credential})
        return (result or {}).get("credential")

    def close(self, attempt_id):
        self._call("close", {"attempt_id": attempt_id})

    def wait_delivered(self, attempt_id, timeout):
        timeout = float(timeout)
        self._call("wait_delivered", {"attempt_id": attempt_id, "timeout": timeout},
                   timeout=timeout + 15)

    def unconsumed_responses(self, attempt_id):
        return self._call("unconsumed_responses", {"attempt_id": attempt_id}) or []

    def consume(self, attempt_id, response_id, content, evidence_ref, *, cli_version=None):
        self._call("consume", {"attempt_id": attempt_id, "response_id": response_id,
                               "content": content, "evidence_ref": evidence_ref,
                               "cli_version": cli_version})

    def audit_tool_rejections(self, attempt_id, results, evidence_ref):
        self._call("audit_tool_rejections", {"attempt_id": attempt_id, "results": results,
                                             "evidence_ref": evidence_ref})

    def seal(self, attempt_id):
        result = self._call("seal", {"attempt_id": attempt_id})
        return result["turns"], result["seal"]


class RemoteShard:
    """One gateway shard served by an independent process.

    Presents the GatewayShard surface — ``name``, ``url``, ``port``,
    ``registry``, ``start``/``close`` — so scheduling and the ShardAdmission
    barrier never learn which form a shard takes. ``start()`` proves the
    process is reachable and answers as this shard; ``close()`` asks it to
    shut down, best-effort, because the shard process owns its lifecycle.
    """

    def __init__(self, spec, *, token, trace_dir, timeout=30.0):
        self.spec = spec
        self.trace_dir = Path(trace_dir)
        self.registry = RemoteRegistry(spec.admin_url, token, timeout=timeout)

    @property
    def name(self):
        return self.spec.name

    @property
    def url(self):
        return self.spec.url

    @property
    def port(self):
        return urlsplit(self.spec.url).port

    @property
    def admin_url(self):
        return self.spec.admin_url

    def start(self):
        status = self.registry.status()
        if status.get("shard") != self.spec.name:
            raise RuntimeError(
                f"gateway shard {self.spec.name!r} at {self.spec.admin_url} "
                f"answers as {status.get('shard')!r}")

    def close(self):
        with contextlib.suppress(OSError, ValueError):
            self.registry.request("POST", "/shutdown", timeout=5.0)
