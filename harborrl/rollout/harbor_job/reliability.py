"""Reliability control plane for Sandbox rollout failures.

The ledger is independent from rewards and trajectory exports. It records
infrastructure decisions so platform recovery can be audited without turning
partial attempts into training data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ATTEMPT_STATES = frozenset({
    "reserved", "creating", "allocated", "starting", "ready", "executing",
    "completed", "failed", "cleaning", "verified_absent",
})
FAILURE_KINDS = frozenset({
    "create-deadline", "node-unhealthy", "schedule-pressure", "startup-timeout",
    "runtime-fault", "transient-io", "storage-fault", "cleanup-fault",
    "cancel-fault", "task-invalid", "unknown",
})
_CREATE_DEADLINE = re.compile(
    r"FailedCreatePodSandBox|DeadlineExceeded|pod name reserved|CRI-O timeout", re.IGNORECASE)
_NODE = re.compile(r"\b(?:node|sbx)-[a-z0-9-]+\b", re.IGNORECASE)
_NO_SPACES = re.compile(r"\s+")


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _event_digest(event):
    """Hash all semantic fields, excluding the digest itself."""
    encoded = json.dumps({key: value for key, value in event.items()
                          if key != "event_sha256"},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sanitize_evidence(value):
    if isinstance(value, dict):
        return {str(key): ('<redacted>' if re.search(
            r'(?:token|credential|password|secret|authorization|api[_-]?key)', str(key), re.IGNORECASE)
            else sanitize_evidence(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_evidence(item) for item in value]
    if isinstance(value, str):
        text = re.sub(r"https?://\S+", "<url>", value)
        text = re.sub(r"\b(?:token|credential|authorization|api[_-]?key)\s*[:=]\s*\S+", "<redacted>", text, flags=re.IGNORECASE)
        return _NO_SPACES.sub(" ", text).strip()[:2048]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def classify_failure(stage, error, *, evidence=None):
    stage = str(stage or "unknown").lower()
    if stage == 'protocol':
        return 'task-invalid'
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    if isinstance(evidence, dict):
        text += " " + json.dumps(evidence, sort_keys=True, default=str)
    elif evidence:
        text += " " + str(evidence)
    text = sanitize_evidence(text)
    lowered = text.lower()
    if any(word in lowered for word in ("enospc", "no space left", "read-back mismatch")):
        return "storage-fault"
    if _CREATE_DEADLINE.search(lowered):
        return "create-deadline"
    insufficient = "failedscheduling" in lowered or (
        "insufficient" in lowered and any(word in lowered for word in ("cpu", "memory", "device")))
    if insufficient:
        return "schedule-pressure"
    if any(word in lowered for word in ("incomplete read", "connection reset", "download 500", "transient io")):
        return "transient-io"
    if stage in {"create", "allocation"} and any(word in lowered for word in ("503", "504", "deadline", "timeout")):
        return "create-deadline"
    if stage in {"startup", "readiness", "install"}:
        return "startup-timeout"
    if stage in {"cleanup", "delete", "reconcile"}:
        return "cleanup-fault"
    if stage == "cancel":
        return "cancel-fault"
    if stage == "storage":
        return "storage-fault"
    if "task lock" in lowered or "verifier lock" in lowered:
        return "task-invalid"
    if stage in {"execute", "runtime", "command"} or isinstance(error, RuntimeError):
        return "runtime-fault"
    return "unknown"


def extract_node(error, evidence=None):
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    if isinstance(evidence, dict):
        text += " " + json.dumps(evidence, sort_keys=True, default=str)
    match = _NODE.search(text)
    return match.group(0).lower() if match else None


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts_per_slot: int = 3
    backoff_seconds: tuple = (5.0, 15.0, 45.0)
    max_backoff_seconds: float = 120.0
    node_consecutive_failures: int = 2
    node_cooldown_seconds: float = 900.0

    def __post_init__(self):
        if type(self.max_attempts_per_slot) is not int or self.max_attempts_per_slot < 1:
            raise ValueError("max_attempts_per_slot must be a positive integer")
        if not self.backoff_seconds or any(not math.isfinite(x) or x < 0 for x in self.backoff_seconds):
            raise ValueError("backoff_seconds must contain finite non-negative values")
        if not math.isfinite(self.max_backoff_seconds) or self.max_backoff_seconds < 0:
            raise ValueError("invalid max_backoff_seconds")
        if type(self.node_consecutive_failures) is not int or self.node_consecutive_failures < 1:
            raise ValueError("node_consecutive_failures must be positive")
        if not math.isfinite(self.node_cooldown_seconds) or self.node_cooldown_seconds < 0:
            raise ValueError("invalid node_cooldown_seconds")

    def delay(self, retry_index):
        if type(retry_index) is not int or retry_index < 0:
            raise ValueError("retry_index must be non-negative")
        return min(self.backoff_seconds[min(retry_index, len(self.backoff_seconds) - 1)], self.max_backoff_seconds)


@dataclass
class InfraBudget:
    attempts: int
    create_retries: int
    cleanup_reconciliations: int
    storage_faults: int
    wall_clock_seconds: float

    def __post_init__(self):
        for name in ("attempts", "create_retries", "cleanup_reconciliations", "storage_faults"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.wall_clock_seconds) or self.wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be positive")
        self.remaining = {
            "attempts": self.attempts,
            "create_retries": self.create_retries,
            "cleanup_reconciliations": self.cleanup_reconciliations,
            "storage_faults": self.storage_faults,
        }
        self.expires_at = time.monotonic() + self.wall_clock_seconds

    def consume(self, name, amount=1):
        if name not in self.remaining:
            raise ValueError(f"unknown infrastructure budget: {name}")
        if type(amount) is not int or amount < 1:
            raise ValueError("budget amount must be positive")
        if amount > self.remaining[name] or self.expired():
            raise InfraBudgetExhausted(f"infrastructure budget exhausted: {name}")
        self.remaining[name] -= amount

    def expired(self):
        return time.monotonic() >= self.expires_at

    def exhausted(self):
        return self.expired() or any(value == 0 for value in self.remaining.values())

    def to_dict(self):
        return {
            "limits": {"attempts": self.attempts, "create_retries": self.create_retries,
                       "cleanup_reconciliations": self.cleanup_reconciliations,
                       "storage_faults": self.storage_faults,
                       "wall_clock_seconds": self.wall_clock_seconds},
            "remaining": dict(self.remaining),
            "wall_clock_expired": self.expired(),
        }


class InfraBudgetExhausted(RuntimeError):
    pass


class RetryableRolloutFailure(RuntimeError):
    """Infrastructure attempt exhaustion after independently completed cleanup."""


class GatewayTransportFailure(RuntimeError):
    """Trusted Gateway transport failure with otherwise valid collected evidence."""


class CleanupUnresolved(RuntimeError):
    """Admission must remain closed until independent reconciliation succeeds."""


class ReliabilityLedger:
    """Append-only event journal for one run or seed."""

    def __init__(self, root, *, policy=None, budget=None, clock=time.monotonic):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self.policy = policy or RetryPolicy()
        self.budget = budget
        self.clock = clock
        self._sequence = 0
        self._nodes = defaultdict(list)
        self._cooldowns = {}
        self._last_event_sha256 = None
        last_budget = None
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                self._sequence = max(self._sequence, int(event["sequence"]))
                if event.get("previous_event_sha256") != self._last_event_sha256:
                    raise ValueError("event hash chain is not append-only")
                expected = event.get("event_sha256")
                if expected != _event_digest(event):
                    raise ValueError("event hash does not match content")
                self._last_event_sha256 = expected
                last_budget = event.get('budget')
                if event.get("node") and event.get("kind") in FAILURE_KINDS:
                    self._record_node(event["node"], float(event.get("monotonic_time", 0)))
        if budget is not None:
            from harborrl.trajectories.native import publish, read_json

            path = self.root / 'budget-lock.json'
            limits = budget.to_dict()['limits']
            if path.exists():
                locked = read_json(path)
                if locked['limits'] != limits:
                    raise ValueError('seed infrastructure budget changed; use a fresh protocol')
            else:
                locked = {'limits': limits, 'expires_utc_epoch': time.time() + budget.wall_clock_seconds}
                publish(path, locked)
            budget.expires_at = time.monotonic() + locked['expires_utc_epoch'] - time.time()
            if last_budget is not None:
                if last_budget['limits'] != limits:
                    raise ValueError('seed infrastructure event budget differs from lock')
                budget.remaining = dict(last_budget['remaining'])
        elif last_budget is not None or (self.root / 'budget-lock.json').exists():
            raise ValueError('cannot reopen a budgeted seed without its infrastructure budget')

    def event(self, attempt_id, state, *, stage="", kind="", node=None,
              sandbox_id=None, decision="", detail=None):
        if state not in ATTEMPT_STATES:
            raise ValueError(f"unknown attempt state: {state}")
        if kind and kind not in FAILURE_KINDS:
            raise ValueError(f"unknown failure kind: {kind}")
        self._sequence += 1
        event = {
            "sequence": self._sequence,
            "time_utc": _utc_now(),
            "monotonic_time": self.clock(),
            "attempt_id": str(attempt_id),
            "state": state,
            "stage": str(stage or ""),
            "kind": str(kind or ""),
            "node": str(node).lower() if node else None,
            "sandbox_id": str(sandbox_id) if sandbox_id else None,
            "decision": str(decision or ""),
            "budget": self.budget.to_dict() if self.budget else None,
            "detail": sanitize_evidence(detail),
        }
        event["previous_event_sha256"] = self._last_event_sha256
        event["event_sha256"] = _event_digest(event)
        self._last_event_sha256 = event["event_sha256"]
        encoded = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def _record_node(self, node, timestamp):
        failures = [stamp for stamp in self._nodes[node] if timestamp - stamp < self.policy.node_cooldown_seconds]
        failures.append(timestamp)
        self._nodes[node] = failures
        if len(failures) >= self.policy.node_consecutive_failures:
            self._cooldowns[node] = timestamp + self.policy.node_cooldown_seconds
            return True
        return False

    def observe_failure(self, attempt_id, stage, error, *, evidence=None,
                        sandbox_id=None, retry_index=0):
        kind = classify_failure(stage, error, evidence=evidence)
        node = extract_node(error, evidence)
        delay = self.policy.delay(retry_index)
        node_cooled = self._record_node(node, self.clock()) if node else False
        decision = "fail-seed" if kind == 'task-invalid' else "retry"
        budget_error = None
        if self.budget:
            try:
                if kind == "storage-fault":
                    self.budget.consume("storage_faults")
                elif kind == "cleanup-fault":
                    self.budget.consume("cleanup_reconciliations")
                elif stage in {"create", "allocation"}:
                    self.budget.consume("create_retries")
                if self.budget.expired():
                    raise InfraBudgetExhausted('infrastructure wall clock exhausted')
            except InfraBudgetExhausted as exc:
                decision = "fail-seed"
                budget_error = exc
        self.event(attempt_id, "failed", stage=stage, kind=kind, node=node,
                   sandbox_id=sandbox_id, decision=decision,
                   detail={"error": error if isinstance(error, str) else f"{type(error).__name__}: {error}",
                           "evidence": evidence,
                           "retry_delay_s": delay,
                           "node_cooldown": node_cooled})
        if budget_error is not None:
            raise budget_error
        return kind, node, delay if decision == "retry" else None

    def node_available(self, node):
        return not node or self.clock() >= self._cooldowns.get(node, 0)

    def next_node(self, candidates, preferred=None):
        available = [node for node in candidates if self.node_available(node)]
        return preferred if preferred in available else (available[0] if available else None)

    def reserve(self, attempt_id, *, task_digest, policy_version, retry_index, node=None):
        if self.budget:
            self.budget.consume("attempts")
        return self.event(attempt_id, "reserved", stage="attempt", node=node,
                          decision="start",
                          detail={"task_digest": task_digest, "policy_version": policy_version,
                                  "retry_index": retry_index})


def summarize_reliability(root):
    path = Path(root) / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [event["sequence"] for event in events] != list(range(1, len(events) + 1)):
        raise ValueError("event sequence is not append-only")
    previous = None
    for event in events:
        if event.get("previous_event_sha256") != previous or event.get("event_sha256") != _event_digest(event):
            raise ValueError("event hash chain is not append-only")
        previous = event["event_sha256"]
    attempts = defaultdict(list)
    for event in events:
        attempts[event["attempt_id"]].append(event)
    states = {attempt_id: items[-1]["state"] for attempt_id, items in attempts.items()}
    kinds = Counter(event["kind"] for event in events if event["kind"])
    completed = sum(any(event['state'] == 'completed' for event in items)
                    for items in attempts.values())
    failed = sum(any(event['state'] == 'failed' for event in items)
                 for items in attempts.values())
    cleaned = sum(state == "verified_absent" for state in states.values())
    return {
        "events": len(events),
        "attempts": len(attempts),
        "completed_attempts": completed,
        "failed_attempts": failed,
        "verified_absent_attempts": cleaned,
        "terminal_attempts": sum(any(event['state'] in {'completed', 'failed', 'verified_absent'}
                                     for event in items) for items in attempts.values()),
        "failure_counts": dict(sorted(kinds.items())),
        "retry_decisions": sum(bool(event.get("kind")) and event.get("decision") == "retry"
                               for event in events),
        "fail_seed_decisions": sum(event.get("decision") == "fail-seed" for event in events),
        "node_cooldown_events": sum(
            bool((event.get("detail") or {}).get("node_cooldown")) for event in events),
    }
