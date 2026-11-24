import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from harborrl.rollout.harbor_job.reliability import (
    InfraBudget,
    InfraBudgetExhausted,
    ReliabilityLedger,
    RetryPolicy,
    classify_failure,
    sanitize_evidence,
    summarize_reliability,
)
from harborrl.rollout.runtime import _failure_stage, _run_slot


class RetrySlots:
    max_attempts = 2

    def __init__(self, identities):
        self.identities = identities
        self.history = []

    def start(self, slot):
        return self.identities[min(self._starts, len(self.identities) - 1)]

    _starts = 0

    def finish(self, slot, ir=None, *, error=None):
        assert (ir is None) != (error is None)
        self.history.append((slot, "valid" if ir is not None else "invalid"))
        if error is not None:
            self._starts += 1


def test_classifier_covers_phase6_faults_and_sanitizes_evidence():
    evidence = {
        "event": "node sbx-011 failed",
        "url": "https://secret.example/path?token=value",
    }
    cases = [
        ("create", "FailedCreatePodSandBox/DeadlineExceeded", "create-deadline"),
        ("startup", "agent not ready", "startup-timeout"),
        ("execute", "command exited 137", "runtime-fault"),
        ("cleanup", "GET did not become 404", "cleanup-fault"),
        ("storage", "write read-back mismatch", "storage-fault"),
    ]
    for stage, error, expected in cases:
        assert classify_failure(stage, error, evidence=evidence) == expected
    sanitized = json.dumps(sanitize_evidence(evidence))
    assert "secret.example" not in sanitized and "token=value" not in sanitized


def test_failure_events_are_append_only_and_summarized(tmp_path):
    ledger = ReliabilityLedger(tmp_path)
    ledger.reserve("a0", task_digest="d" * 64, policy_version="1",
                   retry_index=0, node="node-a")
    ledger.event("a0", "creating", stage="create", node="node-a", decision="retry")
    kind, node, delay = ledger.observe_failure(
        "a0", "create", "FailedCreatePodSandBox node node-a")

    assert (kind, node, delay) == ("create-deadline", "node-a", 5.0)
    ledger.reserve("a1", task_digest="d" * 64, policy_version="1",
                   retry_index=1, node="node-b")
    ledger.event("a1", "completed", stage="execute", decision="complete")
    assert summarize_reliability(tmp_path) == {
        "events": 5,
        "attempts": 2,
        "completed_attempts": 1,
        "failed_attempts": 1,
        "verified_absent_attempts": 0,
        "terminal_attempts": 2,
        "failure_counts": {"create-deadline": 1},
        "retry_decisions": 1,
        "fail_seed_decisions": 0,
        "node_cooldown_events": 0,
    }


def test_event_journal_rejects_history_rewrites(tmp_path):
    ledger = ReliabilityLedger(tmp_path)
    ledger.reserve("a0", task_digest="d" * 64, policy_version="1", retry_index=0)
    path = tmp_path / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["state"] = "completed"
    lines[0] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="append-only"):
        summarize_reliability(tmp_path)


def test_node_cooling_selects_another_candidate(tmp_path):
    now = [100.0]
    ledger = ReliabilityLedger(tmp_path, policy=RetryPolicy(
        node_consecutive_failures=2, node_cooldown_seconds=900), clock=lambda: now[0])
    ledger.observe_failure("a0", "create", "CRI-O timeout node sbx-011")
    ledger.observe_failure("a1", "create", "FailedCreatePodSandBox node sbx-011")

    assert ledger.next_node(["sbx-011", "sbx-014"]) == "sbx-014"
    assert summarize_reliability(tmp_path)["node_cooldown_events"] == 1
    now[0] += 900
    assert ledger.next_node(["sbx-011", "sbx-014"], preferred="sbx-011") == "sbx-011"


@pytest.mark.parametrize("lifecycle_state,startup_stage,expected", [
    ("creation_unknown", None, "create"),
    ("allocated", None, "startup"),
    ("created", "readiness", "readiness"),
    ("delete_failed", None, "cleanup"),
    ("deleted", None, "execute"),
])
def test_failure_stage_uses_sanitized_runner_evidence(
        tmp_path, lifecycle_state, startup_stage, expected):
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "sandbox-lifecycle.json").write_text(
        json.dumps({"state": lifecycle_state, "sandbox_id": "sbx-test"}), encoding="utf-8")
    if startup_stage:
        (runner / "sandbox-startup-failed.json").write_text(
            json.dumps({"stage": startup_stage}), encoding="utf-8")
    assert _failure_stage(tmp_path, sandbox=True) == expected


def test_infrastructure_budget_bounds_create_retries(tmp_path):
    ledger = ReliabilityLedger(tmp_path, budget=InfraBudget(
        attempts=4, create_retries=1, cleanup_reconciliations=1,
        storage_faults=1, wall_clock_seconds=3600))
    ledger.reserve("a0", task_digest="d" * 64, policy_version="1", retry_index=0)
    kind, _, delay = ledger.observe_failure("a0", "create", "endpoint 503")
    assert (kind, delay) == ("create-deadline", 5.0)

    ledger.reserve("a1", task_digest="d" * 64, policy_version="1", retry_index=1)
    with pytest.raises(InfraBudgetExhausted, match="create_retries"):
        ledger.observe_failure("a1", "create", "endpoint 504")


def _reliability_runtime_config(tmp_path):
    from types import SimpleNamespace

    return SimpleNamespace(
        max_attempts=2,
        task_worker_overrides={},
        runner_transport="local",
        workers=(),
        sandbox=None,
        gateway_routing="modulo",
        gateway_url="http://gateway",
        runner_python=sys.executable,
        max_output_tokens=128,
    )


def test_runtime_failure_retries_as_a_new_ledger_attempt(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as runtime_module

    policy = RetryPolicy(backoff_seconds=(0, 0), node_consecutive_failures=100)
    ledger = ReliabilityLedger(tmp_path, policy=policy)
    identities = [
        SimpleNamespace(attempt_id=f"attempt-{index}", group_id="group-0",
                        slot_id="0", task_digest="d" * 64, policy_version="1")
        for index in range(2)
    ]
    slots = RetrySlots(identities)
    calls = []

    async def attempt(identity, **kwargs):
        calls.append(identity.attempt_id)
        kwargs["root"].mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            raise RuntimeError("synthetic runtime failure")
        return {"identity": identity.__dict__}

    monkeypatch.setattr(runtime_module, "run_attempt", attempt)
    config = _reliability_runtime_config(tmp_path)
    current = SimpleNamespace(rollout_root=tmp_path, semaphore=asyncio.Semaphore(1),
                               profile={}, registry=None, reliability=ledger,
                               shards=(SimpleNamespace(url=config.gateway_url, registry=None),))
    result = asyncio.run(_run_slot(
        config, current, slots, 0,
        {"id": "task", "path": "/task", "task_digest": "d" * 64}, None))

    assert result == {"identity": identities[1].__dict__}
    assert calls == ["attempt-0", "attempt-1"]
    assert slots.history == [("0", "invalid"), ("0", "valid")]
    summary = summarize_reliability(tmp_path)
    assert summary == {
        "events": 6,
        "attempts": 2,
        "completed_attempts": 1,
        "failed_attempts": 1,
        "verified_absent_attempts": 0,
        "terminal_attempts": 2,
        "failure_counts": {"runtime-fault": 1},
        "retry_decisions": 1,
        "fail_seed_decisions": 0,
        "node_cooldown_events": 0,
    }


def test_runtime_budget_exhaustion_fails_seed_without_next_attempt(tmp_path, monkeypatch):
    from harborrl.rollout import runtime as runtime_module

    ledger = ReliabilityLedger(tmp_path, policy=RetryPolicy(
        backoff_seconds=(0, 0)), budget=InfraBudget(
        attempts=1, create_retries=1, cleanup_reconciliations=1,
        storage_faults=1, wall_clock_seconds=3600))
    identities = [SimpleNamespace(attempt_id="attempt-0", group_id="group-0",
                                  slot_id="0", task_digest="d" * 64, policy_version="1")]
    slots = RetrySlots(identities)
    calls = []

    async def attempt(identity, **kwargs):
        calls.append(identity.attempt_id)
        raise RuntimeError("synthetic runtime failure")

    monkeypatch.setattr(runtime_module, "run_attempt", attempt)
    config = _reliability_runtime_config(tmp_path)
    current = SimpleNamespace(rollout_root=tmp_path, semaphore=asyncio.Semaphore(1),
                              profile={}, registry=None, reliability=ledger,
                              shards=(SimpleNamespace(url=config.gateway_url, registry=None),))
    with pytest.raises(InfraBudgetExhausted, match="attempts"):
        asyncio.run(_run_slot(config, current, slots, 0,
                              {"id": "task", "path": "/task",
                               "task_digest": "d" * 64}, None))

    assert calls == ["attempt-0"]
    summary = summarize_reliability(tmp_path)
    assert summary["attempts"] == 1
    assert summary["fail_seed_decisions"] == 1
    assert slots.history == [("0", "invalid"), ("0", "invalid")]


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX fsynced budget lock')
def test_restart_preserves_seed_budget_and_rejects_changed_limits(tmp_path):
    def budget(attempts=2):
        return InfraBudget(attempts=attempts, create_retries=1, cleanup_reconciliations=1,
                           storage_faults=1, wall_clock_seconds=3600)
    first = ReliabilityLedger(tmp_path, budget=budget())
    first.reserve('a0', task_digest='d' * 64, policy_version='1', retry_index=0)
    restarted = ReliabilityLedger(tmp_path, budget=budget())
    assert restarted.budget.remaining['attempts'] == 1
    assert abs(restarted.budget.expires_at - first.budget.expires_at) < .1
    restarted.reserve('a1', task_digest='d' * 64, policy_version='1', retry_index=1)
    exhausted = ReliabilityLedger(tmp_path, budget=budget())
    with pytest.raises(InfraBudgetExhausted):
        exhausted.reserve('a2', task_digest='d' * 64, policy_version='1', retry_index=2)
    with pytest.raises(ValueError, match='budget changed'):
        ReliabilityLedger(tmp_path, budget=budget(3))


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX fsynced budget lock')
def test_locked_seed_cannot_remove_budget_before_first_reservation(tmp_path):
    ReliabilityLedger(tmp_path, budget=InfraBudget(
        attempts=2, create_retries=1, cleanup_reconciliations=1,
        storage_faults=1, wall_clock_seconds=3600))
    with pytest.raises(ValueError, match='without its infrastructure budget'):
        ReliabilityLedger(tmp_path)
