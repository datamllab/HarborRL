"""Strict, runtime-independent contracts for native trajectories (IR v2)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical_sampling(sampling):
    """Give equivalent JSON probabilities one representation before hashing."""
    result = dict(sampling)
    for key in ("temperature", "top_p"):
        if not finite(result[key]):
            raise ValueError(f"sampling {key} must be a finite number")
        result[key] = float(result[key])
    return result


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), parse_constant=reject, object_pairs_hook=unique)


def publish(path, value):
    """Publish once, idempotently; concurrent conflicting writers never overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encode(value) + b"\n"
    fd, temp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"conflicting immutable record: {path.name}")
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        os.unlink(temp)
    return path


@dataclass(frozen=True)
class Identity:
    run_id: str
    batch_id: str
    group_id: str
    slot_id: str
    attempt_id: str
    trajectory_id: str
    task_digest: str
    policy_version: str
    harness_digest: str
    reward_digest: str
    sampling_digest: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise ValueError("identity fields must be nonempty strings")

    def to_dict(self):
        return asdict(self)

    @property
    def group_key(self):
        value = self.to_dict()
        for key in ("slot_id", "attempt_id", "trajectory_id"):
            value.pop(key)
        return digest(value)


@dataclass(frozen=True)
class RewardProfile:
    key: str = "reward"
    scale: float = 1.0
    offset: float = 0.0
    raw_range: tuple[float, float] | None = (0.0, 1.0)

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("reward key is required")
        if not finite(self.scale) or not finite(self.offset):
            raise ValueError("reward transform must be finite")
        if self.raw_range is not None:
            if (len(self.raw_range) != 2 or not all(finite(x) for x in self.raw_range)
                    or self.raw_range[0] > self.raw_range[1]):
                raise ValueError("invalid raw reward range")
            object.__setattr__(self, "raw_range", tuple(self.raw_range))

    @property
    def digest(self):
        return digest(asdict(self))

    def resolve(self, rewards):
        if not isinstance(rewards, dict) or not rewards or any(not finite(v) for v in rewards.values()):
            raise ValueError("all raw rewards must be finite numbers, not bools")
        if self.key not in rewards:
            raise ValueError(f"missing selected reward key: {self.key}")
        raw = rewards[self.key]
        if self.raw_range is not None and not self.raw_range[0] <= raw <= self.raw_range[1]:
            raise ValueError("reward outside declared range")
        training = self.scale * raw + self.offset
        if not finite(training):
            raise ValueError("nonfinite transformed reward")
        return {"raw_rewards": rewards, "selected_key": self.key, "raw_reward": raw,
                "training_reward": training, "reward_profile": json.loads(encode(asdict(self))),
                "reward_profile_digest": self.digest}


def validate_turn(turn, identity, trial_id):
    errors = []
    for key in ("request_id", "response_id", "engine_id", "tokenizer_digest", "template_digest"):
        if not isinstance(turn.get(key), str) or not turn[key]:
            errors.append(key)
    if turn.get("identity") != identity.to_dict() or turn.get("harbor_trial_id") != trial_id:
        errors.append("turn_identity")
    for key in ("input_ids", "output_ids"):
        ids = turn.get(key)
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            errors.append(key)
    logps = turn.get("logprobs")
    if (not isinstance(logps, list) or not isinstance(turn.get("output_ids"), list)
            or len(logps) != len(turn["output_ids"]) or any(not finite(x) or x > 0 for x in logps)):
        errors.append("logprobs")
    if turn.get("logprob_semantics") != "raw_model":
        errors.append("logprob_semantics")
    if turn.get("policy_version") != identity.policy_version:
        errors.append("policy_version")
    if turn.get("delivery") != "consumed_confirmed" or not turn.get("consumption_ref"):
        errors.append("consumption")
    if turn.get("role") != "policy":
        errors.append("generation_role")
    if not isinstance(turn.get("sampling"), dict) or digest(turn["sampling"]) != identity.sampling_digest:
        errors.append("sampling")
    for key in ("client_request", "serving_input", "response"):
        if key not in turn or turn.get(key + "_digest") != digest(turn[key]):
            errors.append(key + "_digest")
    declared = {tool['name'] for tool in turn.get('client_request', {}).get('tools', [])}
    if any(block.get('type') == 'tool_use' and block.get('name') not in declared
           for block in turn.get('response', {}).get('content', [])) and not turn.get('tool_rejection_ref'):
        errors.append('unconfirmed_tool_rejection')
    if turn.get("finish_reason") not in ("end_turn", "tool_use", "max_tokens", "stop_sequence"):
        errors.append("finish_reason")
    return errors


def validate_v2(ir):
    """Recompute eligibility; never trust a caller-supplied readiness flag."""
    errors = []
    try:
        identity = Identity(**ir["identity"])
        trial = ir["harbor_trial_id"]
        if ir.get("schema_version") != "2" or not isinstance(trial, str) or not trial:
            errors.append("schema_or_trial")
        receipt, seal, turns = ir["evaluation"], ir["trace_seal"], ir["turns"]
        if receipt.get("identity") != identity.to_dict() or receipt.get("harbor_trial_id") != trial:
            errors.append("evaluation_identity")
        if receipt.get("evaluation_status") != "valid":
            errors.append("evaluation_status")
        profile = RewardProfile(**receipt["reward_profile"])
        resolved = profile.resolve(receipt["raw_rewards"])
        if profile.digest != identity.reward_digest or any(receipt.get(k) != v for k, v in resolved.items()):
            errors.append("reward_recompute")
        if not receipt.get("source_result_digest") or not receipt.get("source_artifact_digest"):
            errors.append("evaluation_evidence")
        if seal.get("identity") != identity.to_dict() or seal.get("harbor_trial_id") != trial:
            errors.append("seal_identity")
        if seal.get("pending_requests") != 0 or seal.get("status") != "sealed" or seal.get("errors"):
            errors.append("trace_not_sealed")
        if not isinstance(turns, list) or not turns:
            errors.append("empty_turns")
            return errors
        request_ids = [t["request_id"] for t in turns]
        if (len(set(request_ids)) != len(request_ids) or seal.get("request_ids") != request_ids
                or seal.get("turns_digest") != digest(turns)):
            errors.append("request_completeness")
        if len({t["response_id"] for t in turns}) != len(turns):
            errors.append("duplicate_response")
        for turn in turns:
            errors.extend(validate_turn(turn, identity, trial))
        if len({(t["tokenizer_digest"], t["template_digest"]) for t in turns}) != 1:
            errors.append("mixed_tokenizer_template")
        if ir.get("termination") not in ("completed", "budget_truncated"):
            errors.append("termination")
        if ir.get("termination") == "budget_truncated":
            evidence = ir.get('termination_evidence')
            if (not isinstance(evidence, dict) or evidence != receipt.get('termination_evidence')
                    or evidence.get('source_result_digest') != receipt.get('source_result_digest')
                    or evidence.get('consumed_response_ids') != [t['response_id'] for t in turns]):
                errors.append("truncation_evidence")
        elif ir.get('termination_evidence') is not None or receipt.get('termination_evidence') is not None:
            errors.append('unexpected_termination_evidence')
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        errors.append(f"malformed_contract:{type(exc).__name__}")
    return sorted(set(errors))


def assemble(identity, trial_id, turns, seal, receipt, *, termination="completed", termination_evidence=None):
    ir = {"schema_version": "2", "identity": identity.to_dict(), "harbor_trial_id": trial_id,
          "turns": turns, "trace_seal": seal, "evaluation": receipt, "termination": termination,
          "termination_evidence": termination_evidence}
    errors = validate_v2(ir)
    # Synthetic transport/serving tests are useful but never eligible for real training.
    synthetic = any(t.get("evidence_kind") != "serving" for t in turns) or receipt.get("evidence_kind") != "harbor"
    ir.update(integrity_errors=errors, readiness="EVAL_ONLY" if errors else ("CONTRACT_READY" if synthetic else "RL_READY"))
    return ir


def require_ready(ir, *, allow_synthetic=False):
    errors = validate_v2(ir)
    real = (ir.get("evaluation", {}).get("evidence_kind") == "harbor"
            and all(t.get("evidence_kind") == "serving" for t in ir.get("turns", [])))
    if not real and not allow_synthetic:
        errors.append("synthetic_evidence")
    if errors:
        raise ValueError("native IR rejected: " + ", ".join(errors))
