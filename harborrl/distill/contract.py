"""Fail-closed on-policy batch admission independent of Native RL_READY."""
from harborrl.trajectories.native import digest, finite

ROW_FIELDS = {"schema_version", "sample_id", "task_id", "task_digest", "split", "student_behavior_version",
              "student_evidence", "teacher_revision", "teacher_evidence", "identity", "prompt_ids", "response_ids",
              "behavior_logprobs", "teacher_logprobs", "loss_mask", "sampling", "stop_reason", "score_request_digest",
              "score_response_digest", "response", "verification", "prm", "digest"}


def score_request(row):
    return {key: row[key] for key in ("prompt_ids", "response_ids", "identity", "teacher_revision", "student_behavior_version")}


def validate_row(row, *, version, identity, context, allow_synthetic=False, require_teacher=True):
    if not isinstance(row, dict) or set(row) != ROW_FIELDS or row["schema_version"] != "math-onpolicy-batch-v1":
        raise ValueError("Invalid on-policy batch row fields")
    if digest({k: v for k, v in row.items() if k != "digest"}) != row["digest"]:
        raise ValueError("Batch content digest mismatch")
    if row["student_behavior_version"] != version or row["identity"] != identity:
        raise ValueError("Stale student version or tokenizer/template/vocabulary mismatch")
    if row["student_evidence"] != "observed" or row["split"] != "train":
        raise ValueError("Only observed student generation on the train split is admitted")
    if require_teacher and (not row["teacher_revision"] or row["teacher_evidence"] not in {"observed", "synthetic"}):
        raise ValueError("Missing locked teacher scoring evidence")
    if row["teacher_evidence"] == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic teacher is test-only")
    if row["prm"] and row["prm"]["evidence_kind"] == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic PRM is test-only")
    if row["sampling"] != {"temperature": 1.0, "top_p": 1.0, "raw_logprobs": True}:
        raise ValueError("Batch must use raw on-policy sampling")
    p, r = row["prompt_ids"], row["response_ids"]
    if not p or not r or len(p) + len(r) > context:
        raise ValueError("Empty or over-context sequence")
    if any(type(token) is not int or not 0 <= token < identity["vocab_size"] for token in p + r):
        raise ValueError("Invalid token ID")
    for key in ("behavior_logprobs", "teacher_logprobs", "loss_mask"):
        if len(row[key]) != len(r) and not (key == "teacher_logprobs" and not require_teacher and row[key] == []):
            raise ValueError("Token/probability/mask length mismatch")
    for key in ("behavior_logprobs", "teacher_logprobs"):
        if any(not finite(value) or value > 0 for value in row[key]):
            raise ValueError("Nonfinite or positive raw log probability")
    if row["loss_mask"] != [1] * len(r):
        raise ValueError("Only actual single-turn generated tokens enter this batch")
    verification = row["verification"]
    rewards = {"correct": 1.0, "incorrect": -1.0, "unparseable": 0.0, "unsupported": 0.0}
    if (not isinstance(verification, dict) or verification.get("status") not in rewards
            or verification.get("reward") != rewards[verification["status"]]):
        raise ValueError("Invalid verification reward")
    prm = row["prm"]
    if prm is not None:
        values = prm.get("token_returns", [])
        if len(values) != len(r) or any(not finite(value) for value in values):
            raise ValueError("Invalid PRM token returns")
        cursor, cumulative = len(r), 0.0
        for step in reversed(prm.get("steps", [])):
            if (type(step.get("start")) is not int or type(step.get("end")) is not int
                    or not 0 <= step["start"] < step["end"] == cursor or not finite(step.get("score"))):
                raise ValueError("Invalid PRM step boundary or score")
            cumulative += step["score"]
            if values[step["start"]:cursor] != [cumulative] * (cursor - step["start"]):
                raise ValueError("PRM returns disagree with step rewards")
            cursor = step["start"]
        if cursor:
            raise ValueError("PRM steps do not cover the response")
    if row["score_request_digest"] != digest(score_request(row)):
        raise ValueError("Scorer request mismatch")
    if row["score_response_digest"] != digest({"request_digest": row["score_request_digest"], "logprobs": row["teacher_logprobs"]}):
        raise ValueError("Scorer response mismatch")
    if row["stop_reason"] not in {"eos", "length"}:
        raise ValueError("Unknown stop reason")
    return row
