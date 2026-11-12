"""Per-trial collection. Reward artifacts audit, but never replace, terminal results."""
import hashlib
from pathlib import Path

from harborrl.trajectories.native import publish, read_json


def collect(identity, trial_id, result_path, verifier_dir, profile, *, evidence_kind="harbor",
            termination_evidence=None):
    if profile.digest != identity.reward_digest:
        raise ValueError("reward profile differs from attempt manifest")
    result_path, verifier_dir = Path(result_path), Path(verifier_dir)
    # Caller retries incomplete files; no receipt is published before this succeeds.
    result = read_json(result_path)
    if result.get("id") != trial_id or not result.get("finished_at"):
        raise ValueError("missing terminal result or mismatched trial")
    if result.get("step_results") or result.get("verifier_environment_mode") != "shared":
        raise ValueError("only single-step shared verifier is enabled")
    result_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if result.get("exception_info"):
        if (not isinstance(termination_evidence, dict)
                or termination_evidence.get('profile') != 'claude_code_2.1.141_max_turns_v1'
                or termination_evidence.get('source_result_digest') != result_digest
                or result['exception_info'].get('exception_type') != 'NonZeroAgentExitCodeError'):
            raise ValueError("exceptional trial requires a separately verified termination profile")
    elif termination_evidence is not None:
        raise ValueError('termination evidence contradicts normal trial result')
    rewards = (result.get("verifier_result") or {}).get("rewards")
    artifact = verifier_dir / "reward.json"
    if artifact.exists():
        artifact_rewards = read_json(artifact)
    else:
        artifact = verifier_dir / "reward.txt"
        artifact_rewards = {"reward": float(artifact.read_text().strip())}
    resolved = profile.resolve(rewards)
    profile.resolve(artifact_rewards)  # reject bool / nonfinite before numerical comparison
    if rewards != artifact_rewards:
        raise ValueError("reward artifact and terminal result differ")
    return {**resolved, "identity": identity.to_dict(), "harbor_trial_id": trial_id,
            "generation_attempt_id": identity.attempt_id, "evaluation_attempt_id": identity.attempt_id,
            "evaluation_trial_id": trial_id, "source_trial_id": trial_id,
            "verifier_environment_mode": "shared", "evaluation_status": "valid",
            "evidence_kind": evidence_kind, "source_result_ref": str(result_path),
            "source_result_digest": result_digest,
            "source_artifact_ref": str(artifact),
            "source_artifact_digest": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            **({'termination_evidence': termination_evidence} if termination_evidence else {})}


def commit_receipt(path, *args, **kwargs):
    receipt = collect(*args, **kwargs)
    publish(path, receipt)
    return receipt
