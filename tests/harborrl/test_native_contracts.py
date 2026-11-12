"""CPU contract evidence, explicitly distinct from live Harbor/SGLang evidence."""
import json
import math

import pytest

from harborrl.rollout.exporters.native import export_group
from harborrl.rollout.harbor_job.collector import collect
from harborrl.trajectories.native import (
    Identity,
    RewardProfile,
    assemble,
    digest,
    publish,
    read_json,
    require_ready,
    validate_v2,
)

SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 64, "stop": []}


def identity(slot="0", **overrides):
    values = {"run_id": "run", "batch_id": "batch", "group_id": "group", "slot_id": slot,
              "attempt_id": "attempt-" + slot, "trajectory_id": "trajectory-" + slot,
              "task_digest": "task", "policy_version": "1", "harness_digest": "claude",
              "reward_digest": RewardProfile().digest, "sampling_digest": digest(SAMPLING)}
    return Identity(**{**values, **overrides})


def receipt(tmp_path, ident, reward, **result_changes):
    root = tmp_path / ident.attempt_id
    root.mkdir(parents=True, exist_ok=True)
    result = {"id": "trial-" + ident.attempt_id, "finished_at": "2026-09-20T00:00:00Z", "verifier_environment_mode": "shared",
              "verifier_result": {"rewards": {"reward": reward}}, **result_changes}
    (root / "result.json").write_text(json.dumps(result))
    (root / "reward.json").write_text(json.dumps({"reward": reward}))
    return collect(ident, result["id"], root / "result.json", root, RewardProfile(), evidence_kind="synthetic")


def make_ir(tmp_path, slot="0", reward=0, nturns=1):
    ident = identity(slot)
    evaluation = receipt(tmp_path, ident, reward)
    trial = evaluation["harbor_trial_id"]
    turns = []
    for n in range(nturns):
        turn = {"identity": ident.to_dict(), "harbor_trial_id": trial, "request_id": f"r-{slot}-{n}",
                "response_id": f"m-{slot}-{n}", "engine_id": "engine", "input_ids": [1, 2], "output_ids": [3, 4],
                "logprobs": [-0.2, -0.3], "logprob_semantics": "raw_model", "policy_version": "1",
                "tokenizer_digest": "tok", "template_digest": "template", "delivery": "consumed_confirmed",
                "consumption_ref": "session#digest", "role": "policy", "sampling": SAMPLING,
                "client_request": {"model": "policy"}, "serving_input": {"input_ids": [1, 2]},
                "response": {"content": [{"type": "text", "text": "done"}]},
                "finish_reason": "end_turn", "evidence_kind": "synthetic"}
        for key in ("client_request", "serving_input", "response"):
            turn[key + "_digest"] = digest(turn[key])
        turns.append(turn)
    seal = {"identity": ident.to_dict(), "harbor_trial_id": trial, "pending_requests": 0, "status": "sealed",
            "request_ids": [t["request_id"] for t in turns], "turns_digest": digest(turns), "errors": []}
    return assemble(ident, trial, turns, seal, evaluation)


def test_roundtrip_and_synthetic_gate(tmp_path):
    ir = make_ir(tmp_path)
    assert ir["readiness"] == "CONTRACT_READY"
    path = publish(tmp_path / "ir.json", ir)
    assert validate_v2(read_json(path)) == []
    require_ready(ir, allow_synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        require_ready(ir)
    publish(path, ir)
    with pytest.raises(ValueError, match="conflicting"):
        publish(path, {**ir, "readiness": "RL_READY"})


def test_unavailable_tool_cannot_enter_training_without_native_rejection(tmp_path):
    ir = make_ir(tmp_path)
    turn = ir['turns'][0]
    turn['response']['content'] = [{'type': 'tool_use', 'id': 'u', 'name': 'Check', 'input': {}}]
    turn['response_digest'] = digest(turn['response'])
    ir['trace_seal']['turns_digest'] = digest(ir['turns'])
    with pytest.raises(ValueError, match='unconfirmed_tool_rejection'):
        require_ready(ir, allow_synthetic=True)


@pytest.mark.parametrize("reward", [True, False, None, "1", float("nan"), float("inf"), -1, 2])
def test_invalid_reward(reward):
    with pytest.raises(ValueError):
        RewardProfile().resolve({"reward": reward})


def test_explicit_reward_semantics():
    assert RewardProfile("cost", -.01, 1, (0, 100)).resolve({"cost": 75, "other": 0})["training_reward"] == .25
    assert RewardProfile("score", raw_range=None).resolve({"score": -3})["training_reward"] == -3
    with pytest.raises(ValueError):
        RewardProfile().resolve({"score": 1})


def test_collector_json_priority_and_artifact_bool(tmp_path):
    ident = identity()
    receipt(tmp_path, ident, 0)
    root = tmp_path / ident.attempt_id
    (root / "reward.txt").write_text("1")
    result = collect(ident, "trial-" + ident.attempt_id, root / "result.json", root, RewardProfile())
    assert result["training_reward"] == 0
    (root / "reward.json").write_text('{"reward":false}')
    with pytest.raises(ValueError):
        collect(ident, result["harbor_trial_id"], root / "result.json", root, RewardProfile())


@pytest.mark.parametrize("changes", [{"finished_at": None}, {"exception_info": {"exception_type": "Timeout"}},
                                      {"verifier_environment_mode": "separate"}, {"step_results": [{}]}])
def test_collector_rejects_unverified_lifecycle(tmp_path, changes):
    with pytest.raises(ValueError):
        receipt(tmp_path, identity(), 0, **changes)


@pytest.mark.parametrize("mutation", ["version", "logprob", "receipt", "missing_turn", "reward", "delivery", "seal", "sampling"])
def test_bad_join_fails_closed(tmp_path, mutation):
    ir = make_ir(tmp_path)
    if mutation == "version":
        ir["turns"][0]["policy_version"] = "2"
    elif mutation == "logprob":
        ir["turns"][0]["logprobs"] = []
    elif mutation == "receipt":
        ir["evaluation"]["identity"]["attempt_id"] = "other"
    elif mutation == "missing_turn":
        ir["turns"] = []
    elif mutation == "reward":
        ir["evaluation"]["training_reward"] = 1
    elif mutation == "delivery":
        ir["turns"][0]["delivery"] = "sent"
    elif mutation == "seal":
        ir["trace_seal"]["errors"] = ["retry"]
    elif mutation == "sampling":
        ir["turns"][0]["sampling"] = {"temperature": .7}
    ir["readiness"] = "RL_READY"
    with pytest.raises(ValueError):
        require_ready(ir, allow_synthetic=True)


def test_trajectory_group_statistics_and_weights(tmp_path):
    group = [make_ir(tmp_path, "0", 0, 1), make_ir(tmp_path, "1", 1, 3)]
    exported = export_group(group, 2, allow_synthetic=True)
    assert exported["mean"] == .5
    assert exported["std"] == .5
    assert len(exported["records"]) == 4
    for slot in ("0", "1"):
        records = [r for r in exported["records"] if r["identity"]["slot_id"] == slot]
        assert math.fsum(w for r in records for w in r["token_weights"]) == pytest.approx(.5)
    assert all(r["loss_mask"][:2] == [0, 0] for r in exported["records"])
    with pytest.raises(ValueError, match="synthetic"):
        export_group(group, 2)
    with pytest.raises(ValueError, match="complete"):
        export_group(group[:1], 2, allow_synthetic=True)


def test_zero_variance_and_duplicate_slot(tmp_path):
    group = [make_ir(tmp_path, "0", 0), make_ir(tmp_path, "1", 0)]
    assert export_group(group, 2, allow_synthetic=True)["advantages"] == [0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        export_group([group[0], group[0]], 2, allow_synthetic=True)


def test_duplicate_json_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"reward":0,"reward":1}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)
