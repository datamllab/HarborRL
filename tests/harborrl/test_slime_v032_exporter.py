"""CPU contracts for the backend-neutral exporter and Slime v0.3.2 adapter."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_contracts import identity, make_ir

from harborrl.backends.slime_v032 import launcher as v032_launcher
from harborrl.backends.slime_v032 import versions as v032_versions
from harborrl.backends.slime_v032.advantage import compute_advantages_and_returns
from harborrl.backends.slime_v032.converter import (
    STANDARD_FIELDS,
    convert_samples_to_train_data,
)
from harborrl.backends.slime_v032.loss import (
    LOSS_SCALE_CORRECTION,
    clipped_pg_loss_pure,
)
from harborrl.backends.slime_v032.postprocess import rollout_data_postprocess
from harborrl.backends.slime_v032.rollout import generate_rollout, to_slime_samples
from harborrl.export import contract as export_contract
from harborrl.export.native import export_training_batch, group_statistics
from harborrl.export.validate import batch_weight_sum
from harborrl.trajectories.native import digest


def real_evidence(ir):
    ir = json.loads(json.dumps(ir))
    for turn in ir["turns"]:
        turn["evidence_kind"] = "serving"
    ir["evaluation"]["evidence_kind"] = "harbor"
    ir["trace_seal"]["turns_digest"] = digest(ir["turns"])
    ir["readiness"] = "RL_READY"
    return ir


def reidentify(ir, **overrides):
    ident = identity(**{**ir["identity"], **overrides})
    for turn in ir["turns"]:
        turn["identity"] = ident.to_dict()
        turn["policy_version"] = ident.policy_version
    ir["identity"] = ident.to_dict()
    ir["evaluation"]["identity"] = ident.to_dict()
    ir["trace_seal"]["identity"] = ident.to_dict()
    ir["trace_seal"]["turns_digest"] = digest(ir["turns"])
    return ir


def native_groups(tmp_path):
    first = [
        reidentify(real_evidence(make_ir(tmp_path, "0", 0, 1)), slot_id="0"),
        reidentify(real_evidence(make_ir(tmp_path, "1", 1, 3)), slot_id="1"),
    ]
    second = [
        reidentify(real_evidence(make_ir(tmp_path, "0", 0, 2)), group_id="group-b", slot_id="0"),
        reidentify(real_evidence(make_ir(tmp_path, "1", 0, 1)), group_id="group-b", slot_id="1"),
    ]
    return [first, second]


class FakeSample:
    def __init__(self, unit, span, ir, rollout_id, index):
        self.group_index = None
        self.index = index
        self.rollout_id = rollout_id
        self.tokens = list(span.tokens)
        self.response_length = span.response_length
        self.response = "native Harbor trial"
        self.reward = {"score": unit.reward}
        self.loss_mask = list(span.loss_mask[span.prompt_length:])
        self.rollout_log_probs = list(span.old_logprobs)
        self.weight_versions = [span.policy_version]
        self.metadata = {"native_ir": ir, "native_turn": span.turn_index}


def fake_samples(batch, ir_groups):
    samples = []
    index = 0
    for group, trajectories in zip(batch.groups, ir_groups, strict=True):
        by_id = {ir["identity"]["trajectory_id"]: ir for ir in trajectories}
        for unit in group.trajectories:
            for span in unit.turns:
                samples.append(FakeSample(unit, span, by_id[unit.trajectory_id], unit.rollout_id, index))
                index += 1
    return samples


def test_export_training_batch_goldel(tmp_path):
    groups = native_groups(tmp_path)
    batch = export_training_batch(groups, rollout_ids=[10, 11])

    assert batch.schema_version == "harborrl-training-batch-v1"
    assert batch.reduction == export_contract.REDUCTION
    assert batch.policy_versions == frozenset({"1"})
    assert [group.group_size for group in batch.groups] == [2, 2]
    assert [unit.rollout_id for group in batch.groups for unit in group.trajectories] == [10, 10, 11, 11]
    assert batch.groups[0].reward_mean == pytest.approx(0.5)
    assert batch.groups[0].reward_std == pytest.approx(0.5)
    assert batch.groups[1].reward_std == 0.0
    assert [unit.advantage for unit in batch.groups[1].trajectories] == [0.0, 0.0]
    first_advantages = [unit.advantage for unit in batch.groups[0].trajectories]
    assert first_advantages == [-0.5, 0.5]
    assert [unit.token_count for group in batch.groups for unit in group.trajectories] == [2, 6, 4, 2]
    for group in batch.groups:
        for unit in group.trajectories:
            weight = 1.0 / (group.group_size * unit.token_count)
            for span in unit.turns:
                assert len(span.tokens) == span.prompt_length + span.response_length
                assert len(span.loss_mask) == len(span.tokens)
                assert not any(span.loss_mask[:span.prompt_length])
                assert all(span.loss_mask[span.prompt_length:])
                assert len(span.old_logprobs) == span.response_length
                assert weight == pytest.approx(1.0 / (2 * unit.token_count))
    assert batch_weight_sum(batch) == pytest.approx(2.0)


def test_centered_advantages_keep_reward_scale_and_reject_normalized_batch(tmp_path):
    from dataclasses import replace

    from harborrl.export.validate import validate_batch

    assert group_statistics([0.0, 1.0])[2] == [-0.5, 0.5]
    assert group_statistics([-2.0, 2.0])[2] == [-2.0, 2.0]
    batch = export_training_batch(native_groups(tmp_path), rollout_ids=[10, 11])
    group = batch.groups[0]
    normalized = replace(group, trajectories=tuple(
        replace(unit, advantage=unit.advantage / group.reward_std)
        for unit in group.trajectories
    ))
    with pytest.raises(ValueError, match="centered"):
        validate_batch(replace(batch, groups=(normalized, *batch.groups[1:])))


def test_export_rejects_malformed_groups(tmp_path):
    groups = native_groups(tmp_path)
    with pytest.raises(ValueError, match="at least two"):
        export_training_batch([groups[0][:1]])
    with pytest.raises(ValueError, match="distinct rollout id"):
        export_training_batch(groups, rollout_ids=[7, 7])
    mixed = json.loads(json.dumps(groups[0][0]))
    mixed["turns"][0]["policy_version"] = "2"
    with pytest.raises(ValueError, match="native IR rejected"):
        export_training_batch([[mixed, groups[0][1]]], allow_synthetic=True)
    stale = json.loads(json.dumps(groups[0][0]))
    for turn in stale["turns"]:
        turn["evidence_kind"] = "synthetic"
    with pytest.raises(ValueError, match="synthetic"):
        export_training_batch([[stale, groups[0][1]]])


def test_converter_outputs_only_standard_fields(tmp_path):
    groups = native_groups(tmp_path)
    batch = export_training_batch(groups, rollout_ids=[4, 5])
    samples = fake_samples(batch, groups)
    data = convert_samples_to_train_data(SimpleNamespace(n_samples_per_prompt=2), samples)

    assert tuple(data) == STANDARD_FIELDS
    assert not any(key.startswith("native_") for key in data)
    assert len(data["tokens"]) == 7
    assert data["rollout_ids"] == [4, 4, 4, 4, 5, 5, 5]
    assert data["rollout_mask_sums"] == [2, 6, 6, 6, 4, 4, 2]
    negative = batch.groups[0].trajectories[0].advantage
    positive = batch.groups[0].trajectories[1].advantage
    assert data["rewards"] == pytest.approx([negative, positive, positive, positive, 0.0, 0.0, 0.0])
    assert data["raw_reward"] == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    group_budget = {4: 0.0, 5: 0.0}
    for index, mask_sum in enumerate(data["rollout_mask_sums"]):
        group_budget[data["rollout_ids"][index]] += sum(
            1.0 / (2 * mask_sum) for flag in data["loss_masks"][index] if flag
        )
    assert group_budget == {4: pytest.approx(1.0), 5: pytest.approx(1.0)}
    assert all(data["truncated"][index] == 0 for index in range(7))


def test_converter_fails_closed_on_evidence_drift(tmp_path):
    groups = native_groups(tmp_path)
    batch = export_training_batch(groups, rollout_ids=[4, 5])
    samples = fake_samples(batch, groups)
    complete = convert_samples_to_train_data(SimpleNamespace(n_samples_per_prompt=2), samples)
    assert len(complete["tokens"]) == 7

    dropped = list(samples)
    del dropped[2]
    with pytest.raises(ValueError, match="incomplete native turns"):
        convert_samples_to_train_data(SimpleNamespace(n_samples_per_prompt=2), dropped)
    tampered = list(samples)
    tampered[0].tokens = list(tampered[0].tokens) + [999]
    with pytest.raises(ValueError, match="differs from native evidence"):
        convert_samples_to_train_data(SimpleNamespace(n_samples_per_prompt=2), tampered)
    duplicate = list(samples)
    duplicate.append(duplicate[0])
    with pytest.raises(ValueError, match="duplicate native turn"):
        convert_samples_to_train_data(SimpleNamespace(n_samples_per_prompt=2), duplicate)
    bad_rollout = list(samples)
    bad_rollout[0].rollout_id = 99
    with pytest.raises(ValueError, match="multiple Slime rollout ids"):
        convert_samples_to_train_data(SimpleNamespace(n_samples_per_prompt=2), bad_rollout)


def test_advantage_hook_expands_centered_reward():
    rollout_data = {
        "rewards": [0.25, -0.5],
        "loss_masks": [[1, 1, 0], [1]],
        "rollout_log_probs": [[-0.1, -0.2, -0.3], [-0.4]],
    }
    compute_advantages_and_returns(SimpleNamespace(), rollout_data)
    assert rollout_data["advantages"] == [[0.25, 0.25, 0.25], [-0.5]]
    assert rollout_data["returns"] == rollout_data["advantages"]

    rollout_data["rewards"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compute_advantages_and_returns(SimpleNamespace(), rollout_data)
    with pytest.raises(ValueError, match="mismatched sample counts"):
        compute_advantages_and_returns(SimpleNamespace(), {
            "rewards": [1.0], "loss_masks": [[1], [1]], "rollout_log_probs": [[-1.0], [-1.0]],
        })


def test_pure_clipped_loss_numerics():
    weights = [1.0 / (2 * 4)] * 4
    old = [-0.2] * 4
    new = [-0.1] * 4
    advantages = [1.0] * 4
    ratio = pow(2.718281828459045, 0.1)
    expected = sum(max(-ratio, -min(max(ratio, 0.8), 1.2)) * weight for weight in weights)
    assert clipped_pg_loss_pure(new, old, advantages, weights) == pytest.approx(expected)
    with pytest.raises(ValueError):
        clipped_pg_loss_pure(new, old, advantages, [weight for weight in weights] + [0.0])
    with pytest.raises(ValueError):
        clipped_pg_loss_pure(new, old, advantages, [-weight for weight in weights])
    with pytest.raises(ValueError):
        clipped_pg_loss_pure(new, old, advantages, weights, epsilon=0.0)
    assert LOSS_SCALE_CORRECTION == 1.0


def test_postprocess_accepts_multiple_groups_and_partial_actor_shards(tmp_path):
    args = SimpleNamespace(n_samples_per_prompt=2)
    groups = native_groups(tmp_path)
    batch = export_training_batch(groups, rollout_ids=[10, 11])
    data = convert_samples_to_train_data(args, fake_samples(batch, groups))
    rollout_data_postprocess(args, 0, data)
    shard = {key: values[::2] for key, values in data.items()}
    rollout_data_postprocess(args, 0, shard)


def test_postprocess_rejects_individual_group_overweight():
    args = SimpleNamespace(n_samples_per_prompt=2)
    data = {
        "tokens": [[1, 2]] * 4,
        "response_lengths": [1] * 4,
        "rewards": [0.0] * 4,
        "raw_reward": [0.0] * 4,
        "truncated": [0] * 4,
        "sample_indices": list(range(4)),
        "rollout_ids": [10, 10, 10, 11],
        "loss_masks": [[1]] * 4,
        "rollout_log_probs": [[-0.2]] * 4,
        "rollout_mask_sums": [1, 1, 1, 2],
    }
    # Total weight is below two, but group 10 exceeds its own budget.
    with pytest.raises(ValueError, match="budget.*10"):
        rollout_data_postprocess(args, 0, data)


@pytest.mark.parametrize("group_id", [-1, 0.5, float("nan"), "invalid"])
def test_postprocess_rejects_invalid_group_id(tmp_path, group_id):
    args = SimpleNamespace(n_samples_per_prompt=2)
    groups = native_groups(tmp_path)
    batch = export_training_batch(groups, rollout_ids=[10, 11])
    data = convert_samples_to_train_data(args, fake_samples(batch, groups))
    data["rollout_ids"][0] = group_id
    with pytest.raises(ValueError, match="rollout id"):
        rollout_data_postprocess(args, 0, data)


def test_postprocess_rejects_drift():
    args = SimpleNamespace(n_samples_per_prompt=2)
    base = {
        "tokens": [[1, 2, 3]],
        "response_lengths": [1],
        "rewards": [0.5],
        "raw_reward": [1.0],
        "truncated": [0],
        "sample_indices": [0],
        "rollout_ids": [0],
        "loss_masks": [[1]],
        "rollout_log_probs": [[-0.2]],
        "rollout_mask_sums": [2],
    }
    rollout_data_postprocess(args, 0, dict(base))
    missing = {key: value for key, value in base.items() if key != "raw_reward"}
    with pytest.raises(ValueError, match="missing standard fields"):
        rollout_data_postprocess(args, 0, missing)
    drifted = {**base, "rollout_mask_sums": [0]}
    with pytest.raises(ValueError, match="mask sum"):
        rollout_data_postprocess(args, 0, drifted)
    over_budget = {key: value * 3 if isinstance(value, list) else value for key, value in base.items()}
    over_budget["rollout_mask_sums"] = [1, 1, 1]
    with pytest.raises(ValueError, match="budget"):
        rollout_data_postprocess(args, 0, over_budget)


def test_postprocess_accepts_tensor_like_scalars():
    class Scalar:
        def __float__(self):
            return -0.2

    class IntegerScalar:
        def __float__(self):
            return 1.0

    args = SimpleNamespace(n_samples_per_prompt=2)
    rollout_data_postprocess(args, 0, {
        "tokens": [[1, 2, 3]],
        "response_lengths": [IntegerScalar()],
        "rewards": [Scalar()],
        "raw_reward": [Scalar()],
        "truncated": [IntegerScalar()],
        "sample_indices": [0],
        "rollout_ids": [IntegerScalar()],
        "loss_masks": [[1]],
        "rollout_log_probs": [[Scalar()]],
        "rollout_mask_sums": [IntegerScalar()],
    })


def test_versions_and_doctor_contract():
    expected = v032_versions.expected_versions()
    assert expected["slime_commit"] == "3778dbf6d1a533ab478ecf5ddaa11449a47752b2"
    assert v032_versions.validate_backend_versions(
        slime_version="v0.3.2", slime_commit=expected["slime_commit"],
        megatron_commit=expected["megatron_commit"], sglang_image=expected["sglang_image"],
    )
    with pytest.raises(ValueError, match="does not match"):
        v032_versions.validate_backend_versions(slime_version="0.3.1", slime_commit="x",
                                                megatron_commit="y", sglang_image="z")
    with pytest.raises(ValueError, match="required"):
        v032_versions.validate_backend_versions()
    assert v032_versions.runtime_contract_id(v032_versions.BACKEND_CONTRACT) == "slime-v032"
    with pytest.raises(ValueError):
        v032_versions.runtime_contract_id("slime-legacy")
    with pytest.raises(ValueError):
        v032_versions.runtime_contract_id("unknown")


def test_launcher_hooks_resolve_and_dry_run():
    arguments = v032_launcher.hook_arguments()
    assert "--rollout-function-path" in arguments
    assert arguments.count(["--loss-type", "custom_loss"][0]) == 1
    flags = dict(zip(arguments[::2], arguments[1::2]))
    assignment = {
        "rollout": flags["--rollout-function-path"],
        "converter": flags["--custom-convert-samples-to-train-data-path"],
        "advantage": flags["--custom-advantage-function-path"],
        "loss": flags["--custom-loss-function-path"],
        "postprocess": flags["--rollout-data-postprocess-path"],
    }
    v032_launcher.validate_hook_assignment(assignment)
    with pytest.raises(ValueError, match="exactly match"):
        v032_launcher.validate_hook_assignment({key: value for key, value in assignment.items() if key != "rollout"})
    for module_name in v032_launcher.hook_modules().values():
        assert importlib.import_module(module_name) is not None
    command = v032_launcher.dry_run_command("/train.py", ["--num-rollout", "1"], python="/usr/bin/python")
    assert command[:3] == ["/usr/bin/python", "-u", "/train.py"]
    assert "--num-rollout" in command and "--custom-loss-function-path" in command
    with pytest.raises(ValueError, match="unsupported backend contract"):
        v032_launcher.hook_arguments("slime-legacy")


def test_rollout_hook_locks_policy_version_and_keeps_groups_atomic(tmp_path, monkeypatch):
    groups = native_groups(tmp_path)
    locked = [
        [reidentify(json.loads(json.dumps(ir)), policy_version="7") for ir in group]
        for group in groups
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "policy-pool.json").write_text(json.dumps({"weight_version": "7"}))
    monkeypatch.setenv("RUN_DIR", str(run_dir))
    args = SimpleNamespace(
        rollout_batch_size=2, n_samples_per_prompt=2, rollout_max_response_len=128,
    )
    events = []

    async def runner(args_, groups_, policy_version):
        events.append(("generate", policy_version))
        return json.loads(json.dumps(locked))

    class SlimeDataSource:
        def get_samples(self, count):
            return [
                [SimpleNamespace(metadata={}) for _ in range(2)]
                for _ in range(count)
            ]

    output = generate_rollout(
        args, 3, SlimeDataSource(),
        runner=runner,
        output_factory=SimpleNamespace,
        sample_factory=lambda **fields: SimpleNamespace(**fields),
    )
    assert events == [("generate", None)]
    assert output.metrics["native_group_count"] == 2.0
    assert output.metrics["native_trajectory_count"] == 4.0
    assert output.metrics["native_turn_count"] == 7.0
    assert output.metrics["native_policy_version"] == "7"
    assert [len(group) for group in output.samples] == [4, 3]
    rollout_ids = {sample.rollout_id for group in output.samples for sample in group}
    assert rollout_ids == {6, 7}
    assert all(
        sample.reward == {
            "score": sample.metadata["native_ir"]["evaluation"]["training_reward"]
        }
        for group in output.samples
        for sample in group
    )
    assert all(sample.metadata["native_ir"]["identity"]["policy_version"] == "7" for group in output.samples for sample in group)

    async def stale_runner(args_, groups_, policy_version):
        return [[reidentify(json.loads(json.dumps(ir)), policy_version="8") for ir in group] for group in locked]

    with pytest.raises(ValueError, match="stale policy version"):
        generate_rollout(
            args, 3, SlimeDataSource(),
            runner=stale_runner, output_factory=SimpleNamespace,
            sample_factory=lambda **fields: SimpleNamespace(**fields),
        )
    (run_dir / "policy-pool.json").unlink()
    with pytest.raises(ValueError, match="locked policy pool"):
        generate_rollout(
            args, 3, SlimeDataSource(),
            runner=runner, output_factory=SimpleNamespace,
            sample_factory=lambda **fields: SimpleNamespace(**fields),
        )


@pytest.mark.parametrize("dtype_name", ["int64", "bool", "float32"])
def test_advantage_preserves_fractional_rewards_with_tensor_masks(dtype_name):
    torch = pytest.importorskip("torch")
    data = {
        "rewards": [-0.25, 0.75],
        "loss_masks": [torch.ones(2, dtype=getattr(torch, dtype_name)) for _ in range(2)],
        "rollout_log_probs": [torch.zeros(2) for _ in range(2)],
    }
    compute_advantages_and_returns(SimpleNamespace(), data)
    for key in ("advantages", "returns"):
        assert data[key][0].tolist() == [-0.25, -0.25]
        assert data[key][1].tolist() == [0.75, 0.75]
        assert all(value.dtype == torch.float32 for value in data[key])


def test_to_slime_samples_requires_ir_metadata(tmp_path):
    groups = native_groups(tmp_path)
    batch = export_training_batch(groups, rollout_ids=[1, 2])
    with pytest.raises(ValueError, match="native IR metadata"):
        to_slime_samples(batch, sample_factory=lambda **fields: SimpleNamespace(**fields))


def test_export_package_has_no_backend_imports():
    root = Path(__file__).resolve().parents[2] / "harborrl" / "export"
    forbidden = {"slime", "torch", "megatron"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module.split(".")[0]} if node.module else set()
            else:
                continue
            assert not (names & forbidden), f"{path} imports backend dependencies: {names & forbidden}"


@pytest.mark.parametrize('group_index, expected_advantages, expected_counts', [
    (0, [-0.5, 0.5], [2, 6]),
    (1, [0.0, 0.0], [4, 2]),
])
def test_legacy_export_preserves_grpo_record_semantics(
    tmp_path, group_index, expected_advantages, expected_counts,
):
    from harborrl.rollout.exporters.native import export_group
    group = native_groups(tmp_path)[group_index]
    exported = export_group(group, 2)
    assert exported['export_version'] == 'native-grpo-v2'
    assert exported['advantages'] == expected_advantages
    assert exported['reduction'] == 'sum_token_weighted_loss; no secondary whitening'
    assert exported['synthetic'] is False
    for ir, advantage, count in zip(group, expected_advantages, expected_counts):
        records = [record for record in exported['records']
                   if record['identity']['trajectory_id'] == ir['identity']['trajectory_id']]
        assert len(records) == len(ir['turns'])
        for index, (turn, record) in enumerate(zip(ir['turns'], records)):
            n = len(turn['output_ids'])
            assert record == {
                'identity': ir['identity'], 'turn_index': index,
                'tokens': turn['input_ids'] + turn['output_ids'],
                'prompt_length': len(turn['input_ids']), 'response_length': n,
                'loss_mask': [0] * len(turn['input_ids']) + [1] * n,
                'old_logprobs': turn['logprobs'], 'advantages': [advantage] * n,
                'token_weights': [1.0 / (2 * count)] * n,
                'training_reward': ir['evaluation']['training_reward'], 'padding': False,
            }


def test_native_task_inspection_avoids_legacy_conversion_imports():
    import subprocess
    import sys
    code = (
        'import sys; import harborrl.data.harbor.native_inspector; '
        'assert "harborrl.data.harbor.inspector" not in sys.modules; '
        'assert "harborrl.data.harbor.materializer" not in sys.modules'
    )
    subprocess.run([sys.executable, '-c', code], check=True)
