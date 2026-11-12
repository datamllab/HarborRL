"""Math data boundaries, sampled OPD mathematics and reference-trainer contracts."""
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from harborrl.config import ROOT
from harborrl.distill.config import validate
from harborrl.distill.data import import_samples, load_tasks, verify
from harborrl.trajectories.native import digest


def config(tmp_path, **overrides):
    raw = {"tasks": {"path": str(ROOT / "examples/distill/math.jsonl")},
           "output": {"directory": str(tmp_path / "run")}, "training": {"steps": 1, "batch_size": 2, "microbatch_size": 1},
           "sampling": {"max_new_tokens": 3, "max_context": 128}, **overrides}
    return validate(raw, tmp_path / "config.yaml", allow_synthetic=True)


@pytest.mark.parametrize("response,reference,status", [("2", "2", "correct"), ("#### 1/2", "0.5", "correct"),
    (r"reasoning \boxed{3}", "3", "correct"), ("2", "3", "incorrect"), ("2 or 3", "3", "unparseable"),
    ("__import__('os').system('echo unsafe')", "0", "unparseable"), ("1/0", "1", "unparseable"), ("x" * 20000, "0", "unsupported")])
def test_rational_verifier(response, reference, status):
    assert verify(response, reference)["status"] == status


def test_split_lock_and_curated_import(tmp_path):
    source = ROOT / "examples/distill/math.jsonl"
    tasks = load_tasks(source)
    duplicate = {**tasks[0], "task_id": "different", "split": "test"}
    leaked = tmp_path / "leaked.jsonl"
    leaked.write_text("\n".join(json.dumps(row) for row in [tasks[0], duplicate]))
    with pytest.raises(ValueError, match="leakage"):
        load_tasks(leaked)
    samples = tmp_path / "curated.jsonl"
    samples.write_text(json.dumps({"task_id": tasks[0]["task_id"], "response": "2", "source": "human-reviewed", "license": "MIT"}))
    output = tmp_path / "dataset"
    manifest = import_samples(source, samples, output)
    sample = json.loads(next((output / "samples").glob("*.json")).read_text())
    assert sample["evidence_kind"] == "declared" and sample["token_evidence"] is None
    assert manifest["training_eligibility"] == "not_an_on_policy_batch"
    with pytest.raises(ValueError, match="new directory"):
        import_samples(source, samples, output)


@pytest.mark.parametrize("mutation", ["fake", "unknown", "temperature", "method", "group", "split"])
def test_config_rejections(tmp_path, mutation):
    value = config(tmp_path)
    if mutation == "fake":
        with pytest.raises(ValueError, match="allow-synthetic"):
            validate(value, tmp_path / "config")
        return
    if mutation == "unknown":
        value["training"]["epochs"] = 2
    elif mutation == "temperature":
        value["sampling"]["temperature"] = 0.7
    elif mutation == "method":
        value["teacher"]["kind"] = "none"
    elif mutation == "group":
        value["training"]["group_size"] = 1
    else:
        value["tasks"]["eval_split"] = "train"
    with pytest.raises(ValueError):
        validate(value, tmp_path / "config", allow_synthetic=True)


def test_offline_help_and_plan_without_torch(tmp_path):
    script = '''
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch','transformers','harbor','sglang','opensandbox'}
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from harborrl.cli import main
assert main(['distill','train',sys.argv[1],'--allow-synthetic','--dry-run']) == 0
'''
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(tmp_path)))
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["backend_contract"] == "transformers-math-onpolicy-v1"
    assert not (tmp_path / "run").exists()


def test_opd_expectation_gradient_and_detach():
    torch = pytest.importorskip("torch")
    from harborrl.algorithms.opd import sampled_opd_loss
    logits = torch.tensor([-.5, .2, .8], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([-.2, .4, -.1], dtype=torch.float64, requires_grad=True).log_softmax(0)
    logq = logits.log_softmax(0)
    exact = (logq.exp() * (logq - teacher.detach())).sum()
    expected = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    surrogate = sum(logq[i].detach().exp() * sampled_opd_loss(logq[i].reshape(1, 1),
        logq[i].reshape(1, 1), teacher[i].reshape(1, 1), torch.ones(1, 1)) for i in range(3))
    observed = torch.autograd.grad(surrogate, logits, retain_graph=True)[0]
    torch.testing.assert_close(expected, observed)
    assert observed.norm() > 0
    assert torch.autograd.grad(surrogate, teacher, allow_unused=True)[0] is None


def test_loss_mask_and_microbatch_reduction():
    torch = pytest.importorskip("torch")
    from harborrl.algorithms.opd import trajectory_surrogate
    current = torch.tensor([[-1., -2., 0.], [-3., 0., 0.], [-2., -3., -4.]], requires_grad=True)
    advantages = torch.tensor([[1., 2., 9.], [3., 8., 8.], [1., -1., 2.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]])
    full = trajectory_surrogate(current, advantages, mask)
    split = trajectory_surrogate(current[:2], advantages[:2], mask[:2]) * 2 / 3 + trajectory_surrogate(current[2:], advantages[2:], mask[2:]) / 3
    torch.testing.assert_close(full, split)
    full.backward()
    assert current.grad[0, 2] == 0 and advantages.grad is None


def test_prm_step_alignment_and_equation():
    pytest.importorskip("transformers")
    from harborrl.distill.models import ByteTokenizer
    from harborrl.distill.scoring import process_rewards
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("1 + 1 = 2\n2 * 3 = 7")[1:]
    scored = process_rewards("equation", tokens, tokenizer)
    assert [step["status"] for step in scored["steps"]] == ["correct", "incorrect"]
    assert len(scored["token_returns"]) == len(tokens)
    assert scored["token_returns"][0] == 0 and scored["token_returns"][-1] == -1
    fake = process_rewards("fake", tokens, tokenizer)
    assert fake["evidence_kind"] == "synthetic"


def test_model_shift_padding_and_teacher_probe(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from harborrl.distill.models import load_student, response_logprobs
    from harborrl.distill.scoring import ModelTeacher, probe_teacher
    value = config(tmp_path, teacher={"kind": "tiny_qwen"})
    model, tokenizer = load_student(value, torch.device("cpu"))
    prompt, response = [1, 40, 50], [52, 2]
    scored = response_logprobs(model, [prompt, [1]], [response, [2]], tokenizer.pad_token_id)
    logits = model(torch.tensor([prompt + response])).logits[0].log_softmax(-1)
    torch.testing.assert_close(scored[0][0], logits[2, 52])
    torch.testing.assert_close(scored[0][1], logits[3, 2])
    teacher = ModelTeacher(value, torch.device("cpu"), tokenizer.identity())
    assert probe_teacher(teacher, tokenizer)["probe"] == "passed"
    assert teacher.unchanged()
    with pytest.raises(ValueError, match="mismatch"):
        teacher.score_tokens(prompt, response, {**tokenizer.identity(), "template_digest": "bad"})


def test_training_resume_and_admission(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from harborrl.distill.contract import validate_row
    from harborrl.distill.runner import train
    value = config(tmp_path)
    root = Path(value["output"]["directory"])
    root.mkdir()
    train(value, allow_synthetic=True)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["parameter_changed"] and summary["nonzero_gradient_steps"] == 1
    row = json.loads(next((root / "batches").glob("*.json")).read_text())
    kwargs = {"version": row["student_behavior_version"], "identity": row["identity"], "context": 128}
    with pytest.raises(ValueError, match="Synthetic"):
        validate_row(row, **kwargs)
    with pytest.raises(ValueError, match="Stale"):
        validate_row(row, **{**kwargs, "version": "old"}, allow_synthetic=True)
    tampered = deepcopy(row)
    tampered["behavior_logprobs"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_row(tampered, **kwargs, allow_synthetic=True)
    for key, replacement in (("teacher_logprobs", [0.5] * len(row["response_ids"])),
                             ("loss_mask", [0] * len(row["response_ids"])),
                             ("verification", {"status": "correct", "reward": -1.0})):
        altered = deepcopy(row)
        altered[key] = replacement
        altered["score_response_digest"] = digest({"request_digest": altered["score_request_digest"],
                                                   "logprobs": altered["teacher_logprobs"]})
        altered["digest"] = digest({k: v for k, v in altered.items() if k != "digest"})
        with pytest.raises(ValueError):
            validate_row(altered, **kwargs, allow_synthetic=True)
    resume = root / "checkpoint-000001.pt"
    resumed = deepcopy(value)
    resumed["output"]["directory"] = str(tmp_path / "resumed")
    resumed["training"]["steps"] = 2
    Path(resumed["output"]["directory"]).mkdir()
    train(resumed, allow_synthetic=True, resume=resume)
    continuous = deepcopy(resumed)
    continuous["output"]["directory"] = str(tmp_path / "continuous")
    Path(continuous["output"]["directory"]).mkdir()
    train(continuous, allow_synthetic=True)
    expected = torch.load(Path(continuous["output"]["directory"]) / "checkpoint-000002.pt", weights_only=True)
    actual = torch.load(Path(resumed["output"]["directory"]) / "checkpoint-000002.pt", weights_only=True)
    assert expected["parameter_digest"] == actual["parameter_digest"]


def test_prm_failure_precedes_optimizer(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from harborrl.distill import runner
    value = config(tmp_path, method="prm", teacher={"kind": "none"}, prm={"kind": "fake"})
    root = Path(value["output"]["directory"])
    root.mkdir()
    def invalid(*args):
        return {"evidence_kind": "synthetic", "steps": [], "token_returns": []}
    monkeypatch.setattr(runner, "process_rewards", invalid)
    with pytest.raises(ValueError, match="PRM"):
        runner.train(value, allow_synthetic=True)
    assert not list(root.glob("checkpoint-*.pt"))
    assert not (root / "summary.json").exists()


def test_deployment_requires_explicit_opt_in(tmp_path, monkeypatch):
    from harborrl.distill.cli import plan
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deployment.yaml").write_text("invalid deployment")
    assert plan(config(tmp_path))["deployment_source"] == "embedded-local"
