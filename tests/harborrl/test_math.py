"""Math data boundaries, sampled OPD mathematics and process-reward alignment."""
import json

import pytest

from harborrl.config import ROOT
from harborrl.math.data import import_samples, load_tasks, verify


@pytest.mark.parametrize("response,reference,status", [("2", "2", "correct"), ("#### 1/2", "0.5", "correct"),
    (r"reasoning \boxed{3}", "3", "correct"), ("2", "3", "incorrect"), ("2 or 3", "3", "unparseable"),
    ("__import__('os').system('echo unsafe')", "0", "unparseable"), ("1/0", "1", "unparseable"), ("x" * 20000, "0", "unsupported")])
def test_rational_verifier(response, reference, status):
    assert verify(response, reference)["status"] == status


@pytest.mark.parametrize("response,reference,status", [
    (r"#### Final Answer: $\boxed{57,500}$", "57500", "correct"),
    ("work\n#### **42**\nMore explanation.", "42", "correct"),
    (r"\boxed{14} years ago.", "14", "correct"),
    (r"\boxed{12} corrected to \boxed{13}", "13", "correct"),
    (r"\boxed{1/2}", "0.5", "correct"),
    (r"\boxed{12,34}", "1234", "unparseable"),
    (r"\boxed{1/0}", "1", "unparseable"),
    ("The answer could be 2 or 3", "3", "unparseable"),
    (r"\boxed{2}", "3", "incorrect"),
])
def test_gsm8k_numeric_verifier(response, reference, status):
    result = verify(response, reference, "gsm8k_numeric_v1")
    assert result["status"] == status
    assert result["verifier"] == "gsm8k_numeric_v1"
    assert result["reward"] == (1.0 if status == "correct" else 0.0)


def test_verifier_version_is_locked_in_task_import(tmp_path):
    row = {**load_tasks(ROOT / "examples/math/tasks.jsonl")[0], "verifier": "gsm8k_numeric_v1"}
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(row))
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"task_id": row["task_id"], "response": r"\boxed{999} done",
                                   "source": "fixture", "license": "MIT"}))
    output = tmp_path / "out"
    import_samples(tasks, samples, output)
    result = json.loads(next((output / "verification").glob("*.json")).read_text())
    assert result["verifier"] == "gsm8k_numeric_v1" and result["reward"] == 0.0
    assert verify(r"\boxed{2} done", "2")["status"] == "unparseable"
    with pytest.raises(ValueError, match="Unsupported numeric verifier"):
        verify("2", "2", "unknown")


def test_split_lock_and_curated_import(tmp_path):
    source = ROOT / "examples/math/tasks.jsonl"
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
    from harborrl.math.scoring import process_rewards

    class Tokenizer:
        def decode(self, tokens):
            return bytes(tokens).decode("ascii")

    tokenizer = Tokenizer()
    tokens = list(b"1 + 1 = 2\n2 * 3 = 7")
    scored = process_rewards("equation", tokens, tokenizer)
    assert [step["status"] for step in scored["steps"]] == ["correct", "incorrect"]
    assert len(scored["token_returns"]) == len(tokens)
    assert scored["token_returns"][0] == 0 and scored["token_returns"][-1] == -1
    fake = process_rewards("fake", tokens, tokenizer)
    assert fake["evidence_kind"] == "synthetic"
