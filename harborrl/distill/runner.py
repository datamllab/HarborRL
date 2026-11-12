"""Bounded reference trainer: raw student rollout, scoring, one optimizer update."""
import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from harborrl.algorithms.opd import trajectory_surrogate
from harborrl.trajectories.native import digest, publish

from .config import load
from .contract import score_request, validate_row
from .data import load_tasks, verify
from .models import generate, load_student, parameter_digest, response_logprobs
from .scoring import FakeTeacher, ModelTeacher, probe_teacher, process_rewards


def training_signature(config, tasks_digest):
    value = deepcopy(config)
    for key in ("steps", "microbatch_size", "save_interval"):
        value["training"].pop(key)
    value.pop("output")
    value.pop("deployment")
    return digest({"config": value, "tasks_digest": tasks_digest})


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".pending")
    torch.save(value, temporary)
    temporary.replace(path)


def pad(values, length, device):
    return torch.stack([torch.cat((value, value.new_zeros(length - len(value)))) if isinstance(value, torch.Tensor)
                        else torch.tensor([*value, *([0] * (length - len(value)))], device=device) for value in values])


def optimize(model, optimizer, rows, config, tokenizer, world):
    """Equal-size rank shards; mean per trajectory, then mean over global batch."""
    device = next(model.parameters()).device
    local_count = len(rows)
    optimizer.zero_grad(set_to_none=True)
    loss_sum, maximum_error = 0.0, 0.0
    micro = config["training"]["microbatch_size"]
    if config.get('consistency', {}).get('mode', 'off') != 'off':
        from harborrl.consistency.reference import check_reference
        check_reference(model, rows, config, tokenizer)
    for offset in range(0, local_count, micro):
        chunk = rows[offset:offset + micro]
        current_rows = response_logprobs(model, [r["prompt_ids"] for r in chunk], [r["response_ids"] for r in chunk], tokenizer.pad_token_id)
        maximum = max(map(len, current_rows))
        current = pad(current_rows, maximum, device)
        behavior = pad([r["behavior_logprobs"] for r in chunk], maximum, device)
        mask = pad([r["loss_mask"] for r in chunk], maximum, device)
        error = float(((current.detach() - behavior).abs() * mask).max())
        maximum_error = max(maximum_error, error)
        if error > config["training"]["behavior_tolerance"]:
            raise ValueError(f"Actor no longer matches frozen behavior: {error}")
        advantage = torch.zeros_like(current)
        if config["method"] in {"opd", "prm_opd"}:
            teacher = pad([r["teacher_logprobs"] for r in chunk], maximum, device)
            advantage += config["training"]["opd_weight"] * (teacher - behavior)
        if config["method"] in {"prm", "prm_opd"}:
            advantage += config["prm"]["weight"] * pad([r["prm"]["token_returns"] for r in chunk], maximum, device)
        rewards = torch.tensor([r["verification"]["reward"] for r in chunk], device=device).unsqueeze(-1)
        advantage += config["training"]["outcome_weight"] * rewards
        loss = trajectory_surrogate(current, advantage, mask) * (len(chunk) / local_count)
        loss.backward()
        loss_sum += float(loss.detach())
    if world > 1:
        for parameter in model.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss_sum, "gradient_norm": float(norm), "max_behavior_error": maximum_error}


def train(config, *, allow_synthetic=False, resume=None):
    torch.set_num_threads(1)
    rank, world, local_rank = (int(os.environ.get(key, default)) for key, default in
                               (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    if world != config["deployment"]["world_size"]:
        raise ValueError("Launch world size differs from the declared data-parallel layout")
    cuda = config["deployment"]["device"] == "cuda"
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(seconds=120))
    root = Path(config["output"]["directory"])
    try:
        tasks = load_tasks(config["tasks"]["path"])
        selected = [task for task in tasks if task["split"] == "train"]
        if not selected:
            raise ValueError("No train tasks")
        signature = training_signature(config, digest(tasks))
        model, tokenizer = load_student(config, device)
        identity = tokenizer.identity()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0)
        start = 0
        if resume:
            checkpoint = torch.load(resume, map_location=device, weights_only=True)
            if checkpoint["signature"] != signature or checkpoint["identity"] != identity:
                raise ValueError("Checkpoint config/data/tokenizer identity mismatch")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start = checkpoint["step"]
            if parameter_digest(model) != checkpoint["parameter_digest"]:
                raise ValueError("Checkpoint parameter digest mismatch")
        if start >= config["training"]["steps"]:
            raise ValueError("training.steps must exceed resumed completed steps")
        initial = parameter_digest(model)
        teacher = None
        if config["teacher"]["kind"] != "none":
            teacher = (FakeTeacher(identity, config["sampling"]["max_context"]) if config["teacher"]["kind"] == "fake"
                       else ModelTeacher(config, device, identity))
            capabilities = probe_teacher(teacher, tokenizer)
            if rank == 0:
                publish(root / "teacher-probe.json", capabilities)
        if rank == 0:
            publish(root / "runtime.json", {"torch": str(torch.__version__), "transformers": __import__("transformers").__version__,
                "device": str(device), "world_size": world, "identity": identity, "parameter_count": sum(p.numel() for p in model.parameters()),
                "initial_parameter_digest": initial, "resume": str(resume) if resume else None,
                "synthetic_allowed": allow_synthetic, "method": config["method"], "signature": signature,
                "teacher_evidence": teacher.evidence_kind if teacher else None, "tasks_digest": digest(tasks)})
        metrics = []
        local_count = config["training"]["batch_size"] // world
        for step in range(start, config["training"]["steps"]):
            begin = time.monotonic()
            version = parameter_digest(model)
            rows = []
            for index in range(rank * local_count, (rank + 1) * local_count):
                task = selected[(step * config["training"]["batch_size"] + index) % len(selected)]
                prompt = tokenizer.encode(task["prompt"])
                tokens, behavior, stop = generate(model, tokenizer, prompt, config["sampling"]["max_new_tokens"],
                    config["sampling"]["max_context"], config["training"]["seed"] + step * config["training"]["batch_size"] + index)
                text = tokenizer.decode(tokens)
                teacher_scores = teacher.score_tokens(prompt, tokens, identity) if teacher else []
                row = {"schema_version": "math-onpolicy-batch-v1", "sample_id": f"{step:06d}-{index:06d}",
                       "task_id": task["task_id"], "task_digest": digest(task), "split": "train",
                       "student_behavior_version": version, "student_evidence": "observed",
                       "teacher_revision": teacher.revision if teacher else None,
                       "teacher_evidence": teacher.evidence_kind if teacher else None,
                       "identity": identity, "prompt_ids": prompt, "response_ids": tokens,
                       "behavior_logprobs": behavior, "teacher_logprobs": teacher_scores, "loss_mask": [1] * len(tokens),
                       "sampling": {"temperature": 1.0, "top_p": 1.0, "raw_logprobs": True}, "stop_reason": stop,
                       "response": text, "verification": verify(text, task["reference"]),
                       "prm": process_rewards(config["prm"]["kind"], tokens, tokenizer) if config["prm"]["kind"] != "none" else None}
                row["score_request_digest"] = digest(score_request(row))
                row["score_response_digest"] = digest({"request_digest": row["score_request_digest"], "logprobs": teacher_scores})
                row["digest"] = digest(row)
                validate_row(row, version=version, identity=identity, context=config["sampling"]["max_context"],
                             allow_synthetic=allow_synthetic, require_teacher=teacher is not None)
                publish(root / "batches" / f"{row['sample_id']}.json", row)
                rows.append(row)
            if world > 1:
                dist.barrier()
            result = optimize(model, optimizer, rows, config, tokenizer, world)
            after = parameter_digest(model)
            if teacher and not teacher.unchanged():
                raise ValueError("Teacher changed or received gradients")
            result.update(step=step + 1, rank=rank, student_behavior_version=version, updated_version=after,
                          parameter_changed=after != version, teacher_unchanged=True,
                          tokens=sum(len(r["response_ids"]) for r in rows), seconds=time.monotonic() - begin,
                          verification_counts={status: sum(r["verification"]["status"] == status for r in rows)
                              for status in ("correct", "incorrect", "unparseable", "unsupported")})
            publish(root / "metrics" / f"{step + 1:06d}-rank{rank}.json", result)
            metrics.append(result)
            if rank == 0:
                print(json.dumps(result), flush=True)
                if (step + 1) % config["training"]["save_interval"] == 0 or step + 1 == config["training"]["steps"]:
                    atomic_checkpoint(root / f"checkpoint-{step + 1:06d}.pt", {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step + 1, "identity": identity,
                        "signature": signature, "parameter_digest": after})
            if world > 1:
                versions = [None] * world
                dist.all_gather_object(versions, after)
                if len(set(versions)) != 1:
                    raise ValueError("Data-parallel ranks diverged")
        if rank == 0:
            summary = {"status": "completed", "completed_steps": config["training"]["steps"], "resumed_step": start,
                       "initial_parameter_digest": initial, "final_parameter_digest": parameter_digest(model),
                       "parameter_changed": initial != parameter_digest(model), "teacher_unchanged": True,
                       "world_size": world, "method": config["method"], "capability_improvement": "not_evaluated",
                       "synthetic_allowed": allow_synthetic, "nonzero_gradient_steps": sum(m["gradient_norm"] > 0 for m in metrics)}
            publish(root / "summary.json", summary)
            print(json.dumps(summary), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate(config, checkpoint, *, split=None):
    torch.set_num_threads(1)
    tasks = load_tasks(config["tasks"]["path"])
    model, tokenizer = load_student(config, torch.device(config["deployment"]["device"]))
    state = torch.load(checkpoint, map_location=next(model.parameters()).device, weights_only=True)
    if state["signature"] != training_signature(config, digest(tasks)) or state["identity"] != tokenizer.identity():
        raise ValueError("Evaluation checkpoint config/data/tokenizer mismatch")
    model.load_state_dict(state["model"])
    if parameter_digest(model) != state["parameter_digest"]:
        raise ValueError("Checkpoint digest mismatch")
    selected = [task for task in tasks if task["split"] == (split or config["tasks"]["eval_split"])]
    if not selected:
        raise ValueError("Empty held-out split")
    rows = []
    for index, task in enumerate(selected):
        tokens, _, _ = generate(model, tokenizer, tokenizer.encode(task["prompt"]), config["sampling"]["max_new_tokens"],
                                config["sampling"]["max_context"], config["training"]["seed"] + index)
        response = tokenizer.decode(tokens)
        rows.append({"task_id": task["task_id"], "response": response, "verification": verify(response, task["reference"])})
    return {"schema_version": "math-evaluation-v1", "checkpoint_digest": state["parameter_digest"],
            "split": split or config["tasks"]["eval_split"], "samples": rows,
            "accuracy": sum(r["verification"]["status"] == "correct" for r in rows) / len(rows),
            "scope": "pipeline smoke; not evidence of capability improvement"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config = load(args.config, allow_synthetic=args.allow_synthetic)
    try:
        train(config, allow_synthetic=args.allow_synthetic, resume=args.resume)
    except BaseException as exc:
        publish(Path(config["output"]["directory"]) / f"failure-rank{os.environ.get('RANK', '0')}.json",
                {"status": "failed", "type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
