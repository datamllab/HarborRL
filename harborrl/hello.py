"""A portable CPU walkthrough with real task/verifier execution and fixture tokens."""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from harborrl.data.harbor.native_inspector import inspect_native
from harborrl.export.native import export_training_batch
from harborrl.export.validate import batch_weight_sum
from harborrl.rollout.harbor_job.collector import collect
from harborrl.trajectories.native import (
    Identity,
    RewardProfile,
    assemble,
    digest,
    publish,
)

PROFILE = {"agent_timeout_sec": 120, "cli_version": "2.1.141", "max_turns": 4, "model": "policy",
           "setup_timeout_sec": 300, "verifier_timeout_sec": 60}
VERIFIER = r'''import json, sys
from pathlib import Path
target, output = map(Path, sys.argv[1:])
reward = int(target.is_file() and target.read_text().strip() == 'ok')
output.mkdir(parents=True, exist_ok=True)
(output / 'reward.json').write_text(json.dumps({'reward': reward}) + '\n')
print(json.dumps({'reward': reward}))
'''


def create_inputs(root):
    """Materialize a native task, its lock and a valid agent profile. No external dependencies."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    task = root / "tasks" / "hello_world"
    for part in ("tests", "environment"):
        (task / part).mkdir(parents=True, exist_ok=True)
    (task / "instruction.md").write_text("Create /hello containing exactly ok.\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'version = "1.0"\n[environment]\nos = "linux"\ncpus = 1\nmemory_mb = 4096\nstorage_mb = 4096\ngpus = 0\nallow_internet = true\n'
        '[agent]\nname = "claude-code"\nmax_turns = 4\n[verifier]\ntype = "shared"\ncommand = "tests/test.sh"\n', encoding="utf-8")
    (task / "tests" / "verify.py").write_text(VERIFIER, encoding="utf-8")
    script = task / "tests" / "test.sh"
    script.write_bytes(b'#!/bin/sh\nset -eu\npython3 /tests/verify.py /hello /logs/verifier\n')
    script.chmod(0o755)
    (task / "environment" / "Dockerfile").write_text('FROM python:3.12-slim\nWORKDIR /\n', encoding="utf-8")
    report = inspect_native(task)
    if report["status"] != "NEEDS_PROBE":
        raise ValueError(f"Hello World task inspection failed: {report}")
    publish(root / "catalog.json", [{"id": "hello_world", "revision": "hello-v1",
                                     "path": str(task), "task_digest": report["task_digest"],
                                     "reward_profile": asdict(RewardProfile())}])
    publish(root / "claude-code-profile.json", PROFILE)
    return task, report


def _trajectory(root, task, slot, content, task_digest):
    identity = Identity(run_id="hello-world", batch_id="cpu-walkthrough", group_id="hello-group",
                        slot_id=str(slot), attempt_id=f"attempt-{slot}", trajectory_id=f"trajectory-{slot}",
                        task_digest=task_digest, policy_version="hello-fixture-v1", harness_digest=digest(PROFILE),
                        reward_digest=RewardProfile().digest, sampling_digest=digest(SAMPLING))
    trial = root / identity.attempt_id
    trial.mkdir()
    target = trial / "hello"
    # Execute the fixed lesson in a child process; all writes stay inside this run directory.
    subprocess.run([sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])",
                    str(target), content], check=True, timeout=15)
    subprocess.run([sys.executable, str(task / "tests" / "verify.py"), str(target), str(trial / "verifier")],
                   check=True, timeout=15, capture_output=True, text=True)
    rewards = json.loads((trial / "verifier" / "reward.json").read_text())
    trial_id = f"hello-trial-{slot}"
    publish(trial / "result.json", {"id": trial_id, "finished_at": datetime.now(timezone.utc).isoformat(),
                                    "verifier_environment_mode": "shared", "verifier_result": {"rewards": rewards}})
    evaluation = collect(identity, trial_id, trial / "result.json", trial / "verifier", RewardProfile(),
                         evidence_kind="hello_world_verifier")
    publish(trial / "reward-receipt.json", evaluation)
    turn = {"identity": identity.to_dict(), "harbor_trial_id": trial_id,
            "request_id": f"request-{slot}", "response_id": f"response-{slot}", "engine_id": "hello-fixture",
            "input_ids": [1, 2, 3], "output_ids": [4, 5], "logprobs": [-0.2, -0.3],
            "logprob_semantics": "raw_model", "policy_version": identity.policy_version,
            "tokenizer_digest": digest("hello-fixture-tokenizer"), "template_digest": digest("hello-fixture-template"),
            "delivery": "consumed_confirmed", "consumption_ref": "hello-fixture#consumed", "role": "policy",
            "sampling": SAMPLING, "client_request": {"model": "hello-fixture"},
            "serving_input": {"input_ids": [1, 2, 3]},
            "response": {"content": [{"type": "text", "text": content}]},
            "finish_reason": "end_turn", "evidence_kind": "hello_world_fixture"}
    for key in ("client_request", "serving_input", "response"):
        turn[key + "_digest"] = digest(turn[key])
    seal = {"identity": identity.to_dict(), "harbor_trial_id": trial_id, "pending_requests": 0,
            "status": "sealed", "request_ids": [turn["request_id"]], "turns_digest": digest([turn]), "errors": []}
    ir = assemble(identity, trial_id, [turn], seal, evaluation)
    publish(trial / "native-ir.json", ir)
    print(f"  Attempt {slot + 1}: wrote {content!r}; verifier reward={rewards['reward']}; {ir['readiness']}", flush=True)
    return ir


SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 8, "stop": []}


def run(root):
    root = Path(root).expanduser().resolve()
    print("HarborRL Hello World | CPU walkthrough", flush=True)
    print("Runs a scripted task and its real verifier, then exports a GRPO batch with fixture tokens.", flush=True)
    print("No model inference, Sandbox allocation or GPU weight update is performed.", flush=True)
    task, report = create_inputs(root)
    print(f"[1/3] Created native task, locked catalog and agent profile: {root}", flush=True)
    print("[2/3] Execute incorrect and correct attempts, collect verifier receipts", flush=True)
    group = [_trajectory(root, task, index, content, report["task_digest"])
             for index, content in enumerate(("try again", "ok"))]
    batch = export_training_batch([group], allow_synthetic=True)
    data = asdict(batch)
    data["policy_versions"] = sorted(data["policy_versions"])
    publish(root / "training-batch.json", data)
    rewards = [unit.reward for unit in batch.groups[0].trajectories]
    advantages = [unit.advantage for unit in batch.groups[0].trajectories]
    if rewards != [0.0, 1.0] or not advantages[0] < 0 < advantages[1] or batch_weight_sum(batch) != 1.0:
        raise ValueError("Hello World reward/export contract failed")
    summary = {"status": "passed", "mode": "cpu-walkthrough", "evidence": "scripted-task-and-fixture-tokens",
               "rewards": rewards, "advantages": advantages, "output_root": str(root),
               "catalog": str(root / "catalog.json"), "profile": str(root / "claude-code-profile.json")}
    publish(root / "hello-result.json", summary)
    print("[3/3] PASS: exported training-batch.json; rewards=[0, 1]; group-normalized advantages verified", flush=True)
    print(f"Artifacts: {root}\nNext: harborrl init --out-dir <new-config-dir> (use the catalog and profile above).", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harborrl hello", description=__doc__)
    parser.add_argument("--out-dir", default=None, help="New artifact directory; defaults to ./harborrl-hello-<timestamp>")
    args = parser.parse_args(argv)
    root = args.out_dir or (
        "harborrl-hello-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # noqa: DTZ005
    )
    try:
        run(root)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
