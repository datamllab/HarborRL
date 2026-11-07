# Tutorial: CPU Hello World

This walkthrough exercises native task inspection, verifier rewards, trajectory
assembly, and GRPO batch export without a model, GPU, Docker daemon, or task dataset.

## Run it

From the repository root:

```bash
uv sync --extra dev
uv run harborrl hello --out-dir ./harborrl-hello
```

The output directory must not already exist. Omit `--out-dir` to let HarborRL create
a timestamped directory in the current working directory.

## What happens

1. HarborRL writes a valid single-step native task, locked catalog, and Claude Code
   profile.
2. Two scripted attempts write different content to the task target.
3. The actual verifier runs in a child process and produces rewards `0` and `1`.
4. Each attempt receives immutable identity, trace, and verifier-receipt evidence.
5. Native IR v2 is assembled and revalidated.
6. A two-trajectory training group is exported with negative and positive
   group-normalized advantages.

The walkthrough supplies fixture token IDs and log probabilities, so its trajectories
are `CONTRACT_READY`, not live `RL_READY` evidence.

## Inspect the result

```bash
uv run python -m json.tool harborrl-hello/hello-result.json
uv run python -m json.tool harborrl-hello/catalog.json
uv run python -m json.tool harborrl-hello/training-batch.json
uv run python -m json.tool harborrl-hello/attempt-1/native-ir.json
```

Important files include:

```text
harborrl-hello/
├── catalog.json
├── claude-code-profile.json
├── hello-result.json
├── training-batch.json
├── attempt-0/
│   ├── native-ir.json
│   ├── result.json
│   └── reward-receipt.json
├── attempt-1/
│   ├── native-ir.json
│   ├── result.json
│   └── reward-receipt.json
└── tasks/hello_world/
```

Check the exported rewards and advantages:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

batch = json.loads(Path("harborrl-hello/training-batch.json").read_text())
rows = batch["groups"][0]["trajectories"]
print([(row["reward"], row["advantage"]) for row in rows])
PY
```

Expected shape: the reward-0 trajectory has a negative advantage and the reward-1
trajectory has a positive advantage.

## Continue

Reuse the generated catalog and profile while learning the configuration wizard:

```bash
uv run harborrl init --out-dir ./my-first-config
```

Real training still requires a real model checkpoint, calculated fingerprints,
runner environment, and deployment acceptance.
