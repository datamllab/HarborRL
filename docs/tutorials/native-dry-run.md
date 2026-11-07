# Tutorial: native training dry run

Use the bundled files to resolve the exact MATH selection and launch mapping without
GPU dependencies or remote access.

## 1. Inspect supported choices

```bash
uv run harborrl options models --json
uv run harborrl options algorithms --json
uv run harborrl options tasks --json
uv run harborrl options harnesses --json
```

## 2. Run offline validation

```bash
uv run harborrl validate examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  --json
```

The expected overall status is `valid_with_deferred_checks`. The example contains
placeholder machine/model paths, and offline validation deliberately defers live
capacity and connectivity.

## 3. Resolve the launch plan

```bash
uv run harborrl train examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  --dry-run > /tmp/harborrl-plan.json

uv run python -m json.tool /tmp/harborrl-plan.json
```

The plan should select:

| Field | Expected value |
| --- | --- |
| Model | `qwen` |
| Algorithm | `grpo` |
| Task provider | `native_harbor` |
| Harness | `claude_code` |
| Backend contract | `slime-v0.3.2-native-v1` |
| Layout | `split` |

The plan also contains the normalized training configuration, deployment digest,
backend environment mapping, and final training launcher path.

## 4. Try a validated override

```bash
uv run harborrl train examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  --set training.num_rollout=2 \
  --set sampling.groups_per_batch=2 \
  --dry-run > /tmp/harborrl-plan-overridden.json
```

Values are YAML-decoded and revalidated. An unknown field or invalid GPU/group
constraint exits nonzero.

## What this tutorial does not prove

No remote machine is contacted, no run directory is created, and no model or GPU is
loaded. Replace placeholders, run `doctor`, and follow the
[native workflow](../training/native.md) before a live launch.
