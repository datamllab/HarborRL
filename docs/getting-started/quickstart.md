# Quick start

This path starts with a CPU-only executable lesson, inspects the registered support
surface, and finishes with a side-effect-free native launch plan.

## 1. Install

```bash
uv sync --extra dev
```

## 2. Run Hello World

Choose a directory that does not already exist:

```bash
uv run harborrl hello --out-dir ./harborrl-hello
```

Expected final output includes `PASS`, rewards `[0, 1]`, and the location of the
artifacts. This executes a real local verifier and exports a GRPO batch with fixture
tokens. It does not call a model or update weights.

Inspect the summary and batch:

```bash
uv run python -m json.tool harborrl-hello/hello-result.json
uv run python -m json.tool harborrl-hello/training-batch.json
```

## 3. Query the registry

```bash
uv run harborrl list models
uv run harborrl list algorithms
uv run harborrl list tasks
uv run harborrl list harnesses
```

`list` always emits JSON. For field descriptions, datasets, deployment presets, and
compatibility constraints, use the richer offline catalog:

```bash
uv run harborrl options
uv run harborrl options fields --json
```

## 4. Validate the bundled configuration

```bash
uv run harborrl validate examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  --json
```

The example is structurally valid but contains placeholder model and environment
paths. A `valid_with_deferred_checks` result is expected: offline validation does not
claim that the target GPU, services, or checkpoints are ready.

## 5. Resolve a launch plan

```bash
uv run harborrl train examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  --dry-run
```

The command loads the harness profile and task catalog, applies schema validation,
selects the registered MATH components, and prints the resolved plan. It performs no
remote probes and starts no training process.

## Next steps

- Build a site-specific bundle with the [configuration CLI](../configuration-wizard.md).
- Learn the exact [MATH compatibility rules](../math/index.md).
- Follow the [native training workflow](../training/native.md).
- Choose an [embedded](../deployment/embedded.md) or
  [service](../deployment/service.md) topology.
