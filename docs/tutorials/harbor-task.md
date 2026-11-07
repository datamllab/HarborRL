# Tutorial: inspect and lock a Harbor task

This tutorial performs the static inspection used before a native task enters a
training catalog. Inspection never executes task instructions or verifier code.

## 1. Inspect the bundled task

Choose a new evidence directory:

```bash
uv run harborrl hub inspect \
  --tasks-dir examples/native/tasks \
  --dataset hello-world \
  --revision v0.1-example \
  --run-dir ./task-inspection
```

The command finds every `task.toml`, validates the native single-step shape, computes
a tree digest, and appends a UUID-named JSONL evidence file. It also writes
`task-inspection/report.md`.

```bash
uv run python - <<'PY'
import json
from pathlib import Path

for evidence in sorted(Path("task-inspection").glob("*.jsonl")):
    for line in evidence.read_text().splitlines():
        row = json.loads(line)
        print(row["task"], row["status"], row.get("task_digest"))
PY
```

For the bundled task, the status should be `NEEDS_PROBE`: static structure is valid,
but execution and reward behavior have not been tested.

## 2. Understand the statuses

| Status | Meaning |
| --- | --- |
| `NEEDS_PROBE` | Structurally eligible for a later runtime probe |
| `UNSUPPORTED` | A detected feature is outside the current task contract |
| `INVALID` | Required files or metadata are malformed/missing |

Changing a status label does not make a task trainable.

## 3. Create a lock record

For each accepted task, create exactly these fields:

```json
{
  "id": "hello_world",
  "revision": "v0.1-example",
  "path": "tasks/hello_world",
  "task_digest": "DIGEST_FROM_INSPECTION",
  "reward_profile": {
    "key": "reward",
    "offset": 0.0,
    "scale": 1.0,
    "raw_range": [0.0, 1.0]
  }
}
```

Use a JSON list for a `.json` catalog or exactly one record on every line for
`.jsonl`; blank JSONL lines are rejected. Paths resolve relative to the catalog. The
reward profile must match the verifier's actual artifact; it is not inferred from
task content.

The bundled
[JSON catalog](https://github.com/datamllab/HarborRL/blob/main/examples/native/catalog.json)
and
[JSONL catalog](https://github.com/datamllab/HarborRL/blob/main/examples/native/catalog.jsonl)
are complete examples.

## 4. Revalidate through a dry run

When a configuration loads the catalog, HarborRL recomputes each digest and rejects
changed task content:

```bash
uv run harborrl train examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  --dry-run
```

The dry run validates locks and materializes no run. A separate live task probe and
training acceptance are still required.
