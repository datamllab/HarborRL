# Tasks

HarborRL consumes native Harbor tasks and locks them into either a JSON list or a
JSONL catalog. It does not maintain benchmark-specific conversion formats.

## Native task layout

```text
task/
├── task.toml
├── instruction.md
├── environment/
│   └── Dockerfile
└── tests/
    └── test.sh
```

`task.toml` declares the environment, agent, resources, and verifier.
`instruction.md` becomes the agent prompt. The verifier writes a reward artifact
under `/logs/verifier/` when Harbor executes the task.

## Registered providers

| Provider | Selected when | Catalog encoding |
| --- | --- | --- |
| `native_harbor` | `tasks.catalog` does not end in `.jsonl` | One nonempty JSON list |
| `harbor_jsonl` | `tasks.catalog` ends in `.jsonl` | One task lock on every line; blank lines are invalid |

Both formats require exactly these fields per task:

```json
{
  "id": "hello_world",
  "revision": "v0.1-example",
  "path": "tasks/hello_world",
  "task_digest": "2293cd4682597d9c6c5a4bb35d2ded9978873a8deb48ee07fae4e58121c79691",
  "reward_profile": {
    "key": "reward",
    "offset": 0.0,
    "scale": 1.0,
    "raw_range": [0.0, 1.0]
  }
}
```

Paths resolve relative to the catalog. IDs must be unique, revisions nonempty, and
digests 64-character lowercase SHA-256 values. Loading a catalog recomputes each
task's content digest and rejects a changed or unsupported task.

## Inspect before locking

```bash
harborrl hub inspect \
  --tasks-dir /path/to/tasks \
  --dataset my-dataset \
  --revision pinned-revision \
  --run-dir ./task-inspection
```

Inspection is static and CPU-only. It writes immutable JSONL evidence plus a summary;
it does not execute the task, allocate Docker, or establish trainability. Follow the
[task tutorial](../tutorials/harbor-task.md) to turn valid inspection evidence into a
catalog.

## Dataset claims

Repository manifests distinguish bundled examples, planned sources, static checks,
and dynamic evidence. A collection name or static compatibility result does not mean
that HarborRL ships the tasks or has completed RL training on them. See
[compatibility](compatibility.md).
