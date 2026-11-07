# Run artifacts

HarborRL writes immutable or append-only evidence under a unique run directory. For
native training the root is:

```text
OUTPUT_ROOT/training/YYYYMMDD-HHMMSS-XXXXXXXX/
```

Service mode uses `deployment.storage.run_root`; embedded mode uses
`training.output.root`.

## Launch evidence

| Artifact | Purpose |
| --- | --- |
| `launch.json` | Resolved plan, run path, source evidence, materialized inputs |
| `effective-config.json` | Component IDs, environment mapping, versions, field usage |
| `effective-command.json` | Final executable argv after launcher expansion |
| `source-manifest.json` | Git commit, dirty flag, and source hashes |
| `source.patch` | Binary-capable diff of relevant tracked source changes |
| `source-new/` | Copies of relevant untracked source files, when present |
| `config/native_catalog.json` | Normalized locked task entries |
| `config/native_tasks.jsonl` | Materialized prompt rows consumed by the backend |

Source evidence covers HarborRL implementation, scripts, examples, and project
metadata. Provider credentials and proxy values are deliberately excluded.

## Rollout evidence

`native-rollouts/` contains group and attempt identities, gateway requests/responses,
trace seals, Harbor results, verifier artifacts, reward receipts, Native IR v2, and
group exports. Exact nesting follows run/group/slot identity rather than a
human-editable filename convention.

Key invariants include:

- task, harness, reward, sampling, tokenizer, and template identity;
- one locked policy version per group;
- confirmed response consumption;
- immutable request/response and verifier digests;
- no pending gateway requests at trace seal; and
- complete groups before training export.

## Policy and training lifecycle

| Artifact | Purpose |
| --- | --- |
| `policy-pool.json` | Currently admitted serving pool/version |
| `policy-pool-history.jsonl` | Append-only policy admission history |
| `gpu-launcher.sh` | Generated service-mode GPU launcher |
| `launcher.pid` | Detached GPU launcher process ID |
| `train-stdout.log` | Service-mode GPU process output |
| `exit-code` | Durable GPU launcher exit status |
| `control-readiness.json` | GPU-to-controller control route evidence |

HarborRL never treats a missing/quiet log as proof of progress. Query the target and
durable state explicitly.

## Consistency evidence

When consistency checks are enabled, `consistency/` contains immutable serving-turn
sidecars, sync references, pool admissions, and `batch-*.json` reports. These records
contain bounded numerical diagnostics, not raw prompt/response text.

## Cleanup debt

`cleanup-required.json` marks a resource whose deletion could not be confirmed. The
runtime scans for these markers before admitting new work. Inspect the referenced
Docker or Sandbox resource, confirm cleanup, and remove only the resolved marker;
there is no automatic deletion of potentially live resources during recovery.

See [runtime lifecycle and recovery](../rollout-runtime-lifecycle.md) for the full
procedure.
