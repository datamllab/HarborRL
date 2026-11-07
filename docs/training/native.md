# Native training workflow

Native training requires site-prepared task inputs, checkpoints, worker runtimes,
shared storage, and the pinned Slime/Megatron/SGLang stack. HarborRL does not download
those resources during validation or launch planning.

## Prerequisites

- A schema-2 training configuration with real tokenizer/template fingerprints.
- A locked JSON or JSONL task catalog and exact Claude Code profile.
- A schema-1 deployment, or an intentional embedded-local deployment.
- Python 3.12+, Harbor `0.23.0`, Docker, and task files on each Docker rollout worker.
- A GPU target with the requested visible GPU budget, checkpoints, backend commits,
  SGLang image, and output filesystem.
- Required network routes and, for service mode, shared storage visible at the same
  configured run root.

## 1. Bootstrap the backend stack

```bash
bash scripts/bootstrap_backends.sh "/absolute/path/to/harborrl-backends"
. "/absolute/path/to/harborrl-backends/harborrl-backend.env"
```

Review and approve this network-heavy step before execution. Existing generated
environment file is then sourced by the following commands.

## 2. Resolve the plan

```bash
uv run harborrl train /absolute/path/to/config.yaml \
  --deployment /absolute/path/to/deployment.yaml \
  --dry-run
```

Dry-run validates the configuration, profile, and task locks and prints the plan. It
does not require backend environment variables, contact workers, create a run
directory, or start services.

## 3. Run preflight

```bash
uv run harborrl doctor /absolute/path/to/config.yaml \
  --deployment /absolute/path/to/deployment.yaml
```

Doctor runs the checks relevant to the selected topology. For an SSH GPU target,
machine-local backend imports are enforced again by the GPU launcher rather than
being inferred from the controller.

## 4. Start a bounded run

```bash
uv run harborrl train /absolute/path/to/config.yaml \
  --deployment /absolute/path/to/deployment.yaml
```

This command can start remote processes, consume GPU capacity, run untrusted task
instructions, and allocate Docker or Sandbox resources. Coordinate resource use and
cleanup before launching.

For embedded mode, HarborRL creates a new timestamped run directory under
`output.root`, freezes the launch/effective configuration, materializes catalog and
prompt data, and starts a local training target. Service mode starts the controller
and always dispatches through the SSH training target; a schema `local` GPU target
therefore requires working localhost SSH. Use embedded mode for a direct local child.

`scripts/launch_native.sh` can bootstrap and source backends for the embedded/default
discovery path, but it accepts no `--deployment` argument and changes to the
repository root before invoking HarborRL. Prefer the direct commands above for an
explicit or service deployment.

## 5. Inspect evidence

Check at least:

- `launch.json` and `effective-config.json` for the frozen plan and environment;
- command and source audit records;
- native rollout traces, verifier receipts, and cleanup markers;
- training/GPU stdout and exit-code records;
- policy-pool transitions and consistency reports when enabled; and
- checkpoints at the configured save interval.

See [run artifacts](../reference/artifacts.md) and
[lifecycle recovery](../rollout-runtime-lifecycle.md).

!!! note "A completed checkpoint is not a quality claim"

    Report the exact task set, policy versions, rewards, acceptance scope, hardware,
    and backend versions. Learning improvement requires separate analysis.
