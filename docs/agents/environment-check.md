# AI agent deployment and training verification protocol

This document is the authoritative protocol for an AI agent that helps a user
deploy HarborRL and verify whether the user's machines can run training. Follow
it before searching for unrelated instructions or improvising installation
steps.

## Practiced deployment workflow

Use this workflow as the default unless the user supplies a stricter site
process. It deliberately separates diagnosis, approved repair, and acceptance.

1. Ask for or infer the target topology and config path. Run only read-only
   checks until the user approves a change.
2. Return the structured report from this protocol, including every confirmed
   blocker and every `unknown` check. Do not replace the report with a prose-only
   summary.
3. Propose the smallest next command that removes one blocker. Stop and wait for
   explicit user consent before installing, downloading, pulling, mounting, or
   starting anything.
4. After each approved change, rerun the affected phase and the repository
   doctor. Never assume that a command succeeded.
5. For native training, finish with a doctor run and launcher dry run. Report
   static deployment as accepted only when both pass and every statically
   verifiable machine check is `ready`. An unstarted live gateway remains an
   explicit deferred item rather than a silent pass.
6. Start real training only after the user confirms resource consumption and the
   intended scope. A successful dry run is not permission to launch training.

## Acceptance levels

Use one of these levels in the final report so the user can tell whether the
agent verified the target they actually chose:

| Level | Meaning | Required evidence |
| --- | --- | --- |
| `cpu_development` | The controller can develop and test HarborRL | Phase 1 passes, `harborrl --help` works, repository tests pass |
| `deployment_preflight` | The configured cross-machine deployment is statically ready | Every statically verifiable role and machine check passes, doctor reports no failed checks, and the CLI dry run resolves the plan |
| `training` | A real training launch passed the user-selected smoke scope | Training artifacts or logs show the selected scope completed; list any explicitly deferred live checks separately |

Do not label a CPU-only result as `deployment_preflight`. Do not label a dry run
as `training`. Live-service checks may remain `unknown` only when they are not
part of the selected acceptance level; they must still appear in `not_checked`.
If evidence for the selected level is incomplete, its acceptance verdict is
`unknown` even when individual installation checks pass.

## Objective

Determine the readiness of each required machine role, report confirmed missing
dependencies and resources, and propose the smallest safe next action. Do not
equate “the repository cloned successfully” or “unit tests pass” with “native
GPU training is ready.”

Evaluate these roles separately:

| Role | Purpose | GPU requirement |
| --- | --- | --- |
| CPU development machine | Clone HarborRL, create the locked environment, run CLI and tests | No |
| Harbor rollout worker | Execute isolated Harbor tasks and verifiers in Docker | No NVIDIA GPU required |
| Native GPU training machine | Run Slime/Megatron/SGLang training and policy serving | Yes |
| Controller | Read the YAML config and launch HarborRL | No GPU requirement of its own, but it must reach every selected service |

A deployment may combine these roles on one machine or split them across
machines. In particular, Harbor task rollout workers can be CPU machines while
policy serving and training run on a GPU machine.

## Safety and evidence rules

1. Identify the target topology before changing the environment.
2. Start with read-only inspection commands. Do not install packages, pull
   images, download model weights, start services, or write configuration
   without the user's explicit consent.
3. Never expose or request provider API keys, SSH private keys, passwords, or
   tokens. Command output and paths are sufficient for this protocol.
4. Use three verdict values:
   - `ready`: every required check passed.
   - `not_ready`: at least one requirement is confirmed missing.
   - `unknown`: the agent lacks access, the user has not supplied a path or
     topology, or a check is intentionally deferred until a live service runs.
5. Never turn an unchecked item into a pass. `unknown` is a valid conclusion.
6. Record the exact command, exit status, and concise evidence for failures.
7. After environment changes, rerun the relevant checks rather than assuming
   the change succeeded.

## Required input

If the user does not already provide this information, ask only for the missing
items that block diagnosis:

- Whether the target is CPU development only, deployment/doctor validation, or
  an actual training run.
- Whether rollout workers and GPU training are on one machine or different
  machines.
- The training YAML path and deployment YAML path, when they exist.
- SSH host aliases and the absolute HarborRL checkout path on each rollout
   worker.
- The absolute paths of the model actor/reference checkpoints and the output
   filesystem when GPU training is in scope.

Do not ask for credentials. Existing local SSH configuration must be used.

## Phase 1: CPU development baseline

Run these read-only checks from the HarborRL checkout:

```bash
git rev-parse HEAD
git status --short
uv --version
uv lock --check
uv sync --extra dev --dry-run
```

Interpretation:

- `uv` must be version 0.9 or newer.
- `uv lock --check` must report that `uv.lock` is up to date.
- The dry run must be able to resolve Python 3.10 or newer and the locked
  development dependencies.
- Uncommitted changes are not an environment failure, but report them because
  they can make reproduction ambiguous.

With user consent, create or update the locked environment and verify the CLI:

```bash
uv sync --extra dev
uv run harborrl --help
```

Only after that setup should the agent run repository tests. A CPU-only machine
that passes these checks is `cpu_development=ready`; it is not necessarily
ready for native GPU training.

## Phase 2: Interpret the deployment topology

When a YAML config exists, read these fields before running commands:

- `harbor.python`: absolute Python interpreter used by every rollout worker.
- `harbor.version`: required Harbor version; the current native contract uses
  `0.23.0`.
- `harbor.workers`: SSH host aliases used by the controller.
- `gateway.advertised_url`: URL that rollout workers must reach.
- `gateway.host` and `gateway.port`: listener binding.
- `model.checkpoint` and `model.reference`: paths consumed by the GPU runtime.
- `deployment.num_gpus`, `actor_gpus`, `rollout_gpus`,
  `actor_tensor_parallel_size`, `rollout_gpus_per_engine`, and `layout`.
- `tasks.catalog`, `harness.profile`, and `output.root`.

For every absolute path, verify it on the machine that consumes it. Do not
assume that two machines share a filesystem merely because they can SSH to one
another. If a shared filesystem is claimed, verify that the path and expected
content are visible from both sides.

The remote runner is launched with a command equivalent to:

```bash
ssh -o BatchMode=yes -o ClearAllForwardings=yes \
  -o StrictHostKeyChecking=accept-new WORKER \
  'cd /absolute/Harbor-RL && exec /absolute/harbor/python -m harborrl.rollout.harbor_job.runner'
```

Therefore each rollout worker needs the HarborRL source tree at the same
absolute path used by the controller, unless the user provides a corrected
site-specific launch arrangement.

## Phase 3: CPU Harbor rollout worker checks

From the controller, check noninteractive SSH and worker identity first:

```bash
ssh -o BatchMode=yes WORKER 'uname -a && pwd'
ssh -o BatchMode=yes WORKER 'test -d /absolute/Harbor-RL && echo repo-present'
```

Then run the repository-native probe using the YAML worker and interpreter
values:

```bash
ssh -o BatchMode=yes WORKER \
  'cd /absolute/Harbor-RL && /absolute/harbor/python -m harborrl.rollout.harbor_job.runner --probe'
```

A passing probe must:

- exit with status 0,
- report Python `3.12.0` or newer,
- report `harbor_version: 0.23.0`,
- report usable Harbor `ClaudeCodeOptions`, `TrialConfig`, and `Trial` APIs.

Also check Docker from the worker's normal environment:

```bash
docker version
docker info
```

The Docker daemon must be reachable by the Harbor worker. A worker without an
NVIDIA GPU is acceptable; a worker without a usable Docker daemon is not.

Task images may only be discovered while executing a task. If an image has not
been pulled, report image availability as `unknown`, not as failed. Network or
registry failures observed during the first live task are rollout-worker
failures even though model training itself may be healthy.

Finally, verify DNS/routing for the hostname in `gateway.advertised_url`. If
the gateway is not running yet, report connectivity as `unknown`. A connection
refused or timeout after the gateway is running is a failure.

## Phase 4: Native GPU training checks

On the GPU machine, run:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
df -h .
```

Check all of the following:

- `nvidia-smi` exits successfully.
- The number of visible GPUs is at least `deployment.num_gpus`.
- For `layout: split`, `actor_gpus + rollout_gpus <= num_gpus`.
- For `layout: colocate`, `actor_gpus == rollout_gpus == num_gpus`.
- `actor_gpus` is divisible by `actor_tensor_parallel_size`.
- `rollout_gpus` is divisible by `rollout_gpus_per_engine`.
- The model attention-head configuration is compatible with both parallel
  sizes; `harborrl doctor` performs this check when the checkpoint is readable.
- The checkpoint/output filesystem has at least 128 GiB free unless the user
  explicitly changes the launcher's checkpoint minimum. Report lower free space
  as `not_ready`.

GPU count alone does not prove that a specific model fits. Unless model size,
checkpoint metadata, sequence length, and available accelerator memory give a
defensible estimate, report model memory fit as `unknown`.

The pinned backend stack is normally installed with:

```bash
bash scripts/bootstrap_backends.sh /path/to/harborrl-backends
. /path/to/harborrl-backends/harborrl-backend.env
```

Treat the bootstrap as a mutating, network-heavy operation and request consent
first. After sourcing `harborrl-backend.env`, verify:

- `SLIME_DIR` is a directory containing `train.py`.
- `MEGATRON_DIR` is a directory when set; it may be omitted when the official
  Slime image supplies Megatron.
- `SGLANG_IMAGE` names the pinned image expected by this repository.
- Docker can inspect the image; if not, mark it missing or not yet pulled.
- The training Python environment can import the packages and backend hooks
  checked by `harborrl doctor`.

## Phase 5: Cross-machine deployment checks

For a CPU-rollout/GPU-training split, verify each direction independently:

1. Controller to every Harbor worker: noninteractive SSH and the runner probe.
2. Every Harbor worker to `gateway.advertised_url`: hostname resolution and
   TCP connectivity once the gateway is running.
3. GPU machine to model/reference checkpoint paths and writable output root.
4. Controller to paths needed to materialize catalogs and launch artifacts.
5. Worker to every task path referenced by the catalog.
6. All machines to any image registry required by their assigned task.

Do not use localhost evidence from one machine to approve a service on another
machine. If the agent can inspect only the controller, mark remote checks as
`unknown` and tell the user which command must run on which host.

## Phase 6: Repository doctor

After the CPU environment exists and a YAML config is available, run from the
controller:

```bash
uv run harborrl doctor --config path/to/config.yaml \
  --deployment path/to/deployment.yaml
```

Omit `--deployment` only when the target intentionally uses the implicit
embedded/default topology. The native launcher wrapper has no `--deployment` option
and changes to the repository root, so use it only when that discovery behavior is
intended:

```bash
bash scripts/launch_native.sh --config path/to/config.yaml --doctor
```

For `deployment_preflight` acceptance, also run the safe launcher dry run and
retain its resolved plan as evidence:

```bash
uv run harborrl train path/to/config.yaml \
  --deployment path/to/deployment.yaml --dry-run
```

Requirements:

- Parse the JSON object from stdout; do not rely only on the exit code.
- Treat every check with `ok: false` as a failure.
- Preserve the check's `path`, `service`, `dependency`, `resource`, or `task`
  identity in the final report.
- Remember that `doctor` explicitly defers the live serving semantic probe.
   A passing `doctor` therefore validates configuration and static preflight,
   not a complete training outcome.

Do not start a training run merely because `doctor` passes. Ask the user for
the intended validation scope and confirm resource consumption first.

## Required agent report

End the diagnosis with a concise human summary followed by this JSON shape:

```json
{
  "protocol": "harborrl/environment-check",
  "protocol_version": 2,
  "repo_commit": "string or null",
  "verdict": "ready | not_ready | unknown",
  "acceptance": {
    "level": "cpu_development | deployment_preflight | training",
    "verdict": "ready | not_ready | unknown",
    "evidence": [
      "uv run harborrl --help exited 0",
      "uv run pytest exited 0"
    ]
  },
  "roles": {
    "cpu_development": "ready | not_ready | unknown",
    "harbor_rollout_worker": "ready | not_ready | unknown",
    "gpu_training": "ready | not_ready | unknown",
    "cross_machine_deployment": "ready | not_ready | unknown"
  },
  "machines": [
    {
      "name": "user-provided alias",
      "roles": ["controller", "harbor_rollout_worker", "gpu_training"],
      "verdict": "ready | not_ready | unknown"
    }
  ],
  "missing": [
    {
      "role": "gpu_training",
      "machine": "gpu-host",
      "item": "visible GPU budget",
      "severity": "required",
      "evidence": "nvidia-smi exited with 127",
      "fix": "Install the NVIDIA driver and rerun nvidia-smi."
    }
  ],
  "not_checked": [
    {
      "item": "live gateway connectivity",
      "reason": "gateway was not running during the initial diagnosis"
    }
  ],
  "doctor": {
    "executed": true,
    "passed": false,
    "failed_checks": []
  }
}
```

Omit roles that the user explicitly excludes from the target, and explain the
omission in the human summary. A top-level `verdict` is `ready` only when every
role required by the user's stated target is `ready`. Otherwise use
`not_ready` when at least one required role has a confirmed blocker, or
`unknown` when readiness cannot be established.

The `acceptance.verdict` is scoped to the selected acceptance level. For
example, `deployment_preflight` can be `ready` while the top-level verdict is
`unknown` because a live gateway semantic probe is intentionally deferred; that
deferred probe must remain visible in `not_checked`.
