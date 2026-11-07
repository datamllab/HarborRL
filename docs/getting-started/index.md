# Getting started

HarborRL separates portable CPU contracts from site-specific GPU and worker
requirements. You can learn the data flow and validate a configuration before
installing the native backend stack.

## Choose a path

| Goal | Start here | Requires |
| --- | --- | --- |
| Understand the pipeline | [CPU Hello World](../tutorials/hello-world.md) | Python 3.10+ |
| Prepare a real configuration | [Configuration CLI](../configuration-wizard.md) | Existing task/profile/model paths |
| Inspect Harbor tasks | [Task tutorial](../tutorials/harbor-task.md) | Local Harbor-format task files |
| Resolve a native launch plan | [Native dry run](../tutorials/native-dry-run.md) | Example files only |
| Check a target deployment | [Environment verification](../agents/environment-check.md) | Access to configured machines |
| Start GPU training | [Native training](../training/native.md) | Pinned backend stack, GPUs, workers |

## The core flow

1. A strict schema-2 training file selects one registered model, algorithm, task
   provider, and harness.
2. A strict schema-1 deployment file places the controller, GPU target, shared
   storage, and rollout workers. If it is omitted, HarborRL derives an embedded-local
   topology.
3. The controller validates the configuration and builds a launch plan.
4. Harbor runners execute locked tasks. The policy gateway records requests,
   responses, raw log probabilities, policy versions, and consumption evidence.
5. Verifier rewards and complete groups are exported to the pinned Slime GRPO
   backend; equal-reward groups produce zero advantages.

## What each check proves

| Check | What it establishes | What it does not establish |
| --- | --- | --- |
| `harborrl validate` | Offline schema, semantic, and cross-file consistency | Connectivity, GPU capacity, model loading |
| `harborrl train ... --dry-run` | Referenced inputs load and a launch plan resolves | Remote readiness or resource availability |
| `harborrl doctor` | Configured environment and topology checks | Training quality or a completed run |
| CPU test suite | Scheduling, evidence, export, and lifecycle contracts | Live CUDA, SGLang, or model numerics |
| Live training evidence | The selected smoke scope actually ran | Generalized learning improvement |

!!! warning "A dry run is intentionally side-effect free"

    It does not contact remote machines or allocate GPU/Sandbox resources. Run
    `doctor` for live checks and obtain operator approval before starting training.

## Supported platforms

The core package supports Python 3.10 or newer. CPU development and the Hello World
walkthrough do not require Linux. Native GPU training, Docker-based Harbor workers,
and the optional Sandbox runner have additional platform and Python requirements;
see [installation](installation.md) and [deployment](../deployment.md).
