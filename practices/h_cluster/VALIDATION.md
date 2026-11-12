# Merged Controller and Sandbox validation

Validation date: 2026-10-22. This is an infrastructure acceptance record, not a
model benchmark. Private addresses, credentials, task artifacts and platform IDs
are intentionally excluded.

## Setup

- One developer host runs Controller, RolloutService, Gateway and Harbor Runner.
- The optional H-cluster adapter creates real instances through OpenSandbox SDK
  0.1.9. Harbor 0.23.0 runs a pinned Claude Code 2.1.141 binary in each instance.
- Sandbox model requests use the developer mailbox relay. The developer reaches
  GPU SGLang; Sandbox does not need a direct GPU route.
- Eight H200 GPUs run Qwen3-8B: four actor GPUs with TP4 and four serving GPUs
  with TP4. All tests retain strict serving and consumption evidence checks.

## Observed results

| Test | Running instance peak | Actual agent execution peak | Result |
|---|---:|---:|---|
| Serial baseline | 1 | 1 | Four real trajectories, two training iterations and two checkpoints |
| SDK command capacity probe | 8 ready | Not an agent test | Eight concurrent ten-second commands completed in 11.037 seconds |
| Four-dataset stress | 4 | 2 | Batch rejected; four valid trajectories, three failed groups |
| Eight-dataset stress | 6 | 3 | Batch rejected; four valid trajectories, six failed groups |
| Three-dataset regression | 6 | 6 | Batch rejected after an agent API error |
| Two-dataset regression before actor fix | 6 | 6 | Six valid trajectories; actor incorrectly rejected two groups' weights |
| Identical regression after actor fix | 6 | 6 | First complete batch trained and saved a checkpoint; second batch hit an unknown generated tool |

The two-dataset regression uses AARR and an existing HumanEvalFix task, three
samples per group, two groups per batch, a total instance limit of six, three
instances per Environment, startup concurrency four, and a 32-turn CLI limit.
These tasks were selected after the broad stress tests to isolate infrastructure
behavior. They are not an unbiased dataset sample. Each test starts from the
reference model; successful subsets from rejected batches are never repackaged
as new training batches.

The SDK path demonstrably supports three simultaneous instances from one
Environment in this deployment. This does not override per-Environment capacity,
project quotas or platform scheduling. Pending allocations can take minutes.

## Core fixes

- `9a84641a`: port preflight handles immediate HTTP-server restarts after
  TIME_WAIT while rejecting an existing listener. Real socket tests cover both.
- `dc208dc9`: actor postprocessing validates each group's local weight budget
  using its unique rollout ID, rather than limiting a multi-group batch to one.
  Partial actor shards remain valid; excess weight in one group cannot be hidden
  by another group's smaller weight. Regression tests cover both properties.

After these fixes, all 217 CPU tests passed, including the optional practice
tests. Changed Python files passed Ruff and the public-tree audit passed.

## Limits

Stable eight-agent mixed-dataset training has not passed. Observed failures
include Pending/startup errors, missing offline verifier dependencies, CLI turn
limits, unknown generated tools, and CLI tool-argument normalization that fails
exact consumption auditing. The relay is not a general internet proxy.

In the second batch of the final regression, the model generated an undeclared
`finish` tool. Gateway returned a protocol error and the trial was rejected even
though its verifier reward was positive. This must not be bypassed or converted
into a synthetic successful trajectory.

Successful training batches had uniform rewards within each group and zero
gradient norm. Optimizer execution, checkpoint creation and serving policy
version changes establish workflow execution, not learning improvement or
nonzero-gradient scaling correctness. The current `native_group_count` training
metric rounds microbatch weights and must not be used to count complete groups;
use RolloutService group results and audited trajectories instead.

To repeat this test, configure private site files using the practice examples,
pin task/image/binary digests, start with a serial complete-group test, then
increase binding and global concurrency within actual platform limits. Record
agent start/end intervals separately from instance states. Retain rejected
groups and verify deletion of every allocated instance after completion.
