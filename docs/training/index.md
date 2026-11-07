# Training

HarborRL's native path turns audited Harbor trajectories into complete GRPO groups
for a pinned Slime backend. Configuration, rollout evidence, export contracts, and
backend launch are separate layers so each can be validated at the earliest useful
stage.

## Training stages

1. **Load configuration.** Resolve relative paths, apply repeatable overrides, and
   validate the selected MATH combination.
2. **Resolve a deployment.** Load an explicit schema-1 file, discover
   `./deployment.yaml`, or derive an embedded-local topology.
3. **Preflight.** `doctor` checks the relevant local and remote roles before a run
   directory is created.
4. **Materialize inputs.** Freeze the original task locks and prompt rows into the
   run directory without rewriting tasks, tests, or rewards.
5. **Collect trajectories.** Run isolated attempts against a policy-versioned
   gateway and verify response consumption plus Harbor verifier evidence.
6. **Export complete groups.** Recompute Native IR eligibility, rewards, advantages,
   token masks, and batch weights.
7. **Update and publish.** Launch the pinned Slime hooks, synchronize weights, and
   admit the next serving policy version.

## Commands and side effects

| Command | Remote probes | Creates run | Starts training |
| --- | ---: | ---: | ---: |
| `harborrl validate ...` | No | No | No |
| `harborrl train ... --dry-run` | No | No | No |
| `harborrl doctor ...` | Yes | No | No |
| `harborrl train ...` | Yes | Yes | Yes |

The real training command fails before launch if any required doctor check fails.

## Before a live run

- Replace all example paths, hostnames, and fingerprint placeholders.
- Lock task content and an exact Claude Code profile.
- Verify the backend commits/image and the target Python environments.
- Confirm shared storage and bidirectional network routes.
- Run a complete-group smoke scope before increasing concurrency.
- Treat checkpoint creation as execution evidence, not proof of learning quality.

Continue with [configuration](configuration.md), the
[native training workflow](native.md), and the [run artifact reference](../reference/artifacts.md).
