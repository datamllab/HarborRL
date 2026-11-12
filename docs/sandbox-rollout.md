# Sandbox rollout

Sandbox is an optional execution environment under the common Controller,
RolloutService, Runtime and Harbor Runner. Docker execution remains the default.
The core does not import the SDK until an explicitly selected runner needs it.

## Standard OpenSandbox

Use Python 3.12 or newer for the runner and install `pip install -e '.[sandbox]'`.
Start with `examples/native/sandbox.yaml`. Set the service origin (without a
trailing /v1), a private credential-file reference and exact task image bindings.
Standard credentials are opaque tokens, not a platform-specific AK:SK format.

Select the environment in your training config:

```yaml
harbor:
  python: /opt/harborrl/runner/bin/python
  version: 0.23.0
  runner_transport: local
  workers: []
  environment:
    runtime: sandbox
    config: /srv/harborrl/site/sandbox.yaml
```

For service placement, use `rollout.provider: sandbox` in the deployment file.
A local runner requires an empty deployment worker list. SSH runners require
explicit workers, the same shared task/evidence paths and the adapter installed
on each runner. Runner placement is independent of the environment provider.
The target and route must be validated before claiming that an SSH/Sandbox
deployment works on a real platform.

The standard adapter uses Harbor's OpenSandbox implementation and adds durable
allocation tracking, bounded lifecycle operations and cleanup-debt records.
Custom trusted adapters subclass ManagedOpenSandboxEnvironment and are selected
through `adapter: module:Class`; arbitrary task input cannot select code.
Provider settings belong in `options` and per-image binding `options`.
They must not contain inline secrets. Core plan construction does not import
the adapter, read credentials, connect to a platform or allocate instances.

The optional `installation` block supports a pinned offline Claude binary:
`mode: offline_binary`, `version`, `binary`, and its 64-character `sha256`.
Without it, Harbor performs normal agent setup, which can need network access.

For original tasks that define a Dockerfile without a prebuilt image, a binding
can include `task_digests: [SHA256]`. These are HarborRL task tree digests,
including file bytes and modes, rather than registry archive digests. Build the
task's original environment, pin the resulting image in the binding, and keep
the task files unchanged. The runner applies that image to a copy of Harbor's
in-memory environment config. A digest-bound image only accepts the listed
tasks; changed content cannot fall back to the image name. Each task digest
may appear in only one binding. Bindings without `task_digests` retain exact
image-name matching.

The Claude Code harness profile accepts optional `append_system_prompt` as a
nonempty string. Harbor passes it literally to the pinned CLI's
`--append-system-prompt` option. Use it for explicit harness conventions when
the original task expects a different tool interface, and freeze the same
profile for training and evaluation. The original task files and verifier stay
unchanged; tools execute through the native CLI's declared tool interfaces.
The option is omitted by default.

Optional `environment_timeout_sec` is a positive integer in the harness profile.
It sets Harbor's outer environment startup budget through the public
`environment_build_timeout_multiplier`, relative to the unchanged task's
`environment.build_timeout_sec` (Harbor's default is 600 seconds). Omitting it
preserves the task's startup budget. This is separate from `setup_timeout_sec`,
which covers agent installation, and from Sandbox SDK `create_timeout_s` and
`request_timeout_s`. For example, use `environment_timeout_sec: 720` to allow
a 600-second SDK creation budget plus runner overhead. Freeze all these
settings before training and evaluation.
The Runner reserves a process-shared startup slot in Harbor's public
`ENVIRONMENT_START` hook before the environment timer starts. Queue waiting
therefore does not consume `environment_timeout_sec`; the coordinator's total
attempt deadline still bounds the wait. The reservation is released after
creation/readiness or failure, allowing ready agents to run concurrently.
Startup lifecycle evidence distinguishes `waiting`, `reserved` and actual
allocation. `sandbox-requests/` records request method, lifecycle operation,
status or exception type, and elapsed time; it excludes raw URLs, credentials
and response bodies. Authentication failures remain fatal and are not retried.

For pinned Claude Code `2.1.141`, exhausting `max_turns` can produce a valid
`budget_truncated` trajectory. Acceptance requires the matching native CLI
initialization and final `error_max_turns` result, the configured number of
logical turns, a matching session, a final tool-use response,
the completed original verifier result and reward artifact, and confirmed
Sandbox cleanup. The original reward is preserved, including zero; the
trajectory is not retried for its reward. Other agent exceptions, timeouts,
missing responses and unverified CLI versions remain invalid.
The pinned CLI excludes token-limit continuations from its logical turn count.
Each additional response requires a consumed text-only `max_tokens` response
and the immediately following request with the same history, exact assistant
content and pinned native continuation message. Only probed ephemeral cache
placement and string/text-block serialization may differ. The audit records
both response IDs and their trace digests; every sampled token, logprob and
response remains in the trajectory. Arbitrary extra responses are rejected.

Native session consumption permits the CLI to add omitted optional tool
arguments only when they exactly match defaults in that request's declared
tool schema. The trace records the added defaults, schema digest and both
content digests. Sampled model output, token IDs and logprobs remain unchanged.
Changed arguments, added required fields and undeclared defaults are rejected.
The pinned CLI `2.1.141` also removes trailing ECMAScript whitespace on each
line of `Edit.new_string` and `Write.content` for non-Markdown paths. Session
audit accepts only this exact transformation, with matching profile/session
versions and a declared string field. Line separators, indentation, other
arguments and Markdown content remain exact. The trace records a versioned
`consumption_tool_normalization` proof, schema/value digests and both content
digests. The original sampled response and token/logprob arrays stay intact.
Other CLI input transformations still reject the attempt.

For pinned Claude Code `2.1.141`, Qwen calls to unavailable tools retain the
generated name, arguments, text, token IDs and logprobs. The CLI returns its
native `No such tool available` observation and can continue the episode.
The gateway accepts that history only with the exact matching error; session
audit independently requires one matching rejection for each unavailable call.
Missing, duplicate or successful results for unavailable tools reject the
attempt. The trace records `unavailable_tool_calls` and `tool_rejection_ref`.
Malformed tool JSON, invalid arguments, thinking output and unrelated API
errors still invalidate the attempt. Rewards come from the original verifier.

Sandbox creation and readiness failures write `sandbox-startup-failed.json`
with the stage, allocation ID, configured timeouts and exception cause types.
SDK error messages, credentials and signed URLs are excluded. A trial that
fails before creating a native session reports its Harbor exception type
before session auditing. Set `request_timeout_s` to cover slow endpoint
discovery within `create_timeout_s`; increasing the harness setup timeout
alone does not change either limit.

Set optional `training.seed` to an integer in `0..2147483647` to seed the native
trainer and task order and enable deterministic SGLang inference. Each policy
request derives its sampling seed from that seed, the batch, group, slot, task
digest and input tokens. Retries keep the same seed for the same input, and
concurrent dispatch order does not change it. The actual seed is recorded in
each turn's `serving_input.sampling_params`. External tool output and Sandbox
scheduling can still vary, so this does not guarantee identical complete agent
episodes. Omitting the option keeps the backend defaults.

Set `training.export_hf: true` to export HF weights under
`hf/rollout-<id>` at each checkpoint save, including the final rollout, for
independent model serving and evaluation. Optimizer checkpoints remain under
`checkpoints/`. The export option defaults to `false`.

## Concurrency and evidence

`max_live_instances` bounds all active instances; each image binding has its
own limit. Binding capacity is acquired before global capacity to avoid
starvation. `startup_concurrency` separately bounds simultaneous SDK startup
using process-shared locks. Shared-filesystem flock support is required when
runners use the same lock directory across hosts.

Groups use the same identity, policy barrier, verifier and consumption audit
as Docker. Failed slots cancel their group siblings. Direct Slime rollout
batches cancel and await all remaining groups and their slot cleanup before
closing the runtime and propagating a failure or cancellation. RolloutService
requests wait for submitted independent groups before rejecting a partial
batch. Unconfirmed cleanup blocks new admission. Production export never
accepts synthetic evidence.

## Checks

`python scripts/sandbox_sdk_contract.py` checks installed APIs and fake lifecycle
cleanup without contacting a platform. `sandbox_probe` checks the selected
adapter, SDK, credential-file permissions and optional binary locally.
Its explicit `--network` mode is adapter-specific and reports skipped checks.

`python scripts/sandbox_smoke.py CONFIG --output NEW_DIRECTORY --gateway URL`
is a live allocation test and consumes platform resources. A successful CPU
test or dependency probe does not establish live training acceptance.

For Brainbox and developer-mediated networking, see the independently
installable [H-cluster practice](../practices/h_cluster/README.md). H-specific
configuration from the experimental branch must be migrated as described there.
