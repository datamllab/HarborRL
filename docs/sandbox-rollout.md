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

Use `training.resume` to load a Slime checkpoint root, including optimizer and
scheduler state. Relative paths resolve beside the YAML file. The default is
`null`, which starts from the model checkpoint. `training.num_rollout` remains
the total target rollout count. A checkpoint at index `k` resumes at `k + 1`;
set the target to at least `k + 2` to perform another update. A target of
`k + 1` restores state without taking a new training step. Set
`training.max_tokens_per_gpu` (default `16384`)
for the training batch token budget and `training.max_ckpt_keep` (default `2`)
for checkpoint retention; both require positive integers. These options must
be declared in configuration: ambient `RESUME_LOAD`, `MAX_TOKENS_PER_GPU`, and
`MAX_CKPT_KEEP` variables are filtered by the local training target.

The launcher supervises Slime with `harborrl.platform.checkpoint_guard`; the
pinned backend silently ignores unknown retention flags. The supervisor checks
available space before launch (at least 128 GiB, or 1.15 times the resumed
checkpoint size), and removes older generations only after a tracker-published
successor has readable tensor archives and complete requested HF exports. It
records deletion manifests in `checkpoints/<run>/retention.jsonl`. These are
structural checks, not a full tensor restore, and the preflight is not a space
reservation: budget for a complete extra generation while the old one remains.

## Concurrency and evidence

`max_live_instances` bounds all active instances; each image binding has its
own limit. Binding capacity is acquired before global capacity to avoid
starvation. `startup_concurrency` separately bounds simultaneous SDK startup
using process-shared locks. Shared-filesystem flock support is required when
runners use the same lock directory across hosts.

Groups use the same identity, policy barrier, verifier and consumption audit
as Docker. Failed slots cancel and await their group siblings. Direct Slime
rollout batches also settle every remaining group before deciding whether
recovery is permitted. RolloutService requests wait for submitted independent
groups before rejecting a partial batch. Unconfirmed cleanup blocks new
admission. Production export never accepts synthetic evidence.

### Independent cleanup and admission debt

Each Sandbox runner registers a dedicated Linux session, inherited attempt
tag, PID start time, host and boot identity in `runner-process.json`. After the
runner exits, a separate process on the same host observes and terminates any
surviving owned processes. PID reuse, a different host or boot, and unreadable
possible descendants leave ownership unresolved. Protected processes older
than the runner are excluded. The observer also verifies its parent chain and
excludes those exact ancestor PIDs and start times, so a protected SSH service
that launched the observer cannot block cleanup just because it started after
the runner. A changed parent chain, the runner's session, or its inherited
attempt tag rejects the observer as nonindependent. Transient process read
errors retry a complete scan within the cleanup deadline; persistent errors
retain debt and their observations. Process sessions, `/proc` and POSIX signals
are required; this is a Linux runner contract.

The observer reads the attempt's locked runtime and lifecycle, then polls the
platform independently of the SDK. A known allocation is absent only after
GET returns 404, a complete and consistent paginated project inventory excludes
that ID, and the process proof succeeds. DELETE acknowledgement or a lifecycle
string alone cannot close cleanup. The observer can delete only the allocation
ID recorded for that attempt; an unknown POST outcome remains blocked for
ownership reconciliation. Unknown inventory entries are preserved.

Every observation is retained under `cleanup-reconciliations/`. A successful
observation publishes immutable `cleanup-reconciliation.json`, bound to the
runtime, lifecycle and process-owner digests. The original
`cleanup-required.json` markers and lifecycle history remain available; a
validated receipt explains their closure. Repeating a completed reconciliation
returns that receipt.

Image and global live capacity remain reserved while cleanup is unresolved.
Queued requests wake and fail when debt appears, including when all capacity
is retained. After a new independent reconciliation succeeds, the owning
runtime can call `sandbox_admission.reconcile(attempt_root)` to release the
retained capacity. Restarting into an old attempt is forbidden: inspect its
evidence, finish cleanup on the original runner host, then use a fresh protocol.
To invoke the observer manually on that host, supply the nonsecret resolved
configuration on stdin, for example from a private deployment script:

```python
import json
import subprocess
import sys
from harborrl.rollout.harbor_job.sandbox_config import load_sandbox

cfg = load_sandbox('sandbox.yaml')
subprocess.run([sys.executable, '-m', 'harborrl.rollout.harbor_job.reconcile',
                'ATTEMPT_ROOT'], input=json.dumps(cfg) + '\n', text=True, check=True)
```

Credentials remain in the configured permission-restricted file. Cleanup stderr
and receipts belong in the private run evidence directory.

### Whole rollout generation recovery

Sandbox Slime batches freeze task order, group and slot membership, sampling
parameters and the serving policy version before the first generation. A
retryable infrastructure failure invalidates the entire generation, discards
all partial sample IR/reward/response fields, and uses new batch, group, attempt
and trajectory identities for recovery. The task list and policy stay fixed.
Low rewards, protocol or audit errors, cancellation, seed budget exhaustion
and unresolved cleanup never trigger this recovery. A sibling terminal error
discovered during cleanup takes precedence over a retryable failure.

`HARBORRL_NATIVE_GENERATION_POLICY` accepts the positive integer limits
`max_attempts_per_rollout` (default 32) and `max_failed_groups_per_rollout`
(default 8), plus the nonnegative `max_reruns_per_seed` (default 2). The
failed-group limit blocks new attempts after it is exceeded; exhaustion of any
group's slot attempts also invalidates the generation immediately. Seed reruns
are shared across all batches, not reset for each batch.

`rollout-generations/` contains a frozen policy, each batch's input manifest,
generation reservation, failure and superseded records, and a single selected
generation. Only complete results from the expected tasks, slots, policy and
generation can be selected, after the normal training-export audit passes.
An attempt retry within a generation preserves its input-derived sampling seed.
A whole generation rerun changes the batch and group identity, so the existing
SGLang seed derivation produces a new sampling seed. The base training seed,
sampling parameters, tasks and policy stay frozen; recovery is independent of
reward and does not promise bit-for-bit replay across generations.
`generation-budgets/` persists attempt reservations and failed groups under a
shared-filesystem flock. Embedded and service workers use the same bound;
restarting a worker cannot reset it or increase the limits. Reopening an
existing rollout directory is rejected and requires a fresh run protocol.

### Reliability control plane

Sandbox rollouts also maintain an append-only `sandbox-reliability/events.jsonl`
journal under the run root. Every event has a monotonic sequence, UTC and
monotonic timestamps, attempt id, state, failure kind, node, allocation id,
budget snapshot, decision and sanitized detail. Consecutive events are linked
by SHA-256 digests, so rewriting or removing history is detectable. URLs,
credential-looking fields and long diagnostic strings are redacted before the
journal is written. The journal is independent from rewards: an invalid attempt
can be retried as a new attempt id, but its partial trajectory never enters a
training export.

Configure retry spacing and node cooling with
`HARBORRL_NATIVE_RELIABILITY_POLICY` as a JSON object. Supported keys are
`backoff_seconds`, `max_backoff_seconds`, `node_consecutive_failures` and
`node_cooldown_seconds`; `max_attempts_per_slot` remains governed by the
existing native attempt limit. A node reaching the consecutive platform-failure
threshold is unavailable until its cooldown expires. Configure the frozen seed
infrastructure budget with `HARBORRL_NATIVE_INFRA_BUDGET` using positive JSON
integers for `attempts`, `create_retries`, `cleanup_reconciliations` and
`storage_faults`, plus a positive finite `wall_clock_seconds`. Once any budget
is exhausted, the runtime records a fail-seed event and does not reserve another
attempt. These limits must be chosen before a run and cannot be increased after
observing rewards or held-out results.
The seed budget lock persists its original limits and absolute wall-clock
expiry, and journal replay restores the remaining counters. Removing the
budget when reopening a locked seed is rejected, even before its first attempt.
An infrastructure failure remains counted after cleanup succeeds; cleanup
receipts are additional events and do not replace failed outcomes.

The node-cooling ledger can exclude known nodes when candidates are available.
Server-controlled scheduling is still authoritative; this does not force the
platform to choose a different node. Unknown allocation ownership, cleanup
debt and exhausted budgets require reconciliation or a fresh protocol.

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
