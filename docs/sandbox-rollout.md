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

## Concurrency and evidence

`max_live_instances` bounds all active instances; each image binding has its
own limit. Binding capacity is acquired before global capacity to avoid
starvation. `startup_concurrency` separately bounds simultaneous SDK startup
using process-shared locks. Shared-filesystem flock support is required when
runners use the same lock directory across hosts.

Groups use the same identity, policy barrier, verifier and consumption audit
as Docker. Failed slots cancel their group siblings; submitted independent
groups finish before a partial batch is rejected. Unconfirmed cleanup blocks
new admission. Production export never accepts synthetic evidence.

## Checks

`python scripts/sandbox_sdk_contract.py` checks installed APIs and fake lifecycle
cleanup without contacting a platform. `sandbox_probe` checks the selected
adapter, SDK, credential-file permissions and optional binary locally.
Its explicit `--network` mode is adapter-specific and reports skipped checks.

`python scripts/sandbox_smoke.py CONFIG --output NEW_DIRECTORY --gateway URL`
is a live allocation test and consumes platform resources. A successful CPU
test or dependency probe does not establish live training acceptance.

For Brainbox and developer-mediated networking, see the independently
installable [H-cluster practice](deployment/h-cluster.md). H-specific
configuration from the experimental branch must be migrated as described there.
