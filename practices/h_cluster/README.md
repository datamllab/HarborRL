# H-cluster deployment practice

This optional package adapts HarborRL to Brainbox and restricted Sandbox egress.
It shares the core Controller, scheduling, Harbor Runner and trajectory audit;
it does not implement another training loop. Installing HarborRL alone does not
install or import this package. All example addresses and paths are placeholders.

## Topology

```text
GPU training adapter -> developer Controller / RolloutService
                               -> local Harbor Runner -> Sandbox platform
Sandbox Claude -> loopback mailbox <- developer relay pump
                                          -> developer Gateway -> GPU SGLang
```

The developer initiates authenticated platform-proxy requests. Sandbox needs
no direct route to the GPU or developer private listener. The developer must
reach the platform proxy and every advertised inference endpoint; the GPU
must reach the controller. Relay responses, including SSE, are fully buffered.

## Install and configure

Use Python 3.12+ for the runner. From the repository root:

```bash
python -m pip install -e . -e practices/h_cluster
```

Install the practice in the runner environment. The controller can use a separate
Python environment with the dependencies required for its tokenizer and service.
Start with `configs/deployment.example.yaml` and `configs/sandbox.example.yaml`.
Use service placement, a local developer runner and no SSH rollout workers.
Set the training config's `harbor.environment.runtime` to `sandbox`, its
`config` to your Sandbox config, and `harbor.runner_transport` to `local`.

For each task image create a matching platform Environment. Expose command
port 44772 and the configured relay port. Its capacity must cover the binding
limit; use prewarm 0 for an idle test deployment. Pin images and offline Claude
binary hashes. The all-zero example hash must be replaced. Put AK:SK credentials
in the referenced mode-600 file; never commit credentials or private site files.
Tasks and verifiers needing external packages must have those dependencies
prepared offline. The model relay is not a general internet proxy.

The core only interprets common lifecycle settings. This package validates
project, Environment IDs, mounts and relay options. Whole-Gi memory values are
expressed using Gi. Endpoint GET 502/503/504 responses are retried within the
creation deadline; the allocation POST is never retried. Platform Pending can
still time out and must not be mistaken for ready capacity.

## Verify and run

No GPU or platform credentials are needed for the package's local HTTP/mock tests:

```bash
python -m pip install -e '.[dev]' -e practices/h_cluster
python -m pytest practices/h_cluster/tests
python scripts/sandbox_sdk_contract.py
```

After filling in private configuration, use the runner interpreter:

```bash
python -m harborrl.rollout.harbor_job.sandbox_probe /srv/harborrl/site/sandbox.yaml
python -m harborrl.rollout.harbor_job.sandbox_probe /srv/harborrl/site/sandbox.yaml --network
harborrl train /srv/harborrl/site/train.yaml --deployment /srv/harborrl/site/deployment.yaml --dry-run
```

The first command is local. The second performs authenticated, read-only
Environment checks and does not create instances. The dry run parses the launch
plan and does not contact the GPU. Check actual routes before starting training.
Removing `--dry-run` launches the selected GPU job and can allocate Sandbox
instances. Use a new run directory for each test.

Inspect instance lifecycle files and cleanup-required records after cancellation.
An empty instance list can disagree with Environment capacity details; verify
both before reusing capacity. Never delete templates or instances belonging to
other runs. The framework records unresolved cleanup rather than assuming it
succeeded. It does not automatically replace platform Environment templates.

## GPU jobs and outside hosts

Two optional adapters cover machines the core deliberately does not know
about. Both are site adapters; the core only pins their contracts.

**Provisioned GPU jobs.** A deployment may declare `gpu_target.provision`
instead of a fixed `gpu_target.ssh`:

```yaml
gpu_target:
  id: trainer
  transport: ssh
  ssh: null
  provision:
    up: ["python", "-m", "harborrl_h_cluster.rjob", "up", "--config", "/srv/harborrl/site/rjob.json"]
    down: ["python", "-m", "harborrl_h_cluster.rjob", "down", "--config", "/srv/harborrl/site/rjob.json"]
```

The core runs `up` before anything dials the machine, reruns the deferred GPU
probes against the machine it obtained, and runs `down` in its `finally` path —
a failed launch, a cancelled wait or a crashed run all still release it. `up`
prints the machine record as its last stdout line: `host`, optional `user`,
`port`, `identity` and an opaque `handle`; any other key is rejected. `down`
receives the same record on stdin. Nothing secret belongs in either direction.

`rjob up` refuses to submit while a previous pod record exists under its
`workdir`: a second submit silently spends a second allocation. Release test
pods with `down`; the release outcome is published even when it fails, so a
leaked pod is visible instead of assumed clean.

**A host the cluster cannot dial.** When a machine has no inbound route but can
reach the hub, it serves shard ports through reverse forwards:

```bash
python -m harborrl_h_cluster.tunnel install --config /srv/harborrl/site/tunnel.json
```

It prints two commands as JSON, because one paste cannot do both ends: the
host's key is not authorized yet, so it cannot reach the hub until the hub
side runs. Run `hub` on the hub (it authorizes the public key, idempotently),
then run `host` in the outside host's interactive session (it pulls the
package over the same one-way path, writes the site config, and starts the
supervised daemon). Both are idempotent and carry no private key material.

The network portal swallows non-interactive input, so that paste is the only
installation channel; the platform may wipe the host's `/root` and take the
installed key with it, in which case the daemon's state file names the failure
and the pair must be run again. Declare the shard in
`manager.gateway_shards` pointing at the hub's `127.0.0.1:<hub_port>`, and keep
`gateway_hosts` for elastic growth; the two may coexist.

Such a host has no shared storage. `push` copies declared directories back to
the hub over the same ssh direction with bounded retries, and never exposes a
partially written tree.

## Configuration migration

Experimental flat Brainbox configs are intentionally not accepted as standard
OpenSandbox settings. Replace `provider: brainbox` with
`adapter: harborrl_h_cluster.environment:BrainboxEnvironment`; move project and
relay settings under `options`, and Environment ID/mount requirements under
each binding's `options`. The platform always uses command port 44772.
Keep lifecycle limits and offline installation in their common top-level fields.

## Operational limits

Set concurrency within project quotas and each Environment's actual capacity.
Measure agent execution separately from allocated or Running instances.
Pending allocations, offline verifier dependencies, turn limits and invalid
generated tools can reject a complete training group. Session content must
match the recorded model response to pass consumption auditing.

Start with a serial complete-group run before increasing concurrency. Inspect
rejected groups and confirm cleanup of every allocated instance. Checkpoint
creation and policy-version changes alone do not establish learning improvement.
