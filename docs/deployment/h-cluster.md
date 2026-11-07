# H-cluster practice

The optional `practices/h_cluster` package adapts HarborRL to Brainbox and restricted
Sandbox egress. It reuses the core controller, RolloutService, Harbor runner,
trajectory audit, and training path; it is not installed with the core package and
does not provide another algorithm.

## Topology

```text
GPU training adapter ──► developer Controller / RolloutService
                                  │
                                  ▼
                         local Harbor runner ──► Sandbox platform

Sandbox Claude ──► loopback mailbox ◄── developer relay pump
                                             │
                                             ▼
                                   developer Gateway ──► GPU SGLang
```

The developer initiates authenticated platform-proxy requests. A Sandbox instance
does not need a direct route to the GPU or developer-private listener. Relay
responses, including server-sent events, are fully buffered.

## Install

Use Python 3.12 or newer in the runner environment:

```bash
python -m pip install -e . -e practices/h_cluster
```

Start from `practices/h_cluster/configs/deployment.example.yaml` and
`practices/h_cluster/configs/sandbox.example.yaml`. Use service placement, a local
developer runner, and no SSH rollout workers. In the training file, set:

```yaml
harbor:
  runner_transport: local
  workers: []
  environment:
    runtime: sandbox
    config: /absolute/path/to/site/sandbox.yaml
```

Install the practice in the runner environment. The controller may use a separate
Python environment for tokenizer and service dependencies.

For every task image, create a matching platform Environment:

- expose command port `44772` and the configured relay port;
- set capacity high enough for the binding limit and use prewarm `0` for an idle
  test deployment;
- pin the image and offline Claude binary hash;
- prepare external task/verifier dependencies inside the image; and
- keep AK:SK credentials in the referenced mode-600 file, never in YAML or Git.

The model relay is not a general internet proxy. The developer must reach the
platform proxy and every advertised inference endpoint, and the GPU must reach the
controller.

## Validate locally

```bash
python -m pip install -e '.[dev]' -e practices/h_cluster
python -m pytest practices/h_cluster/tests
python scripts/sandbox_sdk_contract.py
```

These tests use local HTTP/mock contracts. They do not allocate platform instances or
prove GPU connectivity.

## Probe a configured site

After filling private site files, use the runner interpreter:

```bash
python -m harborrl.rollout.harbor_job.sandbox_probe \
  /absolute/path/to/site/sandbox.yaml

python -m harborrl.rollout.harbor_job.sandbox_probe \
  /absolute/path/to/site/sandbox.yaml --network

harborrl train /absolute/path/to/site/config.yaml \
  --deployment /absolute/path/to/site/deployment.yaml --dry-run
```

The first probe is local. `--network` performs authenticated, read-only Environment
checks and creates no instances. The dry run parses the plan without contacting the
GPU. Confirm actual routes before removing `--dry-run`.

Endpoint GET `502`, `503`, and `504` responses are retried within the creation
deadline; the allocation POST is never retried. A platform `Pending` state can still
time out and must not be reported as ready capacity.

## Migrate an experimental config

Flat experimental Brainbox configuration is intentionally not accepted by the
standard OpenSandbox schema:

1. Replace `provider: brainbox` with
   `adapter: harborrl_h_cluster.environment:BrainboxEnvironment`.
2. Move project and relay settings under top-level `options`.
3. Move Environment IDs and mount requirements under each binding's `options`.
4. Keep lifecycle limits and offline installation in their common top-level fields.
5. Keep the platform command port fixed at `44772`.

## Cleanup and scaling

After cancellation, inspect instance lifecycle evidence and every
`cleanup-required.json` marker. Verify both the live instance list and Environment
capacity before reuse; an empty list can disagree with capacity accounting. Never
delete templates or instances owned by another run.

Start with one serial complete group. Increase concurrency only after confirming
binding limits, project quotas, relay behavior, verifier dependencies, and cleanup.
Allocated or `Running` instances do not prove that an agent is executing, and a
checkpoint or policy-version change does not prove learning improvement.

## Read the full practice guide

The source package remains the authoritative implementation-specific guide and
configuration reference:

[H-cluster deployment practice](https://github.com/datamllab/HarborRL/tree/main/practices/h_cluster)

H-cluster hostnames, IDs, credentials, mounts, and numerical thresholds are
site-specific and intentionally absent from the core schema.
