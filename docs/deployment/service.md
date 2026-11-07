# Controller and SSH service deployment

Service mode separates the manager/controller, GPU training target, and rollout
workers. The controller hosts the RolloutService and policy gateway; rollout
providers execute tasks independently. Training dispatch always uses the SSH target;
`gpu_target.transport: local` resolves to localhost SSH rather than a local child.

## Start from the example

Copy and edit the
[service deployment example](https://github.com/datamllab/HarborRL/blob/main/examples/native/deployment-service.yaml).
All paths use target-machine semantics and must be absolute.

```bash
harborrl validate site/config.yaml \
  --deployment site/deployment.yaml --check-files

harborrl train site/config.yaml \
  --deployment site/deployment.yaml --dry-run

harborrl doctor site/config.yaml \
  --deployment site/deployment.yaml
```

## Required routes

| Source | Destination | Purpose |
| --- | --- | --- |
| GPU target | `manager.control_url` | Register run, submit groups, announce serving |
| Rollout workers | `manager.gateway_url` | Harness model requests |
| Manager | announced GPU inference endpoint | Policy serving control/data path |
| Manager | GPU target over SSH (including localhost for `local`) | Launch, poll, and cancel training |
| Manager | rollout workers over SSH | Probe and run `ssh_docker` workers |

Service manager origins must be non-loopback plain HTTP(S) origins matching their
declared ports.

## Shared filesystem

Only `shared_fs` storage is implemented. The configured `storage.run_root` must be
visible and writable as declared by the manager, GPU target, and workers. Bulky
trajectory and training data travel through that root; control responses carry paths
and SHA-256 digests.

Do not assume that identical path strings imply a shared mount. Verify visibility and
writability from every consuming machine.

## SSH behavior

SSH targets use the invoking user's SSH configuration and agent. Schema fields may
reference a restricted identity file, but never contain passwords or private-key
contents. Commands disable forwarding and use noninteractive operation.

The service GPU target writes `gpu-launcher.sh` into the shared run directory, starts it
detached, records its PID and exit code, and cancels its process group with escalating
signals. HarborRL does not automatically resubmit a failed training job. Only the
embedded path uses `LocalTrainingTarget` as a direct child process.

## Rollout providers

- `ssh_docker` requires at least one explicit worker with task roots, source checkout,
  runner Python, and Docker access.
- `sandbox` uses the optional environment provider and may have local or SSH runners.
- `embedded` preserves worker discovery from the training configuration.

Start with one complete group and inspect cleanup evidence before increasing
concurrency.
