# Deployment configuration and the controller placement

A `deployment.yaml`
describes where the manager, GPU training target and rollout workers live and
how they reach each other; the training YAML keeps every training-side field
(learning rate, sampling, task catalog) and is unchanged.

```bash
harborrl train train.yaml --deployment deployment.yaml
```

Without `--deployment`, a `deployment.yaml` in the working directory is picked
up; when no file exists the compatible embedded-local topology is used and the
legacy same-machine behavior is preserved bit-for-bit (recorded in the plan as
`deployment.source: embedded-local`).

## What the schema covers (v1)

- `manager` — the HarborRL controller node: `embedded` or `service` mode, bind
  address, control/gateway ports and the URLs other machines use to reach them.
  In `service` mode both URLs must be plain origins that match their ports and
  must not be loopback.
- `gpu_target` — the Slime training machine: `local` or `ssh` transport, the
  checkout (`source_dir`), training interpreter (`train_python`), an optional
  controlled `environment_file` (a JSON variable map, never a shell), and
  `expected_gpus`.
- `storage` — `shared_fs` mode with the shared `run_root` and writability
  expectations per role.
- `rollout` — `ssh_docker` workers: SSH destination, runner interpreter,
  Docker requirement and task roots. The `embedded` provider keeps worker
  discovery in the training config.
- `connectivity` — timeouts and which control-loop hops are required.

Secrets are rejected by the schema: no passwords, keys or API tokens. SSH
destinations resolve through the invoking user's SSH config, agent or a
restricted identity file. Paths are interpreted in target-machine semantics —
the manager never assumes its own `sys.executable` works on the GPU or workers.
See `examples/native/deployment-embedded.yaml` and
`examples/native/deployment-service.yaml`.

## Preflight graph

`train` and `doctor` share one check graph, executed at the earliest launch
stage. Each check records the role and machine it ran against, the outcome,
timing, and a user-executable fix. Any failed check exits nonzero before a run
directory or training process is created.

Layer order: manager local state (run root, token file, ports) → GPU target
(SSH reachability, paths, model paths, visible GPU budget) → rollout workers
(SSH, runner Python >= 3.12, pinned Harbor version, Docker, task roots) →
control-loop addresses (DNS resolution now, live health dials after components
start). `--dry-run` parses both configs and prints the resolved plan without
any remote probing or side effects.

When the GPU target is remote, machine-local dependency probes (Ray, Torch,
Slime imports, SGLang image, `nvidia-smi`) do not run on the manager: they are
enforced fail-closed by `slime_train.sh` at GPU job startup. The manager CPU
needs no GPU dependencies.

## TrainingTarget

`harborrl/deployment/training_target.py` launches one Slime job either as a
local child (legacy path) or over SSH (service mode). The SSH target writes a
generated `gpu-launcher.sh` into the shared run directory, starts it detached
(`setsid`) on the GPU machine, records the launcher pid, polls the shared
`exit-code` record, and cancels by signalling the remote process group
(INT → TERM → KILL). It never resubmits or restarts a job.

## RolloutService (manager mode)

`harborrl/rollout/service.py` is the manager-side control plane: HTTP JSON
with run-scoped tokens.

- `GET /healthz`, `GET /readyz`
- `POST /runs/{run_id}/register` — idempotent by manifest digest; the same id
  with a different manifest is a 409.
- `POST /runs/{run_id}/groups` — idempotent group submission. Same group id
  and content returns the existing state; same id with different content is a
  409; submitting against a non-admitted policy version is a 409. On timeout
  the response says `running` and points at the status endpoint — the client
  must query, never resubmit.
- `GET /runs/{run_id}/groups/{group_id}`, `POST .../cancel`
- `POST /runs/{run_id}/policy/activate` — validates the whole serving pool is
  healthy on the expected version, drains old admission, then opens the new
  version and publishes `policy-pool.json` into the shared run directory.
- `POST /runs/{run_id}/serving/announce` — the GPU job publishes its inference
  endpoint; the manager never hand-configures it.
- `POST /runs/{run_id}/drain`, `POST /runs/{run_id}/cancel`

Bulky data (trajectory manifests, token ids, logprobs, loss masks, group
export) is written to the shared run directory. HTTP responses carry paths
plus SHA-256 digests; `GroupResult` also carries per-slot identity, finish
reason, reward, trace seal status and verifier digest.

The GPU-side adapter (`harborrl/backends/slime_v032/rollout_service_client.py`)
recomputes every manifest digest, checks identities against the request,
requires a complete group of at least two slots on the locked policy version,
and only then converts trajectories into Slime samples. It is selected by
`HARBORRL_ROLLOUT_SERVICE_URL` in the GPU job environment.

## Service shutdown and recovery

The service holds an OS file lock on its shared run directory until shutdown
completes or its process exits. Registration persists the run manifest; a
different manifest for the same run is rejected even after a restart. Run and
group IDs must each be a single directory component.

Registration restores durable group outcomes. Repeating a completed request
returns its original result without regenerating trajectories or rewriting
immutable records. Failed and cancelled groups retain their outcomes. The GPU
client still validates all trajectory evidence before accepting a recovered
result. Completion is reported only after the result has been published.

A group without a durable terminal outcome becomes `unknown`. The service does
not adopt its previous Runner or replay partial work. It writes `unknown.json`
and `native-rollouts/cleanup-required.json`, blocks new groups for that run, and
returns the unknown outcome to polling clients. Inspect and confirm cleanup,
retain the original evidence, then use a fresh run for further sampling. This
recovers results and status; it does not resume an Agent or optimizer.

`RolloutService.close()` blocks admission, cancels groups and waits for their
asynchronous Runner cleanup, drains generation, then closes the Gateway and
executor loop. Its cleanup budget covers the Runner's 120-second cancellation
deadline and configured Sandbox cleanup deadlines. A generation drain alone
does not wait for tools, verification or environment deletion.

A cleanup timeout raises an error and leaves cleanup running. Incomplete
shutdown writes a cleanup-required marker; another close can wait for cleanup
without cancelling the same group again. The controller closes its HTTP listener
even when service shutdown fails. Both Docker and Sandbox runtimes reject new
attempts while observed cleanup-required markers remain. Inspect the resources
before removing resolved markers.

## Operational limits

Service mode requires shared storage visible at the configured paths on the
controller, GPU target and runners. A successful dry run checks configuration;
it does not establish network reachability, available capacity or training
quality. Check connectivity with doctor and inspect run artifacts after launch.

Choose worker and group concurrency for the target deployment. Instance
readiness does not imply that an agent is running, and a completed checkpoint
does not establish learning improvement.
