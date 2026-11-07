# Deployment schema reference

Deployment schema version 1 describes **where** roles run and **how** they connect.
It contains no learning rate, sampling, task, or model semantics. Unknown fields and
inline secrets are rejected.

All role paths are absolute paths interpreted on the named target machine.

## Top level

```yaml
schema_version: 1
manager: {}
gpu_target: {}
storage: {}
rollout: {}
connectivity: {}
```

## `manager`

| Field | Type | Constraint |
| --- | --- | --- |
| `id` | string | Short identifier |
| `mode` | string | `embedded` or `service` |
| `bind` | string | Nonempty listener address |
| `control_port` | integer | `1..65535`; differs from gateway port |
| `control_url` | origin | Port matches; non-loopback in service mode |
| `gateway_port` | integer | `1..65535`; differs from control port |
| `gateway_url` | origin | Port matches; non-loopback in service mode |
| `run_token_file` | absolute path | File reference, never token contents |
| `source_dir` | absolute path | Controller checkout |
| `controller_python` | absolute path | Controller interpreter |

## `gpu_target`

| Field | Type | Constraint |
| --- | --- | --- |
| `id` | string | Short identifier |
| `transport` | string | `local` or `ssh` |
| `ssh` | object or null | Required for SSH; null for local |
| `source_dir` | absolute path | GPU checkout |
| `train_python` | absolute path | GPU training interpreter |
| `environment_file` | absolute path or null | Controlled JSON variable map |
| `expected_gpus` | integer | Positive target inventory |

## `storage`

| Field | Type | Constraint |
| --- | --- | --- |
| `mode` | string | Only `shared_fs` |
| `run_root` | absolute path | Same shared run root for all roles |
| `manager_writable` | boolean | Declared manager access |
| `gpu_writable` | boolean | Declared GPU access |
| `worker_writable` | boolean | Declared worker access |

These are topology declarations, not permission grants. Current preflight probes the
manager's run-root writability; GPU and worker access must be verified on those
machines and remains an operator acceptance check.

## `rollout`

| Field | Type | Constraint |
| --- | --- | --- |
| `provider` | string | `embedded`, `ssh_docker`, or `sandbox` |
| `workers` | array | Required/nonempty for `ssh_docker`; may be empty otherwise |

Each worker contains exactly:

| Field | Type | Constraint |
| --- | --- | --- |
| `id` | string | Unique short identifier |
| `ssh` | object | SSH destination |
| `source_dir` | absolute path | Worker checkout |
| `runner_python` | absolute path | Python 3.12+ runner interpreter |
| `docker` | boolean | Declared Docker requirement |
| `task_roots` | array of absolute paths | At least one root |

## SSH objects

`gpu_target.ssh` and worker `ssh` objects accept only:

| Field | Type | Constraint |
| --- | --- | --- |
| `host` | string | Required destination/alias; not a flag |
| `user` | string | Optional username |
| `port` | integer | Optional positive port |
| `identity` | absolute path | Optional identity-file reference |

Passwords, tokens, and private-key contents are not schema fields.

## `connectivity`

| Field | Type | Constraint |
| --- | --- | --- |
| `command_timeout_s` | integer | Positive |
| `ssh_timeout_s` | integer | Positive |
| `http_timeout_s` | integer | Positive |
| `gpu_to_controller` | boolean | Required route declaration |
| `worker_to_gateway` | boolean | Required route declaration |
| `manager_to_gpu_inference` | boolean | Required route declaration |

Use `harborrl options fields --json` for machine-readable field metadata and
`harborrl validate CONFIG --deployment DEPLOYMENT` for cross-file checks.
