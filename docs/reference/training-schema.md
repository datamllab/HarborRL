# Training schema reference

Schema version 2 is strict. Except for the documented defaults below, every section
and field is required and unknown fields are rejected.

## Top level

| Field | Type | Constraint |
| --- | --- | --- |
| `schema_version` | integer | Exactly `2` |

## Consistency

The entire `consistency` section is optional and defaults to the values shown.

| Field | Type | Default / constraint |
| --- | --- | --- |
| `consistency.mode` | string | `off`; one of `off`, `audit`, `enforce` |
| `consistency.max_abs_delta` | number or null | `null`; nonnegative finite; required for `enforce` |
| `consistency.max_mean_abs_delta` | number or null | `null`; nonnegative finite |
| `consistency.bad_token_delta` | number or null | `null`; nonnegative finite |
| `consistency.max_bad_fraction` | number | `0.0`; in `[0, 1]` |
| `consistency.worst_tokens` | integer | `8`; in `[0, 32]` |

## Execution and inputs

| Field | Type | Constraint |
| --- | --- | --- |
| `execution.backend` | string | Exactly `harbor_job` through the registered task providers |
| `tasks.catalog` | path | Nonempty JSON/JSONL task lock; relative to config |
| `harness.name` | string | Currently `claude_code` |
| `harness.profile` | path | Exact harness profile JSON; relative to config |

## Harbor runner

| Field | Type | Default / constraint |
| --- | --- | --- |
| `harbor.python` | path | Runner interpreter path; relative to config in training semantics |
| `harbor.version` | string | Must be supported by harness; currently `0.23.0` |
| `harbor.workers` | array of strings | Nonempty for SSH; empty for local |
| `harbor.runner_transport` | string | Optional, default `ssh`; `ssh` or `local` |
| `harbor.environment` | object | Optional, default `{runtime: docker}` |

For Docker, `harbor.environment` contains only `runtime: docker`. For Sandbox it
contains exactly `runtime: sandbox` and `config: PATH`.

## Gateway

| Field | Type | Constraint |
| --- | --- | --- |
| `gateway.host` | string | Nonempty listener address |
| `gateway.port` | integer | `1..65535` |
| `gateway.advertised_url` | string | Plain HTTP(S) origin; port equals `gateway.port` |
| `gateway.tokenizer_digest` | string | 64-character lowercase SHA-256 |
| `gateway.template_digest` | string | 64-character lowercase SHA-256 |
| `gateway.audited_raw_logprobs` | boolean | Must be `true` for registered GRPO |

The advertised URL permits no credentials, query, fragment, or path other than `/`.

## Model

| Field | Type | Constraint |
| --- | --- | --- |
| `model.checkpoint` | path | Student checkpoint; relative to config |
| `model.reference` | path | Reference checkpoint; relative to config |
| `model.args_file` | string | Slime preset identifier; registered prefix `qwen` |

## Training

| Field | Type | Default / constraint |
| --- | --- | --- |
| `training.backend_contract` | string | Optional; default `slime-v0.3.2-native-v1` |
| `training.learning_rate` | number | Positive and finite |
| `training.num_rollout` | integer | Positive |
| `training.save_interval` | integer | Positive |

## Sampling

| Field | Type | Constraint |
| --- | --- | --- |
| `sampling.group_size` | integer | At least `2` |
| `sampling.groups_per_batch` | integer | Positive |
| `sampling.max_attempts` | integer | Positive invalid-attempt retry bound |
| `sampling.max_tokens` | integer | Positive and less than `max_context` |
| `sampling.max_context` | integer | Positive |

## GPU deployment

| Field | Type | Constraint |
| --- | --- | --- |
| `deployment.layout` | string | `split` or `colocate` |
| `deployment.num_gpus` | integer | Positive total budget |
| `deployment.actor_gpus` | integer | Positive; divisible by actor TP |
| `deployment.rollout_gpus` | integer | Positive; divisible by engine TP |
| `deployment.actor_tensor_parallel_size` | integer | Positive |
| `deployment.rollout_gpus_per_engine` | integer | Positive |

For `split`, actor plus rollout GPUs may not exceed the total. For `colocate`, actor,
rollout, and total counts must be equal.

## Output

| Field | Type | Constraint |
| --- | --- | --- |
| `output.root` | path | Run root; relative to config |

In service manager mode, the normalized output root must equal
`deployment.storage.run_root`.

Use `harborrl options fields --json` to obtain the authoritative machine-readable
catalog from the installed version.
