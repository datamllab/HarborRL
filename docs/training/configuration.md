# Training configuration

A training file uses strict schema version 2. Unknown top-level sections, unknown
fields, malformed types, and incompatible cross-field values fail before launch.

```yaml
schema_version: 2

execution: {backend: harbor_job}
tasks: {catalog: catalog.json}
harness: {name: claude_code, profile: claude-code-profile.json}

harbor:
  python: /opt/harbor/bin/python
  version: "0.23.0"
  workers: [worker-1, worker-2]

gateway:
  host: 0.0.0.0
  port: 49183
  advertised_url: http://gateway-worker:49183
  tokenizer_digest: "0000000000000000000000000000000000000000000000000000000000000000"
  template_digest: "1111111111111111111111111111111111111111111111111111111111111111"
  audited_raw_logprobs: true

model:
  args_file: qwen3-8B
  checkpoint: /models/qwen3-8b
  reference: /models/qwen3-8b-reference

training:
  backend_contract: slime-v0.3.2-native-v1
  learning_rate: 0.000001
  num_rollout: 1
  save_interval: 1

sampling:
  group_size: 2
  groups_per_batch: 1
  max_attempts: 2
  max_context: 8192
  max_tokens: 1024

deployment:
  layout: split
  num_gpus: 8
  actor_gpus: 4
  rollout_gpus: 4
  actor_tensor_parallel_size: 4
  rollout_gpus_per_engine: 4

output: {root: runs}
```

The two all-zero/all-one digests above are illustrative placeholders. Generate real
fingerprints with the configuration wizard before a live run.

## Resolution rules

- Task catalog, harness profile, runner Python, model checkpoint/reference, output
  root, and Sandbox config paths are normalized relative to the training YAML.
- `consistency` may be omitted; its mode defaults to `off`.
- `training.backend_contract` may be omitted; it defaults to
  `slime-v0.3.2-native-v1`.
- `harbor.runner_transport` defaults to `ssh`.
- `harbor.environment` defaults to `{runtime: docker}`.
- The local runner transport requires `harbor.workers: []`; SSH requires at least
  one destination.

See the [complete training schema](../reference/training-schema.md) for every field.

## Overrides

`--set` is repeatable. Values are parsed as YAML and the whole configuration is
validated again:

```bash
harborrl train site/config.yaml --dry-run \
  --set training.num_rollout=2 \
  --set sampling.groups_per_batch=4 \
  --set consistency.mode=audit
```

Only known `section.field` paths are accepted. An override cannot add a new schema
field.

The dimension flags are selectors over an existing file, not a standalone config
builder:

```bash
harborrl train --config site/config.yaml \
  -model qwen3-8B -harness claude-code -task hello_world -alg grpo \
  --dry-run
```

When any selector is used without `--config` or a positional path, HarborRL reads
`./config.yaml`. `-task` filters the locked catalog to the named task ID.

## Validation sequence

```bash
harborrl validate site/config.yaml \
  --deployment site/deployment.yaml --check-files

harborrl train site/config.yaml \
  --deployment site/deployment.yaml --dry-run

harborrl doctor site/config.yaml \
  --deployment site/deployment.yaml
```

Offline validation can return `valid_with_deferred_checks` with exit code 0. That is
not equivalent to doctor readiness.
