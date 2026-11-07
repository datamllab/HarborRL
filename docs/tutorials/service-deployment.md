# Tutorial: service deployment dry run

Service mode is site-specific: controller origins, SSH aliases, absolute paths, and
shared storage must describe real machines. This tutorial prepares and validates a
copy without starting services.

## 1. Copy the examples

```bash
mkdir -p ./site-example
cp examples/native/train_qwen_native.yaml ./site-example/config.yaml
cp examples/native/deployment-service.yaml ./site-example/deployment.yaml
cp examples/native/catalog.json ./site-example/catalog.json
cp examples/native/claude-code-profile.json ./site-example/claude-code-profile.json
cp -R examples/native/tasks ./site-example/tasks
```

## 2. Replace the placeholders

Edit both YAML files together. In service mode these pairs must match:

| Training field | Deployment field |
| --- | --- |
| `gateway.host` | `manager.bind` |
| `gateway.port` | `manager.gateway_port` |
| `gateway.advertised_url` | `manager.gateway_url` |
| `output.root` | `storage.run_root` |
| `harbor.workers` | Worker SSH destinations, in order |
| `harbor.python` | First worker's `runner_python` |

Also replace model paths and fingerprints, controller/GPU/worker checkout paths,
interpreters, task roots, run-token path, SSH aliases, and reachable origins. All
deployment paths must be absolute.

## 3. Validate offline

```bash
uv run harborrl validate ./site-example/config.yaml \
  --deployment ./site-example/deployment.yaml \
  --json
```

Add `--check-files` after the controller-local paths exist. Remote GPU and worker
paths remain deferred.

## 4. Resolve the service plan

```bash
uv run harborrl train ./site-example/config.yaml \
  --deployment ./site-example/deployment.yaml \
  --dry-run > /tmp/harborrl-service-plan.json
```

Verify that the plan reports `mode: service`, the intended SSH GPU target, worker
IDs, control/gateway URLs, and shared run root.

## 5. Stop before allocation

The next command is `harborrl doctor`, which performs live SSH, filesystem, port, and
connectivity checks. Run it only from the intended controller environment. Removing
`--dry-run` from `train` can launch the GPU job and allocate Docker/Sandbox resources;
obtain operator approval first.
