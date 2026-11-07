# Embedded and local deployment

Embedded mode keeps controller and training lifecycle in one process placement while
allowing Harbor runners to be local or SSH workers. It is the compatibility default
when no deployment file is supplied.

## Resolution

For this command:

```bash
harborrl train site/config.yaml --dry-run
```

HarborRL resolves deployment in this order:

1. an explicit `--deployment PATH`;
2. `deployment.yaml` in the current directory; or
3. a generated `embedded-local` deployment derived from the training config.

The generated deployment uses the current checkout and interpreter, the configured
output root, training-config worker destinations, and the configured GPU count. Its
source is recorded as `embedded-local` in the plan.

## Explicit embedded file

Use an explicit schema-1 file when you want the topology recorded independently:

```bash
harborrl train site/config.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  --dry-run
```

The repository example contains absolute container-style placeholders. Replace them
before doctor or a live run.

## Local runner configuration

To run the Harbor runner on the controller:

```yaml
harbor:
  python: /absolute/path/to/python3.12
  version: "0.23.0"
  runner_transport: local
  workers: []
  environment:
    runtime: docker
```

The gateway's `advertised_url` must still be reachable from task containers. On
Linux, configure a host-gateway mapping before using `host.docker.internal`, or use a
reachable host address.

## Limits

- Embedded placement does not remove the need for GPUs, model files, backend
  dependencies, Docker workers, or network routes.
- An implicit deployment does not activate the full explicit topology preflight
  layer; use a deliberate deployment file for cross-machine acceptance.
- Each run uses a new directory. HarborRL does not resume by reusing an arbitrary
  completed run directory.
