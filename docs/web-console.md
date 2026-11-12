# Local web console

```bash
harborrl web --workspace ./harborrl-web
```

This opens a browser on an available loopback port. Keep the terminal running.
`--port 8765` chooses a fixed port; `--no-browser` prints the URL without opening
it. The workspace stores saved configuration bundles and execution records.
One console process owns a workspace at a time. The web console uses the core
Python dependencies; Node, a frontend build and CDN access are not required.

## Configure and launch

1. Select Local, SSH workers, or Sandbox. Fill controller/GPU/runner paths and
   service origins. Source and interpreter paths refer to the selected machine.
   The controller source directory and Python executable select the checkout and
   environment used by the launched CLI.
2. Provide existing checkpoint/reference directories, a pinned task catalog,
   harness profile and real tokenizer/template fingerprints. The
   [configuration CLI guide](configuration-wizard.md) describes preparing these
   inputs. The console never downloads or generates model inputs.
3. Choose GPU layout, tensor parallelism, rollout batches and sampling settings.
   Allocation defaults are configuration values, not a model memory-fit estimate.
4. Review the generated YAML and validation. **Check local files** additionally
   checks references on the controller. Deferred GPU, connectivity, model and
   Sandbox checks remain explicit. Save the bundle under a new configuration name.
5. Run **Dry-run** to build the actual CLI launch plan. **Doctor** performs live
   checks after confirmation. **Start training** requires a separate confirmation
   and invokes the existing `harborrl train` flow, including its preflight.

Configuration names are limited to letters, digits, underscores and hyphens.
Saved configurations are not overwritten. Reopen one to launch it again, or edit
and save under a new name. The advanced answers editor and YAML/JSON import use
the existing `answers_version: 1` contract. Download an individual generated file
or export answers from the editor. Advanced remote Sandbox runners can be
specified through imported answers; the form preset defaults to a local runner.

H-cluster deployments remain the optional
[H practice](../practices/h_cluster/README.md). Prepare its Sandbox configuration
separately and reference the path. H project/environment identifiers and relay
settings are not part of the core web form.

Windows supports configuration and dry-run. Live training/doctor controls require
the console to run on a POSIX controller, using the same environment and checkout
that would normally run the training CLI. Deployment readiness still depends on
the project's supported Linux training environment.

## Runs and shutdown

Only one execution is active per workspace, including doctor and dry-run. This
prevents duplicate launches within this console; it is not a cluster-wide GPU
scheduler. Other CLI processes or different workspaces can still use the same
resources, and operators must coordinate those allocations.

Each job keeps immutable input files and a durable `job.json`/`console.log` under
`WORKSPACE/jobs/JOB_ID`. Execution appends `web/JOB_ID` to the configured shared
run root and updates both config files in its snapshot. The original saved bundle
is unchanged. The existing runtime then creates its normal `training/...` run
directory there. This isolates artifacts while retaining the configured shared
filesystem. Run logs show the actual snapshot and output paths.

The Runs view polls status and the last 64 KiB of output. **Controller** shows the
CLI output and errors. **GPU training** reads the separate `train-stdout.log` when
the service deployment creates one. Embedded training generally streams into
the controller log. Full logs remain on disk; the browser download contains the
currently displayed tail. This initial release does not aggregate WandB curves
or claim remote/GPU progress from a quiet controller log.

**Stop** sends an interrupt through the existing CLI lifecycle. Local training
forwards it to the training process group; service mode uses the existing SSH
target cancellation. `stopping` means cleanup is pending, not that resources are
already free. Closing the browser does not stop a job. Ctrl+C or SIGTERM in the
server terminal requests cleanup; shutdown waits up to 30 seconds. A timeout
records an unknown state without force-killing arbitrary remote processes.

After a crash or interrupted shutdown, previously running jobs appear as
`unknown`. New launches are blocked. Inspect the target processes, shared run
records and Sandbox cleanup evidence, then use **Confirm cleanup** to acknowledge
that external cleanup is complete. This action records operator confirmation; it
does not kill a stored PID or automatically relaunch a job. The server never
signals a PID recovered from an old session. Job details and **Confirm cleanup**
remain available if a log was never created or cannot be read. The log panel
reports that error explicitly; an unavailable log is not evidence of cleanup.

## Remote controller

On the controller, from the intended checkout/environment:

```bash
python -m harborrl.cli web --workspace /srv/shared/web-console --port 8765 --no-browser
```

On the workstation, create a local SSH forward:

```bash
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 your-controller-alias
```

Open the URL printed by the controller in the workstation browser. If the local
forward uses a different port, replace only the URL port. Leave the `#token=...`
fragment intact. If either port is occupied, choose an unused port; omitting
`--port` selects one automatically and prints it.

The listener is deliberately fixed to `127.0.0.1`. A random per-session token
protects API reads and writes; the browser moves the URL fragment into
sessionStorage and removes it from the address bar. State-changing requests also
require a same-origin JSON request. Responses use CSP, frame blocking and
no-store headers, and assets are packaged locally. Keep the token private: this
is a local single-user console with the controller user's filesystem and training
privileges. It is not a public multi-user service or an Internet-facing gateway.
Do not expose it through a public reverse proxy.

## Implementation and verification

- `harborrl/webui/server.py`: loopback HTTP endpoints, authentication, static
  resources and composition of existing catalog/builder/validators.
- `harborrl/webui/jobs.py`: serial CLI launch, snapshots, logs and lifecycle records.
- `harborrl/webui/cli.py`: CLI arguments, workspace lock and browser opening.
- `harborrl/webui/static/`: vanilla browser UI and vendored Lucide assets.
- `tests/harborrl/test_webui.py`: HTTP boundaries, offline configuration, real
  dry-run, stand-in training launch/cancellation, persistence and process groups.

Training schema 2 and deployment schema 1 are unchanged. No alternate training
engine, generic shell endpoint, secret editor or platform-specific preset is
introduced. Live training remains a real user action after reviewing the
configuration; tests use synthetic inputs and CPU-only process stand-ins.
