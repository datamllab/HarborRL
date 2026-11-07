# Configuration CLI

The configuration tools run with the core package (Python 3.10+ and PyYAML).
They enumerate implemented capabilities, check configuration offline, and create
consistent training schema 2 / deployment schema 1 files. Catalog, preview and
validation stay offline. Model preparation explicitly runs the selected Python
locally or over SSH to fingerprint an existing tokenizer. It does not download
models or allocate Sandbox/GPU jobs.

## First run without a task JSON or GPU

Install the core package and try the bundled CPU walkthrough:

```bash
python -m pip install -e .
harborrl hello --out-dir ./hello-first
harborrl web --workspace ./harborrl-web
```

Hello World executes two scripted attempts at creating a file, runs the actual
task verifier, collects rewards 0 and 1, assembles native IR and exports a GRPO
batch. It uses fixture token evidence, marked `CONTRACT_READY`; it does not perform
model inference or update GPU weights. All files stay under the new output
directory. The generated `catalog.json` and `claude-code-profile.json` can be reused
when configuring real training. The command also works from an installed wheel
outside a source checkout, on Windows or Linux.

## Browse and validate

```bash
harborrl options
harborrl options deployments --json
harborrl options fields --json
harborrl validate site/config.yaml --deployment site/deployment.yaml --json
harborrl validate site/config.yaml --deployment site/deployment.yaml --check-files
```

Catalog dimensions are `models`, `algorithms`, `tasks` (task providers),
`harnesses`, `datasets`, `deployments`, `fields`, and `compat`. JSON keeps the
`catalog_version` and `schema_versions` envelope when selecting a dimension.
The catalog reflects registered implementations, not roadmap claims. Dataset
`status` describes support; `available` describes local task files. A wheel does
not include repository examples, and planned datasets are not ready to train.

Validation defaults to schema, semantics, and cross-file consistency. It does
not open referenced profiles or tasks until `--check-files` is supplied. Run file
checks on the controller: its source/interpreter, profile, task catalog and
Sandbox configuration belong there. Local GPU and runner paths are checked
locally; SSH target paths are deferred. The shared root may be created by a later
run and is not required to exist during configuration. Credentials are never read.

Checks are `passed`, `failed`, or `deferred`. The report is `invalid` if a check
fails, otherwise `valid_with_deferred_checks` while live checks remain. Exit codes
are 0 for no failed checks, 1 for validation/creation failure, and 2 for CLI usage
errors. `--json` writes one JSON object to stdout. Schema validators may report
only their first error. Without `--deployment`, the existing CLI discovery of
`./deployment.yaml` and embedded fallback is preserved and reported.

GPU memory fit, model loading, live tokenizer fingerprints, package versions,
connectivity and platform capacity are not established by offline validation.
Sandbox image bindings are checked locally; trusted site adapters and credentials
are checked later by runner preflight. Sandbox installation/profile timing
compatibility is additionally checked by the existing training dry-run.

## Prepare inputs

1. Obtain Harbor-format tasks and run `harborrl hub inspect --tasks-dir /path/tasks
   --dataset my-tasks --revision pinned-revision --run-dir /path/inspection`.
   The resulting JSONL is an inspection report, not a training catalog. Build a
   JSON list (or JSONL records with a `.jsonl` suffix) containing exactly `id`,
   `revision`, `path`, `task_digest`, and `reward_profile` for each selected task.
   Copy `path` and `task_digest` from a `NEEDS_PROBE` inspection record, assign a
   unique stable `id`, retain the pinned revision, and explicitly set the verifier's
   reward contract (for example `reward_profile: {key: reward}`). See
   [the catalog example](https://github.com/datamllab/HarborRL/blob/main/examples/native/catalog.jsonl). `INVALID` and
   `UNSUPPORTED` inspection results cannot be made trainable by changing labels.
   Do not replace computed digests with placeholder values.
2. Prepare a pinned Claude Code profile. See
   [the native examples](https://github.com/datamllab/HarborRL/tree/main/examples/native) for profile structure. It must match
   the agent and task requirements of your deployment.
3. Stage checkpoint and reference directories on the GPU target. The wizard
   calculates both tokenizer fingerprints automatically, using that target's
   Python with Transformers installed. No manual SHA-256 entry is needed. In Web,
   choose **Prepare model** after setting the checkpoint and environment. The
   operation loads tokenizer files offline with `trust_remote_code=False` and calls
   the same fingerprint function as the rollout runtime. A missing dependency or
   checkpoint produces an actionable error; the draft can still be saved.
4. Identify controller/GPU/runner Python executables, checkout paths, SSH aliases
   and shared storage. Service origins must include their declared ports and be
   reachable by the participating machines.
5. For Sandbox, prepare the existing platform configuration and image bindings.
   H-cluster users follow [the optional H practice](deployment/h-cluster.md)
   and reference its site file. The core CLI never asks for H project/environment
   identifiers or embeds private endpoints or credentials.

## Generate

```bash
harborrl init --out-dir ./site-new
harborrl init --answers my-answers.yaml --out-dir ./site-new
```

Interactive mode explains each required field. Local asks for one shared source
directory and Python interpreter, detected from the host running the wizard;
Controller, GPU and rollout all use those values. Storage is configured
separately. Service modes ask for each remote role and its reachable origins.
The wizard prepares model fingerprints automatically, previews both
configurations and requires `yes` to publish.

Progress is saved after every answer to `OUT_DIR.draft.json`. Type `?` for help,
`:skip` to leave a field empty, or `:save` to save and exit. Ctrl+C or EOF after
choosing a deployment also keeps the draft. Resume it later:

```bash
harborrl init --out-dir ./site-new --resume ./site-new.draft.json
harborrl init --out-dir ./empty-site --save-draft
harborrl init --out-dir ./local-example --example local
harborrl init --out-dir ./sandbox-example --example sandbox
```

`--example` saves a filled reference draft using the current host's detected
environment. Replace example paths and hosts before resuming. `--draft-file`
chooses a different progress file. Incomplete drafts never become runnable
bundles. Draft updates check revisions and reject stale writers.

Non-terminal generation accepts `--answers` or `--resume`. Missing fingerprints
trigger explicit tokenizer preparation; `--prepare-model` recalculates existing
values. Missing required answers never fall back to prompting. Use `--save-draft`
to store incomplete answers without running preparation or final validation.

Start from one of these reference answer templates and replace example paths
before running:

- [embedded-local](https://github.com/datamllab/HarborRL/blob/main/examples/options/embedded-local.answers.yaml): one machine,
  local Harbor runner, colocated training/serving by default.
- [service-ssh](https://github.com/datamllab/HarborRL/blob/main/examples/options/service-ssh.answers.yaml): controller, SSH GPU
  target and explicit SSH Docker workers.
- [service-sandbox-local-runner](https://github.com/datamllab/HarborRL/blob/main/examples/options/service-sandbox.answers.yaml):
  controller and runner here, SSH GPU target, Sandbox task execution.

Templates leave fingerprints empty for automatic preparation. Web also offers
filled Local and Sandbox recipes with notes, download and apply actions.
GPU allocations illustrate topology; confirm memory requirements for your model
before training. For Local Docker, the gateway advertised origin must be
reachable inside task containers. On Linux, configure the host-gateway mapping
if using `host.docker.internal`, or choose a reachable host address.

Answers allow only `answers_version: 1`, `preset`, `training`, `deployment`, and
optional `sandbox_config` and `local_environment`. Local's `local_environment`
contains `source_dir` and `python` and applies to all three roles, taking precedence
over advanced per-role overrides. Overrides use the existing schema field names;
unknown keys fail. Answer values are not evaluated as shell expressions. Gateway fields,
shared run root, GPU budget and worker destinations are derived when only one
side is provided. Contradictory explicit values fail. Supply matching origin URLs
when changing ports. Paths in training overrides are relative to the **final
output directory**, not the answers file. Deployment machine paths are absolute.

The Sandbox preset can also describe remote runners through an answers file:
set `training.harbor.runner_transport: ssh` and provide deployment rollout workers
with `docker: false`. The builder derives training worker destinations and the
first worker's runner Python. All configured interpreters still require preflight.

`--probe-local` explicitly enables bounded local `nvidia-smi` UUID and current
Python version queries. It also reads explicit SSH Host aliases (no wildcard,
Match execution or Include expansion) and gives a parent-directory access hint.
Missing tools and failed commands return `unknown`, not zero GPUs. Suggestions
never silently change answers; no remote probe or write permission test occurs.

The output directory must not exist, and its parent must exist. Publication uses
an exclusive per-target lock, same-parent staging and a directory rename. Existing
files are never overwritten. Failures clean up only this invocation's staging and
lock. After an abrupt process termination, inspect any remaining `.NAME.init.lock`
and staging directory before removing them; automatic stale-lock recovery is not
attempted. The lock coordinates wizard processes, not arbitrary external writers.

The bundle contains `config.yaml`, `deployment.yaml`, and `setup-manifest.json`.
The manifest records schema versions, configuration digests, references and the
offline validation scope, including deferred checks. It does not copy credentials.

The CLI prints these next commands without executing them:

```bash
harborrl validate site-new/config.yaml --deployment site-new/deployment.yaml --check-files
harborrl train site-new/config.yaml --deployment site-new/deployment.yaml --dry-run
harborrl doctor site-new/config.yaml --deployment site-new/deployment.yaml
```

`doctor` performs live environment/connectivity checks. Actual training remains a
separate action. Command examples use POSIX shell quoting.
