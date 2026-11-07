<p align="center">
  <img src="assets/HarborRL-title.jpg" alt="HarborRL" width=800>
</p>

<p align="center">
  <h1>HarborRL</h1>
</p>

<p align="center">
  <strong>A lightweight agentic reinforcement learning framework for everyone</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg" alt="PyTorch 2.4+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
</p>

<p align="center">
  English · <a href="README_zh.md">简体中文</a>
</p>

---

## News

- **2026-10-26** Added an [offline configuration wizard](docs/configuration-wizard.md) and a [local Web console](docs/web-console.md) for configuration, validation, dry-runs, training launch, and logs.
- **2026-10-26** Added [controller/GPU deployment configuration](docs/deployment.md), optional [OpenSandbox execution](docs/sandbox-rollout.md), and opt-in [train/inference consistency checks](docs/train-inference-consistency.md).
- **10/2026** 📣📣📣 HarborRL released

## About

HarborRL is a lightweight framework for training tool-using agents on verifiable, sandboxed tasks. It makes model serving, trajectory generation, tool execution, verifier rewards, and policy updates into a single reproducible pipeline.

HarborRL validates explicit combinations of **Models × Algorithms × Tasks × Harnesses (MATH)**.
The current implementation registers:

- **Model:** Qwen presets served by SGLang.
- **Algorithm:** GRPO through the pinned Slime v0.3.2 native contract.
- **Tasks:** native Harbor task locks in JSON or JSONL catalogs.
- **Harness:** Claude Code through Harbor 0.23.0.

Run `harborrl list models`, `algorithms`, `tasks`, or `harnesses` to query the
installed registry. Dataset compatibility evidence is scoped separately and does
not imply that a task collection ships with HarborRL or has completed RL training.
See the [documentation site](https://datamllab.github.io/HarborRL/) for the exact
support matrix, tutorials, deployment guides, and API reference.

## AI-assisted Configure

For the offline options catalog, interactive configuration, and answer-file setup,
see the [configuration CLI guide](docs/configuration-wizard.md).
Start with `harborrl hello` for a CPU walkthrough requiring no model, GPU or task
JSON. It executes scripted attempts and a real verifier, then exports a GRPO batch
with fixture tokens. Continue with `harborrl options` or `harborrl init --help`.
The wizard detects one shared Local environment, calculates model fingerprints,
and supports incomplete drafts (`:save` / `--resume`).

Run `harborrl web` to open the local browser console for configuration, validation,
drafts, filled Local/Sandbox examples, Hello World, dry-runs, training launch, and logs. See the [web console guide](docs/web-console.md)
for remote-controller access and lifecycle details.

If you want an AI agent to help deploy this repository and verify training,
ask it to first read and strictly follow
[docs/agents/environment-check.md](docs/agents/environment-check.md).

The practiced workflow is to give the agent this prompt, replacing the bracketed
values with your topology:

```text
Read docs/agents/environment-check.md and follow it exactly.
Target: [CPU development only | native split CPU-rollout/GPU-training deployment]
Config: <Path to config.yaml>, or none
Topology: controller=<Your IP address>:<Port Number>;
  rollout workers=<Rollout worker IP address>:<Port Number>;
  GPU machine=<GPU machine IP address>:<Port Number>

First run read-only checks only. Return the required structured report and tell
me the smallest next command that fixes one blocker. Do not install, download,
pull an image, mount storage, or start a service without my explicit consent.
After each approved change, rerun the relevant checks. Finish with harborrl
doctor and harborrl train --dry-run for a native deployment, passing the explicit
deployment file when one is used.
```

Confirm the result yourself against this checklist:

1. The agent reports one acceptance level: `cpu_development`,
   `deployment_preflight`, or `training`; it does not promote a CPU-only check
   or dry run to a higher level.
2. For CPU development, `uv run harborrl --help` and `uv run pytest` both pass.
3. For native deployment, every statically verifiable rollout-worker, GPU, and
   cross-machine check is `ready`; `harborrl doctor` has no failed checks; and
   `uv run harborrl train <Path to config.yaml> --deployment <Path to deployment.yaml>
   --dry-run` resolves the launch plan (omit `--deployment` only for an intentional
   embedded/default deployment).
4. Every `unknown` item names what is deferred and why. Live gateway and
   task-image checks can legitimately remain deferred before training.
5. The agent waits for explicit consent before starting real training.

## Quick Start

The default CPU install does not need a Sandbox SDK or cluster account.
For controller placement, see [deployment](docs/deployment.md). For optional
standard OpenSandbox execution, see [Sandbox rollout](docs/sandbox-rollout.md).
The independently installed [H-cluster practice](practices/h_cluster/README.md)
contains Brainbox-specific deployment and restricted-network relay examples.

### Install

Use Python 3.10 or newer. Runtime dependencies, development dependencies, and the `harborrl` command are locked by [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). The CPU development install does not require CUDA, Docker, model weights, Slime, Megatron-LM, or SGLang.

Install [uv](https://docs.astral.sh/uv/) 0.9 or newer, then synchronize the locked CPU development environment:

```bash
uv sync --extra dev
```

Expected result: uv reads `uv.lock`, creates `.venv/`, and installs HarborRL with `pytest` and `ruff`. The command does not install CUDA kernels, Docker, model weights, Slime, Megatron-LM, or SGLang.

Run commands through the locked environment:

```bash
uv run harborrl --help
uv run pytest
```

Expected result: uv executes the installed entry point and tests without activating
the virtual environment manually. `train` loads a YAML file and can overlay explicit
MATH selectors; `doctor` validates the configured environment and deployment.

For native training, install the pinned backend stack in a writable location:

```bash
bash scripts/bootstrap_backends.sh "<Path to HarborRL backends>"
. "<Path to HarborRL backends>/harborrl-backend.env"
```

The generated file exports `SLIME_DIR`, `MEGATRON_DIR`, and `SGLANG_IMAGE` to child
processes. For files generated by older bootstrap scripts, run `set -a` before
sourcing and `set +a` afterward. See [rollout lifecycle and recovery](docs/rollout-runtime-lifecycle.md)
for cleanup markers and CPU-only regression checks.

### Train

Start with the bundled side-effect-free dry run:

```bash
uv run harborrl train examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml --dry-run

# Overlay registered selectors on an existing configuration and locked task ID.
uv run harborrl train --config examples/native/train_qwen_native.yaml \
  --deployment examples/native/deployment-embedded.yaml \
  -model qwen3-8B -harness claude-code -task hello_world -alg grpo --dry-run
```

For a prepared site configuration, pass both files explicitly:

```bash
uv run harborrl train <Path to config.yaml> \
  --deployment <Path to deployment.yaml>
```

Dimension flags overlay `config.yaml` in the working directory (or the file selected
with `--config`); they do not construct a configuration from scratch. The YAML file
pins checkpoints, workers, serving settings, GPU topology, sampling limits, and output
locations. Start from [examples/native/train_qwen_native.yaml](examples/native/train_qwen_native.yaml)
and keep site-specific files outside the repository. `scripts/launch_native.sh` is a
bootstrap convenience for embedded/default discovery; it has no explicit
`--deployment` option.

## Task Configuration

Training is configured with one YAML file that has a fixed section and field set. Relative paths resolve against the YAML file, and repeated `--set section.field=value` overrides are parsed as YAML values and validated again. Start site-specific configurations from [examples/native/train_qwen_native.yaml](examples/native/train_qwen_native.yaml). Optional [train/inference consistency checks](docs/train-inference-consistency.md) compare frozen rollout scores with the pre-update trainer and probe synchronized inference engines.

The configuration has three broad parts: `execution` is fixed to `harbor_job`; `harbor`, `gateway`, and `deployment` describe rollout workers, service endpoints, and GPU topology; `tasks`, `harness`, `model`, `training`, `sampling`, and `output` describe the experiment and its outputs.

HarborRL consumes Harbor-recognized native tasks directly and does not maintain private conversion formats for individual benchmarks. A typical task directory is:

```text
task/
├── task.toml
├── instruction.md
├── environment/Dockerfile
└── tests/test.sh
```

Here, `task.toml` declares resources, the agent, and the verifier; `instruction.md` is the agent-facing instruction; `environment/Dockerfile` builds the isolated environment; and `tests/test.sh` verifies the result and writes `reward.json` or `reward.txt`. For training, tasks are locked into a catalog with their task ID, upstream revision, local path, content digest, and reward parsing rules; see [examples/native/catalog.json](examples/native/catalog.json).

The bundled runnable example is `hello_world`. It allows internet access so Harbor can install the pinned Claude Code CLI, reserves 4 GiB each for memory and storage, and writes a 0/1 reward to `/logs/verifier/reward.json`. Its profile allows four turns so the agent can observe a tool result and finish. Offline deployments need the exact CLI version preinstalled and a reachable policy gateway. After changing a task, regenerate its catalog digest.

The example pins the runtime-validated Claude Code `2.1.141`. The adapter disables prompt caching and cache warmup, compaction, and nonessential traffic, and always disallows `Agent` and `Task` even with a custom tool denylist. CLI `2.1.0` still makes an independent file-path-extraction model request after Bash and is incompatible with the single-context rollout contract. Probe other CLI versions before use; native session audit rejects auxiliary contexts, summaries, and unconsumed model responses.

Real training tasks are not repackaged by this repository: fetch Harbor-native datasets or tasks from Harbor Hub, and use the [compatibility summary](configs/harbor_hub/public_compatibility_summary.md) to understand validated coverage. After fetching a task root, validate it locally with `harborrl hub inspect` and lock it into your own catalog.

## License

HarborRL is released under the MIT license.

## Citation

If you use HarborRL for your research, please cite our technical report:
```

```

## Acknowledgement

We thank [Slime](https://github.com/THUDM/slime) and [Harbor](https://github.com/harbor-framework/harbor) for developing the foundational infrastructure for training and task execution.
