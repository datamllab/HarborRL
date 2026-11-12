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

- **2026-11-12** Added [gateway horizontal sharding](docs/deployment.md) with static routing, elastic auto-growth (remote-first), and automatic retirement.
- **2026-11-12** Added [Hub deployment mode](docs/deployment.md) for clusters with one-way network paths: nodes pull tasks over HTTP, no cross-machine SSH required.
- **2026-11-12** Added [online math training](docs/math-distill.md) (GRPO, DAPO, process rewards, OPD, PRM distillation) on the shared Slime/Megatron + SGLang pipeline.
- **2026-11-12** Added [configuration wizard](docs/configuration-wizard.md), [web console](docs/web-console.md), and [sandbox rollout](docs/sandbox-rollout.md) with audited cleanup.
- **2026-11-12** 10/2026 📣 HarborRL released

## About

HarborRL is a lightweight framework for training tool-using agents on verifiable, sandboxed tasks. It makes model serving, trajectory generation, tool execution, verifier rewards, and policy updates into a single reproducible pipeline.

HarborRL supports flexible combinations of **Models × Algorithms × Tasks × Harnesses (MATH)**:

- **Models:** RL training using Qwen and GLM as base models.
- **Algorithms:** GRPO and DAPO for RL training.
- **Tasks:** RL training on **200+** Harbor-format tasks in isolated workers, including Terminal-Bench, SWE-Bench, Deep-SWE, and more.
- **Harnesses:** RL training with **40+** harness frameworks, including Claude Code, Codex, LangGraph, SWE-agent, and more.

## AI-assisted Configure

For offline option browsing, configuration validation, and interactive or
answer-file setup, see the [configuration CLI guide](docs/configuration-wizard.md).

For mathematical data validation and scoring utilities, see the [math adapter guide](docs/math-distill.md).

For the offline options catalog, interactive configuration, and answer-file setup,
see the [configuration CLI guide](docs/configuration-wizard.md).
Start with `harborrl options` or `harborrl init --help`.

Run `harborrl web` to open the local browser console for configuration, validation,
dry-runs, training launch, and logs. See the [web console guide](docs/web-console.md)
for remote-controller access and lifecycle details.

For the optional math PRM/OPD reference trainer, see
[Math training pipeline](docs/math-distill.md). It includes explicit synthetic
fixtures for testing real optimizer updates and checkpoint restore.

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
doctor and launch_native --dry-run for a native deployment.
```

Confirm the result yourself against this checklist:

1. The agent reports one acceptance level: `cpu_development`,
   `deployment_preflight`, or `training`; it does not promote a CPU-only check
   or dry run to a higher level.
2. For CPU development, `uv run harborrl --help` and `uv run pytest` both pass.
3. For native deployment, every statically verifiable rollout-worker, GPU, and
   cross-machine check is `ready`; `harborrl doctor` has no failed checks; and
   `uv run harborrl train <Path to config.yaml> --dry-run` resolves the launch
   plan.
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

Expected result: uv executes the installed entry point and tests without activating the virtual environment manually. `train` accepts command composition or a YAML file; `doctor` validates a YAML deployment.

For native training, install the pinned backend stack in a writable location:

```bash
bash scripts/bootstrap_backends.sh <Path to HarborRL backends>
. <Path to HarborRL backends>/harborrl-backend.env
```

### Train

After installation, choose one of the two configuration styles:

```bash
# Compose the four experiment dimensions.
uv run harborrl train -model qwen3-8B -harness claude-code -task terminal-bench -alg grpo

# Load a complete, reproducible experiment definition.
uv run harborrl train <Path to config.yaml>
```

For a bootstrap-aware YAML launch, use:

```bash
bash scripts/launch_native.sh --config <Path to config.yaml> --backends <Path to HarborRL backends>
```

The command style is a shorthand over `config.yaml` in the working directory (or the file selected with `--config`) and selects the four experiment dimensions. The YAML style is the complete launch definition: it also pins checkpoints, workers, serving settings, GPU topology, sampling limits, and output locations. Start from [examples/native/train_qwen_native.yaml](examples/native/train_qwen_native.yaml) and keep site-specific files outside the repository.

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

The bundled runnable example is `hello_world`. Real training tasks are not repackaged by this repository: fetch Harbor-native datasets or tasks from Harbor Hub, and use the [compatibility summary](configs/harbor_hub/public_compatibility_summary.md) to understand validated coverage. After fetching a task root, validate it locally with `harborrl hub inspect` and lock it into your own catalog.

## License

HarborRL is released under the MIT license.

## Citation

If you use HarborRL for your research, please cite our technical report:
```

```

## Acknowledgement

We thank [Slime](https://github.com/THUDM/slime) and [Harbor](https://github.com/harbor-framework/harbor) for developing the foundational infrastructure for training and task execution.
