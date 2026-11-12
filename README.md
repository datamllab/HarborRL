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

- **2026-11-12** Added elastic gateway scaling with static routing, remote-first auto-growth and automatic retirement.
- **2026-11-12** Added Hub deployment mode for one-way-firewall clusters: nodes pull tasks over HTTP, no cross-machine SSH required.
- **2026-11-12** Unified multiple training algorithms on the shared Slime pipeline, validated on mathematical reasoning tasks.
- **2026-11-12** Improved the configuration wizard, web console and sandbox rollout with audited cleanup.
- **2026-10-26** Added an [offline configuration wizard](docs/configuration-wizard.md) and a [local Web console](docs/web-console.md) for configuration, validation, dry-runs, training launch, and logs.
- **2026-10-26** Added [controller/GPU deployment configuration](docs/deployment.md), optional [OpenSandbox execution](docs/sandbox-rollout.md), and opt-in [train/inference consistency checks](docs/train-inference-consistency.md).
- **10/2026** 📣📣📣 HarborRL released

## About

HarborRL is a lightweight framework for training tool-using agents on verifiable, sandboxed tasks. It makes model serving, trajectory generation, tool execution, verifier rewards, and policy updates into a single reproducible pipeline.

HarborRL supports flexible combinations of **Models × Algorithms × Tasks × Harnesses (MATH)**:

- **Models:** RL training using Qwen and GLM as base models.
- **Algorithms:** GRPO and DAPO for RL training.
- **Tasks:** RL training on **200+** Harbor-format tasks in isolated workers, including Terminal-Bench, SWE-Bench, Deep-SWE, and more.
- **Harnesses:** RL training with **40+** harness frameworks, including Claude Code, Codex, LangGraph, SWE-agent, and more.

## AI-assisted Configure

Use `harborrl options` to browse capabilities offline, `harborrl init` for interactive or answer-file setup, and `harborrl web` for the local console. See the [configuration guide](docs/configuration-wizard.md) for the full workflow including AI-agent deployment.


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

Training uses one YAML file with a fixed schema; see [task configuration](docs/task-configuration.md) for the full field reference, task directory layout, and catalog locking.


## License

HarborRL is released under the MIT license.

## Citation

If you use HarborRL for your research, please cite our technical report:
```

```

## Acknowledgement

We thank [Slime](https://github.com/THUDM/slime) and [Harbor](https://github.com/harbor-framework/harbor) for developing the foundational infrastructure for training and task execution.
