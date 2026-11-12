<p align="center">
  <img src="assets/HarborRL-title.jpg" alt="HarborRL" width=800>
</p>

<p align="center">
  <h1>HarborRL</h1>
</p>

<p align="center">
  <strong>面向所有人的轻量级智能体强化学习框架</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg" alt="PyTorch 2.4+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文
</p>

---

## News

- **2026-11-12** 新增网关弹性扩展：静态路由、弹性自动扩容（多机优先）与自动缩容。
- **2026-11-12** 新增 Hub 部署模式：适配单向网络集群，节点经 HTTP 拉取任务，无需跨机 SSH。
- **2026-11-12** 统一接入多种算法，使用数学任务训练验证。
- **2026-11-12** 优化配置向导、Web 控制台与 Sandbox 执行的审计清理。
- **2026-10-26** 新增[离线配置向导](docs/configuration-wizard.md)和[本地 Web 控制台](docs/web-console.md)，支持配置、校验、dry-run、训练启动和日志查看。
- **2026-10-26** 新增 [controller/GPU 部署配置](docs/deployment.md)、可选的 [OpenSandbox 执行能力](docs/sandbox-rollout.md)，以及按需启用的[训练与推理一致性检查](docs/train-inference-consistency.md)。
- **10/2026** 📣📣📣 HarborRL released

## 项目介绍

HarborRL 是一个轻量级框架，用于在可验证、可沙箱隔离的任务上训练会使用工具的智能体。它把模型服务、轨迹生成、工具执行、验证器奖励和策略更新组成一条可复现流水线。

HarborRL 支持 **Models × Algorithms × Tasks × Harnesses (MATH)** 的灵活组合：

- **Models：**使用 Qwen 和 GLM 作为基础模型进行 RL 训练。
- **Algorithms：**支持 GRPO 和 DAPO 等 RL 训练算法。
- **Tasks：**在隔离 worker 中对 **200+** 个 Harbor 格式任务进行 RL 训练，包括 Terminal-Bench、SWE-Bench、Deep-SWE 等。
- **Harnesses：**支持 **40+** 个 harness 框架进行 RL 训练，包括 Claude Code、Codex、LangGraph、SWE-agent 等。

## AI 辅助配置

使用 `harborrl options` 离线浏览能力，`harborrl init` 交互式或答案文件配置，`harborrl web` 打开本地控制台。完整工作流（含 AI 代理部署）见[配置指南](docs/configuration-wizard.md)。


## 快速开始

默认 CPU 安装不需要 Sandbox SDK 或集群账号。
controller 与 GPU 的部署方式见[部署指南](docs/deployment.md)，可选的标准 OpenSandbox 执行方式见[Sandbox rollout 指南](docs/sandbox-rollout.md)。
独立安装的 [H-cluster 实践包](practices/h_cluster/README.md)提供 Brainbox 部署与受限网络中继示例。

### 安装

使用 Python 3.10 或更新版本。Python 依赖、开发依赖和 `harborrl` 命令由 [pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock) 锁定。CPU 开发环境不需要 CUDA、Docker、模型权重、Slime、Megatron-LM 或 SGLang。

安装 [uv](https://docs.astral.sh/uv/) 0.9 或更新版本，然后同步锁定的 CPU 开发环境：

```bash
uv sync --extra dev
```

预期结果：uv 读取 `uv.lock`，创建 `.venv/`，并安装 HarborRL、`pytest` 和 `ruff`。该命令不会安装 CUDA kernel、Docker、模型权重、Slime、Megatron-LM 或 SGLang。

通过锁定环境运行命令：

```bash
uv run harborrl --help
uv run pytest
```

预期结果：uv 直接执行已安装的入口和测试，不需要手动激活虚拟环境。`train` 支持命令组合或 YAML 文件；`doctor` 用于校验 YAML 部署。

如需原生训练，请把固定版本的 native 训练栈安装到可写位置：

```bash
bash scripts/bootstrap_backends.sh <Path to HarborRL backends>
. <Path to HarborRL backends>/harborrl-backend.env
```

### 训练

完成安装后，从两种配置方式中选择一种：

```bash
# 直接组合四个实验维度。
uv run harborrl train -model qwen3-8B -harness claude-code -task terminal-bench -alg grpo

# 加载完整且可复现的实验定义。
uv run harborrl train <Path to config.yaml>
```

如需让 YAML 启动过程自动感知并引导后端，请使用：

```bash
bash scripts/launch_native.sh --config <Path to config.yaml> --backends <Path to HarborRL backends>
```

命令方式是基于当前目录 `config.yaml`（或通过 `--config` 选择的文件）的简写形式，用于选择四个实验维度。YAML 方式是完整的启动定义，还会固定 checkpoint、worker、serving 设置、GPU 拓扑、采样限制和输出位置。可以从[示例配置](examples/native/train_qwen_native.yaml)开始，并把站点专属文件保存在仓库外。

## 任务配置

训练使用单个 YAML 文件配置；完整字段说明、任务目录结构与 catalog 锁定见[任务配置](docs/task-configuration.md)。


## License

HarborRL 使用 MIT 许可证发布。

## Citation

如果你在研究中使用 HarborRL，请引用我们的技术报告：
```

```

## 致谢

感谢 [Slime](https://github.com/THUDM/slime) 和 [Harbor](https://github.com/harbor-framework/harbor) 为训练与任务执行构建的基础设施。
