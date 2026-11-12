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

- **2026-11-12** 新增[网关水平分片](docs/deployment.md)：静态路由、弹性自动生长（远端优先）与自动缩容。
- **2026-11-12** 新增 [Hub 部署模式](docs/deployment.md)：适配单向网络集群，节点经 HTTP 拉取任务，无需跨机 SSH。
- **2026-11-12** 新增[在线数学训练](docs/math-distill.md)（GRPO、DAPO、过程奖励、OPD、PRM 蒸馏），统一接入 Slime/Megatron + SGLang 管线。
- **2026-11-12** 新增[配置向导](docs/configuration-wizard.md)、[Web 控制台](docs/web-console.md)与带审计清理的 [Sandbox 执行](docs/sandbox-rollout.md)。
- **2026-11-12** 10/2026 📣 HarborRL 发布

## 项目介绍

HarborRL 是一个轻量级框架，用于在可验证、可沙箱隔离的任务上训练会使用工具的智能体。它把模型服务、轨迹生成、工具执行、验证器奖励和策略更新组成一条可复现流水线。

HarborRL 支持 **Models × Algorithms × Tasks × Harnesses (MATH)** 的灵活组合：

- **Models：**使用 Qwen 和 GLM 作为基础模型进行 RL 训练。
- **Algorithms：**支持 GRPO 和 DAPO 等 RL 训练算法。
- **Tasks：**在隔离 worker 中对 **200+** 个 Harbor 格式任务进行 RL 训练，包括 Terminal-Bench、SWE-Bench、Deep-SWE 等。
- **Harnesses：**支持 **40+** 个 harness 框架进行 RL 训练，包括 Claude Code、Codex、LangGraph、SWE-agent 等。

## AI 辅助配置

如果你希望让 AI agent 辅助完成本 repo 的部署和训练验证，请让它先阅读并严格遵循
[docs/agents/environment-check.md](docs/agents/environment-check.md)。

离线选项目录、交互式配置和 answer-file 配置方式见[配置 CLI 指南](docs/configuration-wizard.md)。
可以从 `harborrl options` 或 `harborrl init --help` 开始。
运行 `harborrl web` 可打开本地浏览器控制台，完成配置、校验、dry-run、训练启动和日志查看；
远程访问与生命周期说明见 [Web 控制台指南](docs/web-console.md)。

推荐的实际操作流程是把下面这段提示词交给 agent，并把尖括号内容替换成你的机器信息：

```text
阅读 docs/agents/environment-check.md，并严格按该协议执行。
目标：[仅 CPU 开发 | CPU rollout 与 GPU training 分离的原生部署]
配置：<Path to config.yaml>，或 none
拓扑：controller=<Your IP address>:<Port Number>;
  rollout workers=<Rollout worker IP address>:<Port Number>;
  GPU machine=<GPU machine IP address>:<Port Number>

先只执行只读检查。返回协议要求的结构化报告，并告诉我能解决一个阻塞项的
最小下一步命令。未经我明确同意，不要安装、下载、拉取镜像、挂载存储或启动
服务。每次经我同意修改后，重新运行相关检查。原生部署最后执行 harborrl
doctor 和 launch_native --dry-run。
```

你可以按以下清单确认部署是否正确：

1. agent 报告一个验收等级：`cpu_development`、`deployment_preflight` 或
   `training`；不得把 CPU-only 检查或 dry run 提升为更高等级。
2. CPU 开发通过的标准是 `uv run harborrl --help` 和 `uv run pytest` 均成功。
3. 原生部署通过的标准是所有可静态验证的 rollout worker、GPU 和跨机检查均为
   `ready`；`harborrl doctor` 没有失败检查；`uv run harborrl train
   <Path to config.yaml> --dry-run` 能解析出 launch plan。
4. 每个 `unknown` 项都必须说明推迟原因。训练前的 live gateway 和 task image
   检查可以合理地保持为 deferred。
5. agent 必须等待用户明确同意后才能启动真实训练。

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

训练由一个 YAML 文件配置，其 section 和字段集合固定。相对路径按 YAML 文件所在目录解析，可重复使用 `--set section.field=value` 覆盖并再次校验。请从 [examples/native/train_qwen_native.yaml](examples/native/train_qwen_native.yaml) 开始修改 site-specific 配置。

配置大体分为三部分：`execution` 固定使用 `harbor_job`；`harbor`、`gateway` 和 `deployment` 描述 rollout worker、服务入口与 GPU 拓扑；`tasks`、`harness`、`model`、`training`、`sampling` 和 `output` 描述实验内容与输出位置。

HarborRL 直接消费 Harbor 认可的 native task，不维护各 benchmark 的私有任务转换格式。典型任务目录是：

```text
task/
├── task.toml
├── instruction.md
├── environment/Dockerfile
└── tests/test.sh
```

其中 `task.toml` 声明资源、agent 与 verifier，`instruction.md` 是给 agent 的任务说明，`environment/Dockerfile` 构建隔离执行环境，`tests/test.sh` 负责验证并输出 `reward.json` 或 `reward.txt`。训练时，这些任务会被锁定到 catalog 中，记录任务 ID、上游 revision、本地路径、内容 digest 和 reward 解析规则；示例见 [examples/native/catalog.json](examples/native/catalog.json)。

仓库内置的可直接运行示例是 `hello_world`。实际训练任务不由本 repo 重新打包，而是从 Harbor Hub 获取符合 Harbor native task 格式的 dataset/task；可先查看[兼容性汇总](configs/harbor_hub/public_compatibility_summary_zh.md)了解已验证覆盖情况。取回 task root 后，先用 `harborrl hub inspect` 做本地静态校验，再锁定进自己的 catalog。

## License

HarborRL 使用 MIT 许可证发布。

## Citation

如果你在研究中使用 HarborRL，请引用我们的技术报告：
```

```

## 致谢

感谢 [Slime](https://github.com/THUDM/slime) 和 [Harbor](https://github.com/harbor-framework/harbor) 为训练与任务执行构建的基础设施。
