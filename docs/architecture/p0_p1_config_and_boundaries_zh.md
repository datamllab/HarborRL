# P0/P1：配置生效与模块边界

本次变更延续已有 GPU 验收的默认训练路径，保持 README、schema-2 和用户命令不变。涉及导出代码的收敛保留原 GRPO 数学公式与权重规则；本轮验证采用 CPU 合同、旧实现数值对照、shell dry-run 和包构建，不代表新增 GPU 实测。

## 配置如何生效

优先级：YAML 原值 → 按顺序应用 `--set` → 显式 `-model/-harness/-task/-alg` 选择。后端路径、可见 GPU、代理及动态库等部署环境仍由声明的环境变量提供。结构化启动不继承环境中的 `LR`、采样预算、checkpoint 开关或其他未声明训练选项，避免覆盖 YAML。

`training.learning_rate` 现在显式映射为 `LR`，进入 Slime `--lr`。旧默认 `1e-6` 保持不变；此前被忽略的非默认 YAML 学习率现在会生效。

配置字段到环境的直接映射统一维护于 `harborrl/backends/slime_v032/config.py::CONFIG_ENV`；`field_usage()` 同时记录不通过环境消费的字段，测试要求覆盖全部 schema-2 字段。

| 配置 | 实际消费者 |
| --- | --- |
| `execution.backend`、`harness.name` | 当前执行/harness 组合校验及 runner |
| `tasks.catalog` | 静态 task lock 校验；运行时生成锁定 catalog 和 prompt data |
| `harness.profile` | 最终 profile 文件校验；runner 配置 |
| `harbor.python/version/workers` | runner Python、worker runtime probe、worker 调度 |
| `gateway.host/port/advertised_url` | gateway listener 和 worker 访问入口 |
| `gateway.tokenizer_digest/template_digest/audited_raw_logprobs` | 生成证据与训练资格检查 |
| `model.checkpoint/reference/args_file` | `--hf-checkpoint`、`--load`、`--ref-load` 与模型 preset |
| `training.learning_rate/num_rollout/save_interval` | `--lr`、`--num-rollout`、`--save-interval` |
| `training.backend_contract` | 后端 selector 与版本/hooks 检查 |
| `sampling.group_size/groups_per_batch` | `--n-samples-per-prompt`、`--rollout-batch-size` |
| `sampling.max_tokens/max_context` | response/context 预算 |
| `sampling.max_attempts` | 完整组 slot 重试限制 |
| `deployment.*` | GPU 数量、TP、split/colocate 校验和启动参数 |
| `output.root` | run、checkpoint 与指标根目录 |

严格配置校验与文件检查分离：`validate()` 不读 harness/task 文件，也不执行 live probe；`validate_files()` 校验最终选定的 harness profile；`tasks.catalog.catalog()` 校验任务锁；doctor 执行 live 部署检查。`load_config()` 保留加载后校验 profile 的兼容行为，但在 override 完成后检查文件，因此可以替换已经不存在的旧 profile 路径。

## 运行证据

每次结构化训练新增：

- `effective-config.json`：解析后配置、task selectors、组件标识、运行时环境映射、声明的部署信息、doctor 观察到的包版本/后端 commit、源码证据与字段消费者。
- `effective-command.json`：shell 默认值和模型 preset 展开后的实际可执行 argv，在调用 Slime 前记录。
- `dry-run-commands/<command digest>.json`：直接运行 shell dry-run 时的命令证据，允许同目录重复检查不同参数。

有效配置快照不转储整个父进程环境，不记录代理 URL 或环境中的 provider 凭据。原有 `launch.json`、源码 manifest、source patch 和任务产物保留。

`harborrl train --dry-run` 仍只解析并输出 plan，不启动 doctor 或训练；shell 命令证据是在实际执行 shell launcher 时生成的。若进程在进入 launcher 前失败，最终命令文件可能尚不存在。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `config/native.py` | schema-2 解析、类型/约束、路径规则、最终 profile 校验；保留旧 catalog/launch_plan 薄封装 |
| `backends/slime_v032/config.py` | Slime 配置映射与字段消费者清单 |
| `tasks/catalog.py` | native catalog 与 task lock 检查 |
| `tasks/content.py` | task 内容 digest 与共享 TOML reader |
| `platform/preflight.py` | live doctor；task digest 每次 catalog 检查一次 |
| `platform/evidence.py` | 输入准备、源码证据、有效配置与最终命令记录 |
| `platform/native_train.py` | 流程派发、run 目录、子进程、信号转发；旧 doctor/准备函数保留封装 |
| `export/native.py` | canonical immutable GRPO TrainingBatch |
| `rollout/exporters/native.py` | canonical batch 的旧 record 格式视图 |

模型 tool parser 与 harness 插件接口留给 P2；本阶段不提前扩大支持组合，也不改变 serving、loss、同步屏障或 checkpoint 生命周期。

## 历史工具清单与处置

| 历史路径 | 当前调用关系 | 处置 |
| --- | --- | --- |
| `data/harbor/__main__.py` | 模块 CLI → inspector/materializer | 保留历史入口，标注 TerminalEnv 兼容用途 |
| `data/harbor/inspector.py` | 历史 CLI、materializer、probe | 保留；共享 digest/TOML 从 `tasks/content.py` 兼容导出 |
| `data/harbor/materializer.py` | 历史 CLI；生成 task.yaml、Compose、wrapper | 保留以支持旧记录复现，native 训练不使用 |
| `data/harbor/probe.py`、`receipt.py` | 旧 TerminalEnv 动态探测和奖励解析 | 保留并标注兼容用途 |
| `data/harbor/native_inspector.py` | native catalog 与 hub inspect | 直接依赖 `tasks/content.py`，不再导入历史 inspector/materializer |
| `rollout/exporters/native.py` | GroupSlots、offline smoke、旧合同调用者 | 保留格式入口，数学及 batch 校验共用 canonical exporter |

未删除本地环境、缓存、runs 或验收记录。旧接口在确认无复现/外部调用需求后再单独决定退役。

## 回归重点

1. YAML 原值与 `--set` 的学习率均进入最终 `--lr`；预算、GPU、checkpoint 和输出参数通过命令级检查。
2. 父环境的训练变量不能覆盖 YAML；有效配置不包含父环境凭据与代理值。
3. 非法 section、字段类型、override YAML 和 catalog entry 返回可定位的 CLI 错误。
4. 有效组和零方差组、多 turn、不同轨迹长度的旧导出字段与权重保持一致；缺失证据、不完整组和重复身份继续拒绝。
5. native task inspection 不导入历史转换路径；原始 verifier 与 task digest 算法保持不变。

本次验证结果：102 项 CPU 测试通过；Ruff、shell 语法、示例 smoke、训练 plan dry-run、公开树审计通过。旧/新 exporter 在有方差与零方差、多 turn、不同长度样本上的 JSON 输出逐字一致。使用满足 pyproject 构建要求的隔离环境成功构建 sdist/wheel，并在仓库外安装 wheel 后通过 CLI dry-run 与新增模块导入。
