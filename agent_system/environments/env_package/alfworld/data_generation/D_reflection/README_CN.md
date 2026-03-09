# D_reflection 反思数据生成工具

本模块基于论文 **"Agent learning via Early Experience"（GiGPO）**，利用强大语言模型（如 GPT-4o、DeepSeek-Chat）为 ALFWorld 任务中的每步 D_rollout 记录（每步 3 个替代动作）自动生成高质量的链式思维（chain-of-thought）反思数据（D_refl）。

---

## 目录

- [背景与原理](#背景与原理)
- [目录结构](#目录结构)
- [数据格式](#数据格式)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
- [常见问题](#常见问题)

---

## 背景与原理

### 论文核心思想

在 GiGPO 方法中，D_refl 数据集的构建遵循以下逻辑：

> 对于 D_rollout 中的每个步骤，有 3 个替代动作记录 **(s_i, a^k_i, s^k_i)**（k=1,2,3），以及对应的专家动作 **a_i** 和专家后继状态 **s_{i+1}**。脚本将同一步骤的 3 条记录分为一组，一次性传入 prompt，利用强大语言模型生成一条链式思维反思文本，解释为什么专家动作 a_i 优于这 3 个替代动作。

简而言之：
- **输入**：当前状态 s_i、专家动作 a_i（及其后继 s_{i+1}）、3 个替代动作 a^k_i（及其后继 s^k_i）
- **输出**：一条自然语言反思文本，分析 3 个替代动作的局限性并论证专家动作的优越性

### 数据流

```
D_rollout.json  ──→  按 (task_id, step) 分组（每步 3 条替代动作）──┐
                                                                      ▼
dexpert_test_100.json  ──→  构建专家后继状态索引（Expected Outcome）──→  填充 prompt（3 个替代动作一次性传入）
                                                                      │
reflection_prompt.py  ────────────────────────────────────────────────┘
                                                                      │
                                                       强模型 API 调用
                                                                      │
                                                                      ▼
                                                             D_refl.json（每步一条记录）
```

> **注**：专家后继状态 s_{i+1}（Expected Outcome）始终从 `dexpert_test_100.json` 构建的索引中获取。
> 对于轨迹最后一步（step+1 不存在），专家动作已完成任务，Expected Outcome 固定为 `"success"`。
> 替代动作的后继状态 `{State k}`（即 D_rollout 中的 `next_state_sji`）始终存在，不受最后一步影响。

---

## 目录结构

```
D_reflection/
├── __init__.py                  # 包初始化文件
├── reflection_prompt.py         # 反思 prompt 模板（REFLECTION_TEMPLATE）
├── generate_d_refl.py           # 主生成脚本
├── run_generate_d_refl.sh       # Shell 一键执行脚本
├── D_refl.json                  # 生成的反思数据（运行后自动创建）
└── README_CN.md                 # 本文档
```

> **注意**：`D_refl.json` 在首次运行后自动生成，无需手动创建。

---

## 数据格式

### 输入：D_rollout.json（位于 `../D_rollout/` 目录）

D_rollout 中**每个步骤包含 3 条记录**（rollout001~003），共享相同的 `task_id`、`step`、`state_si`、`expert_action_ai`，但各有不同的替代动作和后继状态：

```json
[
  {
    "task_id": "trial_T20190908_110055_655553",
    "idx": 1,
    "id": "traj_0001_step001_rollout001",
    "task": "put a cool mug in coffeemachine.",
    "step": 1,
    "state_si": { "current_state": "-= Welcome to TextWorld, ALFRED! =-\n..." },
    "admissible_actions": ["go to cabinet 1", "go to coffeemachine 1", "..."],
    "expert_action_ai": "go to coffeemachine 1",
    "alternative_action_j": "go to countertop 1",
    "next_state_sji": "You have taken the action 1: 'go to countertop 1'...",
    "gamefile": ["/path/to/game.tw-pddl"],
    "is_expert": false
  },
  {
    "task_id": "trial_T20190908_110055_655553",
    "idx": 2,
    "id": "traj_0001_step001_rollout002",
    "step": 1,
    "alternative_action_j": "go to cabinet 1",
    "next_state_sji": "You have taken the action 1: 'go to cabinet 1'...",
    "...": "（其余字段与 rollout001 相同）"
  },
  {
    "task_id": "trial_T20190908_110055_655553",
    "idx": 3,
    "id": "traj_0001_step001_rollout003",
    "step": 1,
    "alternative_action_j": "go to stoveburner 3",
    "next_state_sji": "You have taken the action 1: 'go to stoveburner 3'...",
    "...": "（其余字段与 rollout001 相同）"
  }
]
```

> **说明**：脚本自动按 `(task_id, step)` 将这 3 条记录分为一组，一次性传入 prompt。专家后继状态（Expected Outcome si+1）由脚本从 `dexpert_test_100.json` 自动推导获取。

### 输出：D_refl.json

**每步一条记录**，包含 3 个替代动作和一条合并的反思文本：

```json
{
  "task_id": "trial_T20190908_110055_655553",
  "idx": 1,
  "id": "traj_0001_step001",
  "task": "put a cool mug in coffeemachine.",
  "step": 1,
  "state_si": { "current_state": "-= Welcome to TextWorld, ALFRED! =-\n..." },
  "expert_action_ai": "go to coffeemachine 1",
  "alternative_actions": [
    { "action": "go to countertop 1", "next_state": "You have taken..." },
    { "action": "go to cabinet 1", "next_state": "You have taken..." },
    { "action": "go to stoveburner 3", "next_state": "You have taken..." }
  ],
  "reflection": "Looking at the situation, my goal is to put a cool mug in coffeemachine...",
  "gamefile": ["/path/to/game.tw-pddl"]
}
```

**字段说明：**

| 字段 | 含义 |
|------|------|
| `task_id` | 任务/轨迹的唯一标识符 |
| `idx` | 全局数据集计数（从 1 开始） |
| `id` | 该条目的唯一标识符（步骤级，格式：`traj_XXXX_stepYYY`） |
| `task` | 任务描述/目标 |
| `step` | 轨迹中的步骤编号（从 1 开始） |
| `state_si` | 当前状态，包含动作历史和当前观察 |
| `expert_action_ai` | 专家动作 a_i |
| `alternative_actions` | 3 个替代动作列表，每项含 `action`（替代动作）和 `next_state`（后继状态） |
| `reflection` | 强模型生成的链式思维反思文本，分析 3 个替代动作的优劣并论证专家动作的优越性 |
| `gamefile` | 对应的游戏文件路径列表 |

---

## 环境依赖

### 必需依赖

```bash
pip install openai>=1.0.0
```

> **提示**：脚本运行时会自动检测并安装 `openai` 包，也可提前手动安装。

### API 密钥

本工具支持任何兼容 OpenAI Chat Completions 格式的 API，包括：

| 服务商 | 推荐模型 | API Base URL |
|--------|----------|-------------|
| OpenAI | `gpt-4o`、`gpt-4-turbo` | `https://api.openai.com/v1` |
| DeepSeek | `deepseek-chat`、`deepseek-reasoner` | `https://api.deepseek.com/v1` |
| 其他兼容 OpenAI 的服务 | 参考对应文档 | 参考对应文档 |

---

## 快速开始

### 方式一：Shell 脚本（推荐，最简单）

```bash
# 切换到项目根目录
cd /path/to/verl-agent

# 设置 API 密钥环境变量
export OPENAI_API_KEY=sk-xxxx

# 一键运行（使用默认 GPT-4o）
bash agent_system/environments/env_package/alfworld/data_generation/D_reflection/run_generate_d_refl.sh
```

### 方式二：使用 DeepSeek 模型

```bash
bash agent_system/environments/env_package/alfworld/data_generation/D_reflection/run_generate_d_refl.sh \
    "your-deepseek-api-key" \
    "deepseek-chat" \
    "https://api.deepseek.com/v1"
```

### 方式三：直接运行 Python 脚本

```bash
# 使用环境变量（推荐）
export OPENAI_API_KEY=sk-xxxx
python3 agent_system/environments/env_package/alfworld/data_generation/D_reflection/generate_d_refl.py

# 或通过参数传入密钥
python3 agent_system/environments/env_package/alfworld/data_generation/D_reflection/generate_d_refl.py \
    --api_key sk-xxxx \
    --model gpt-4o
```

### 方式四：从头重新生成（覆盖已有结果）

```bash
export OPENAI_API_KEY=sk-xxxx
python3 agent_system/environments/env_package/alfworld/data_generation/D_reflection/generate_d_refl.py \
    --no_resume
```

---

## 脚本说明

### generate_d_refl.py（主生成脚本）

**功能：**
- 加载 `D_rollout.json` 和 `dexpert_test_100.json`（路径已内置，无需命令行参数）
- 按 `(task_id, step)` 将 D_rollout 记录分组（每步 3 条替代动作）
- 为每个分组从专家轨迹中推导出专家后继状态 s_{i+1}
- 将 `reflection_prompt.py` 中的 `REFLECTION_TEMPLATE` 一次性填入 3 个替代动作并调用强模型 API
- 支持**断点续传**：中断后重新运行自动跳过已完成条目
- 每 10 条自动保存一次进度，防止意外中断丢失数据

**所有可选参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_key` | 读取 `$OPENAI_API_KEY` | API 密钥 |
| `--model` | `gpt-4o` | 强模型名称 |
| `--api_base` | `https://api.openai.com/v1` | API 基础 URL |
| `--max_tokens` | `512` | 每次生成最大 token 数 |
| `--temperature` | `0.3` | 采样温度（0.0~1.0；低值稳定，高值多样） |
| `--max_workers` | `5` | 并发 API 请求线程数（建议 ≤ 10） |
| `--max_retries` | `3` | 失败重试次数 |
| `--retry_delay` | `5.0` | 重试间隔（秒） |
| `--no_resume` | 不设置 | 从头重新生成，忽略已有结果 |
| `--rollout_file` | 内置路径 | 自定义 D_rollout.json 路径 |
| `--dexpert_file` | 内置路径 | 自定义专家轨迹 JSON 路径 |
| `--output_file` | 内置路径 | 自定义输出 D_refl.json 路径 |

### run_generate_d_refl.sh（Shell 包装脚本）

**功能：** 简化命令行调用，自动切换到项目根目录并检查依赖。

**参数（按顺序）：**

| 位置 | 参数 | 示例 |
|------|------|------|
| `$1` | API 密钥（可选，默认读取环境变量） | `sk-xxxx` |
| `$2` | 模型名称（可选，默认 `gpt-4o`） | `deepseek-chat` |
| `$3` | API Base URL（可选，默认 OpenAI 官方） | `https://api.deepseek.com/v1` |
| `$4` | 额外参数（可选，如 `--no_resume`） | `--no_resume` |

### reflection_prompt.py（反思 prompt 模板）

定义了 `REFLECTION_TEMPLATE`，包含以下占位符（由脚本自动填充）：

| 占位符 | 对应字段 |
|--------|---------|
| `{Situation Description}` | `state_si.current_state` |
| `{Expert Action}` | `expert_action_ai` |
| `{Future State of Expert Action}` | 专家后继状态 s_{i+1}（Expected Outcome，从专家轨迹推导） |
| `{Alt Action 1}` | 第 1 个替代动作（`alternative_actions[0].action`） |
| `{State 1}` | 第 1 个替代动作的后继状态（`alternative_actions[0].next_state`） |
| `{Alt Action 2}` | 第 2 个替代动作（`alternative_actions[1].action`） |
| `{State 2}` | 第 2 个替代动作的后继状态（`alternative_actions[1].next_state`） |
| `{Alt Action 3}` | 第 3 个替代动作（`alternative_actions[2].action`） |
| `{State 3}` | 第 3 个替代动作的后继状态（`alternative_actions[2].next_state`） |

---

## 常见问题

### Q1：如何修改输入/输出路径？

路径内置于 `generate_d_refl.py` 顶部的 `DEFAULT_*` 变量中：

```python
DEFAULT_ROLLOUT_FILE = os.path.join(_ROLLOUT_DIR, "D_rollout.json")
DEFAULT_DEXPERT_FILE = os.path.join(_ROLLOUT_DIR, "dexpert_test_100.json")
DEFAULT_OUTPUT_FILE  = os.path.join(_SCRIPT_DIR, "D_refl.json")
```

或者通过命令行覆盖（高级用法）：

```bash
python3 generate_d_refl.py --rollout_file /path/to/D_rollout.json \
    --dexpert_file /path/to/dexpert.json \
    --output_file /path/to/D_refl.json \
    --api_key sk-xxxx
```

### Q2：为什么部分条目的专家后继状态是 "success"？

对于每条专家轨迹的最后一步（step+1 不存在），专家动作已使任务完成，因此
Expected Outcome（si+1）固定为 `"success"`。这是正常行为，不影响数据质量。

注意：替代动作的后继状态（prompt 中的 `{State k}`，即 D_rollout 的 `next_state_sji`）
始终存在，不受轨迹最后一步的影响。

### Q3：D_rollout 记录是如何分组的？

脚本按 `(task_id, step)` 将 D_rollout 中的记录分组。每组包含同一步骤的 3 条记录
（rollout001~003），它们共享相同的 `task_id`、`step`、`state_si`、`expert_action_ai`，
但各有不同的 `alternative_action_j` 和 `next_state_sji`。

例如：`traj_0001_step001_rollout001`、`traj_0001_step001_rollout002`、
`traj_0001_step001_rollout003` 会被分为一组，输出 ID 为 `traj_0001_step001`。

### Q4：如何使用代理或自定义 API 端点？

通过 `--api_base` 参数指定任意兼容 OpenAI 格式的 API 地址：

```bash
python3 generate_d_refl.py --api_base https://your-proxy.com/v1 --api_key your-key
```

### Q5：生成速度太慢怎么办？

- 适当增大 `--max_workers`（默认 5）以提高并发数，但注意 API 速率限制
- 使用响应速度更快的模型（如 `gpt-4o-mini`）
- DeepSeek 的响应速度通常比 OpenAI 更快，且费用更低

### Q6：中途中断后如何继续？

直接重新运行脚本，断点续传默认开启，自动跳过已完成条目：

```bash
bash run_generate_d_refl.sh
```

---

## 参考文献

本工具基于论文 **"Agent learning via Early Experience"（GiGPO）** 中 D_refl 数据集的构建方法实现。

## 许可证

Apache License 2.0
