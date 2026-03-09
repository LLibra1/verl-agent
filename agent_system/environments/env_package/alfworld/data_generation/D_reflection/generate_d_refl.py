# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
D_refl 反思数据生成脚本。

基于论文：Agent learning via Early Experience（GiGPO）

核心逻辑：
    D_rollout 中每个步骤包含 3 条替代动作记录 (s_i, a^k_i, s^k_i)（k=1,2,3），
    以及对应的专家动作 a_i 和专家后继状态 s_{i+1}。脚本按 (task_id, step) 将
    同一步骤的 3 条记录分为一组，一次性将 3 个替代动作及其后继状态填入 prompt，
    调用强大语言模型生成一条链式思维（chain-of-thought）反思文本，解释为什么
    专家动作 a_i 优于这 3 个替代动作。

输入：
    - D_rollout.json：包含 (s_i, a_i, a^k_i, s^k_i) 的样本列表，每步 3 条记录
    - dexpert_test_100.json：专家轨迹数据，用于查找专家后继状态 s_{i+1}

输出：
    - D_refl.json：每步一条反思记录，包含 3 个替代动作和一条反思文本

路径说明：
    - 输入/输出路径已内置在脚本中，无需通过命令行参数传入
    - 如需修改路径，请编辑脚本底部 main() 函数中的 DEFAULT_* 变量
"""

import os
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from collections import defaultdict

# --------------------------------------------------------------------------- #
#  默认路径配置（直接在脚本中设置，避免繁琐的命令行参数）
# --------------------------------------------------------------------------- #

# 脚本所在目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# D_rollout 目录
_ROLLOUT_DIR = os.path.join(_SCRIPT_DIR, "..", "D_rollout")

# 输入：D_rollout 样本文件
DEFAULT_ROLLOUT_FILE = os.path.join(_ROLLOUT_DIR, "D_rollout.json")
# 输入：专家轨迹文件，用于查找专家后继状态 s_{i+1}
DEFAULT_DEXPERT_FILE = os.path.join(_ROLLOUT_DIR, "dexpert_test_100.json")
# 输出：反思数据文件
DEFAULT_OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "D_refl.json")

# --------------------------------------------------------------------------- #
#  强模型 API 默认配置（支持 OpenAI / DeepSeek / 其他兼容 OpenAI 格式的 API）
# --------------------------------------------------------------------------- #

# 默认模型名称：推荐使用 GPT-4o（OpenAI）或 deepseek-chat（DeepSeek）
DEFAULT_MODEL = "gpt-4o"

# API Base URL（留空则使用 OpenAI 官方地址；若使用 DeepSeek 填入对应地址）
#   OpenAI  : https://api.openai.com/v1
#   DeepSeek: https://api.deepseek.com/v1
DEFAULT_API_BASE = "https://api.openai.com/v1"

# 最大并发请求数（建议 ≤ 10，避免触发速率限制）
DEFAULT_MAX_WORKERS = 5

# 每次 API 请求最大 token 数（生成部分）
DEFAULT_MAX_TOKENS = 512

# 生成时采样温度（低温度保证反思内容质量稳定；0.0~1.0）
DEFAULT_TEMPERATURE = 0.3

# API 调用失败时最大重试次数
DEFAULT_MAX_RETRIES = 3

# 两次重试之间等待秒数
DEFAULT_RETRY_DELAY = 5.0

# --------------------------------------------------------------------------- #
#  日志配置
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  工具函数
# --------------------------------------------------------------------------- #

def load_json(path: str):
    """加载 JSON 文件并返回解析后的对象。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    """将对象保存为格式化 JSON 文件（UTF-8，不转义非 ASCII 字符）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存至: %s", path)


def build_expert_next_state_index(dexpert_data: list) -> dict:
    """
    从专家轨迹数据中构建 (task_id, step) -> 专家后继状态 的索引。

    此索引仅用于填充 prompt 中的 {Future State of Expert Action}（即专家后继状态
    s_{i+1}，对应模板中的 Expected Outcome si+1）。

    注意：prompt 中 {State k}（替代动作的后继状态）来自 D_rollout 的 next_state_sji
    字段，始终存在，与本索引无关，不受最后一步的影响。

    专家轨迹第 step 步的后继状态 s_{i+1} 等于**同一轨迹**第 step+1 步的 state_si。
    索引键使用 (task_id, step)，因此跨轨迹边界不会相互污染：
    轨迹 A 的最后一步绝不会取到轨迹 B 第一步的状态。

    对于轨迹最后一步（step+1 不存在），专家动作已使任务完成，
    专家后继状态（Expected Outcome si+1）人工设为固定字符串 "success"，
    以确保 prompt 能够正常使用。

    参数:
        dexpert_data: 专家轨迹条目列表（dexpert_test_100.json）

    返回:
        字典，键为 (task_id, step)，值为专家后继状态文本（最后一步为 "success"）
    """
    # 按 (task_id, step) 建立状态映射
    step_states: dict = {}
    for item in dexpert_data:
        key = (item["task_id"], item["step"])
        step_states[key] = item["state_si"]["current_state"]

    # 对于 step S：后继状态来自同一 task_id 的 step S+1。
    # 最后一步（step S+1 不存在）：专家动作完成任务，标记为 "success"。
    next_state_index: dict = {}
    for (task_id, step), _ in step_states.items():
        next_key = (task_id, step + 1)
        if next_key in step_states:
            next_state_index[(task_id, step)] = step_states[next_key]
        else:
            next_state_index[(task_id, step)] = "success"

    return next_state_index


def build_prompt(
    template: str,
    alternatives: list,
    situation: str,
    expert_action: str,
    expert_next_state: str,
) -> str:
    """
    将同一步骤的 3 个替代动作及其他字段填入 prompt 模板。

    D_rollout 中每步包含 3 条记录（rollout001~003），按 (task_id, step) 分组后，
    3 个替代动作一次性填入模板的 {Alt Action 1}~{Alt Action 3} 占位符。

    模板占位符与字段的对应关系：
        {Situation Description}          <- state_si.current_state
        {Expert Action}                  <- expert_action_ai
        {Future State of Expert Action}  <- expert_next_state（由 dexpert 推导）
        {Alt Action k}                   <- 第 k 个替代动作（k=1,2,3）
        {State k}                        <- 第 k 个替代动作的后继状态（k=1,2,3）

    参数:
        template:          REFLECTION_TEMPLATE 字符串
        alternatives:      长度为 3 的列表，每个元素为 dict，
                           包含 "action" 和 "next_state" 两个键
        situation:         当前状态文本（state_si.current_state）
        expert_action:     专家动作文本（expert_action_ai）
        expert_next_state: 专家动作执行后的后继状态 s_{i+1}

    返回:
        填充完毕的 prompt 字符串
    """
    prompt = template.replace("{Situation Description}", situation.strip())
    prompt = prompt.replace("{Expert Action}", expert_action.strip())
    prompt = prompt.replace("{Future State of Expert Action}", expert_next_state.strip())

    # 填充 3 个替代动作的占位符 {Alt Action k} 和 {State k}
    for k in range(1, 4):
        alt = alternatives[k - 1]
        prompt = prompt.replace(f"{{Alt Action {k}}}", alt["action"].strip())
        prompt = prompt.replace(f"{{State {k}}}", alt["next_state"].strip())

    return prompt.strip()


# --------------------------------------------------------------------------- #
#  API 调用封装
# --------------------------------------------------------------------------- #

def create_llm_client(api_key: str, api_base: str = DEFAULT_API_BASE):
    """
    创建 OpenAI 兼容客户端（线程安全，可在并发调用中复用）。

    参数:
        api_key:  API 密钥
        api_base: API 基础 URL

    返回:
        OpenAI 客户端实例
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "缺少 openai 包，请执行: pip install openai>=1.0.0"
        )
    return OpenAI(api_key=api_key, base_url=api_base)


def call_llm_api(
    prompt: str,
    client: Any,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> Optional[str]:
    """
    调用兼容 OpenAI Chat Completions 格式的强模型 API 生成反思文本。

    采用 system + user 双角色消息结构：
        - system：指定模型角色（简洁，节省 token）
        - user：填充好的反思 prompt

    参数:
        prompt:      完整的用户 prompt（已填充情境、动作等信息）
        client:      OpenAI 兼容客户端实例（由 create_llm_client 创建，可复用）
        model:       模型名称（如 "gpt-4o"、"deepseek-chat"）
        max_tokens:  最大生成 token 数
        temperature: 采样温度（0.0~1.0；低温度保证内容稳定，默认 0.3）
        max_retries: 失败重试次数上限
        retry_delay: 重试间隔（秒）

    返回:
        模型生成的反思文本字符串，失败时返回 None
    """
    # system 消息：简短角色描述，节省 token
    system_msg = (
        "You are an expert AI assistant that analyzes decision-making in "
        "interactive text environments. Provide concise, logical self-reflection."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(
                "API 调用失败（第 %d/%d 次）: %s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    logger.error("API 调用全部失败，跳过此条目")
    return None


# --------------------------------------------------------------------------- #
#  核心生成逻辑
# --------------------------------------------------------------------------- #

def generate_d_refl(
    rollout_file: str,
    dexpert_file: str,
    output_file: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    resume: bool = True,
) -> list:
    """
    批量生成 D_refl 反思数据集。

    流程：
        1. 加载 D_rollout.json 和 dexpert 专家轨迹
        2. 构建专家后继状态索引（task_id, step） -> s_{i+1}
        3. 按 (task_id, step) 将 D_rollout 记录分组（每步 3 条替代动作）
        4. 为每个分组构建 prompt（一次性填入 3 个替代动作）
        5. 使用线程池并发调用强模型 API 生成反思文本（由 max_workers 控制并发数）
        6. 保存结果到 D_refl.json（每步一条记录）

    参数:
        rollout_file: D_rollout.json 的路径
        dexpert_file: 专家轨迹 JSON 的路径（用于构建专家后继状态索引）
        output_file:  输出 D_refl.json 的路径
        api_key:      API 密钥
        model:        强模型名称
        api_base:     API 基础 URL
        max_tokens:   生成最大 token 数
        temperature:  采样温度（0.0~1.0）
        max_workers:  并发 API 请求线程数（建议 ≤ 10，避免触发速率限制）
        max_retries:  最大重试次数
        retry_delay:  重试间隔（秒）
        resume:       若输出文件已存在，跳过已完成的条目（断点续传）

    返回:
        生成的 D_refl 数据列表
    """
    # ------------------------------------------------------------------ #
    #  导入反思 prompt 模板
    # ------------------------------------------------------------------ #
    # 使用相对路径导入，支持直接运行脚本和模块两种方式
    import importlib.util
    _prompt_path = os.path.join(_SCRIPT_DIR, "reflection_prompt.py")
    spec = importlib.util.spec_from_file_location("reflection_prompt", _prompt_path)
    reflection_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reflection_module)
    TEMPLATE = reflection_module.REFLECTION_TEMPLATE

    # ------------------------------------------------------------------ #
    #  加载数据
    # ------------------------------------------------------------------ #
    logger.info("加载 D_rollout 数据: %s", rollout_file)
    rollout_data: list = load_json(rollout_file)
    logger.info("共 %d 条 rollout 记录", len(rollout_data))

    # ------------------------------------------------------------------ #
    #  按 (task_id, step) 将 D_rollout 记录分组（每步 3 条替代动作）
    # ------------------------------------------------------------------ #
    step_groups: dict = defaultdict(list)
    for item in rollout_data:
        step_groups[(item["task_id"], item["step"])].append(item)

    # 按首条记录的 idx 排序，确保输出顺序稳定
    sorted_group_keys = sorted(step_groups.keys(), key=lambda k: step_groups[k][0]["idx"])

    logger.info(
        "按 (task_id, step) 分组完毕：共 %d 个步骤分组，每组 3 个替代动作",
        len(sorted_group_keys),
    )

    # ------------------------------------------------------------------ #
    #  构建专家后继状态索引（从 dexpert 获取 Expected Outcome si+1）
    # ------------------------------------------------------------------ #
    if not os.path.exists(dexpert_file):
        raise FileNotFoundError(
            f"专家轨迹文件不存在: {dexpert_file}\n"
            "此文件为必需输入，用于构建专家后继状态索引（Expected Outcome si+1）。\n"
            "请确认文件路径正确，或通过 --dexpert_file 参数指定。"
        )
    logger.info("加载专家轨迹数据: %s", dexpert_file)
    dexpert_data: list = load_json(dexpert_file)
    logger.info("共 %d 条专家轨迹记录", len(dexpert_data))
    expert_next_index = build_expert_next_state_index(dexpert_data)
    logger.info("构建专家后继状态索引完毕，共 %d 条", len(expert_next_index))

    # ------------------------------------------------------------------ #
    #  创建 API 客户端（复用同一实例，避免每次调用重建连接）
    # ------------------------------------------------------------------ #
    client = create_llm_client(api_key=api_key, api_base=api_base)

    # ------------------------------------------------------------------ #
    #  断点续传：加载已有结果，跳过已完成条目
    # ------------------------------------------------------------------ #
    results: list = []
    done_ids: set = set()
    if resume and os.path.exists(output_file):
        try:
            results = load_json(output_file)
            done_ids = {r["id"] for r in results if r.get("reflection")}
            logger.info("断点续传：已完成 %d 条，跳过...", len(done_ids))
        except Exception:
            logger.warning("无法加载已有结果文件，从头开始")
            results = []
            done_ids = set()

    # ------------------------------------------------------------------ #
    #  并发生成反思文本（使用线程池，由 max_workers 控制并发数）
    # ------------------------------------------------------------------ #
    skipped_no_next_state = 0
    pending: list = []

    for group_key in sorted_group_keys:
        items = step_groups[group_key]
        task_id, step = group_key
        # 组级 ID：去掉 rollout 后缀，使用步骤级标识
        first = items[0]
        group_id = first["id"].rsplit("_rollout", 1)[0]

        if group_id in done_ids:
            logger.debug("跳过已完成分组: %s", group_id)
            continue

        # 从 dexpert 索引中查找专家后继状态 s_{i+1}（prompt 中的 Expected Outcome）。
        # 轨迹最后一步在索引中固定为 "success"，中间步骤取下一步的 state_si。
        expert_next = expert_next_index.get((task_id, step))
        if expert_next is None:
            logger.warning(
                "分组 %s 找不到专家后继状态（task_id=%s, step=%d），使用占位文本",
                group_id, task_id, step,
            )
            expert_next = (
                "[Expert action leads to task completion / no further state recorded]"
            )
            skipped_no_next_state += 1

        # 构建 3 个替代动作列表（按 rollout 编号排序以保证顺序稳定）
        sorted_items = sorted(items, key=lambda x: x["id"])
        alternatives = [
            {
                "action": it["alternative_action_j"],
                "next_state": it["next_state_sji"],
            }
            for it in sorted_items
        ]

        situation = first["state_si"]["current_state"]
        expert_action = first["expert_action_ai"]

        prompt = build_prompt(
            TEMPLATE, alternatives, situation, expert_action, expert_next
        )
        pending.append({
            "group_id": group_id,
            "task_id": task_id,
            "idx": first["idx"],
            "task": first["task"],
            "step": step,
            "state_si": first["state_si"],
            "expert_action_ai": expert_action,
            "alternatives": alternatives,
            "expert_next": expert_next,
            "prompt": prompt,
            "gamefile": first.get("gamefile", []),
        })

    logger.info("待生成分组数: %d，并发线程数: %d", len(pending), max_workers)

    # 使用字典按 group_id 存储结果
    result_map: dict = {r["id"]: r for r in results}
    completed_count = len(results)
    total_count = len(results) + len(pending)

    def _call_one(group_info: dict):
        """单组并发调用任务（供线程池使用）。"""
        gid = group_info["group_id"]
        logger.info(
            "生成反思: id=%s, task=%s, step=%d",
            gid, group_info["task"][:50], group_info["step"],
        )
        text = call_llm_api(
            prompt=group_info["prompt"],
            client=client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if text is None:
            logger.warning("分组 %s API 调用失败，reflection 置为空字符串", gid)
            text = ""
        entry = {
            "task_id": group_info["task_id"],
            "idx": group_info["idx"],
            "id": gid,
            "task": group_info["task"],
            "step": group_info["step"],
            "state_si": group_info["state_si"],
            "expert_action_ai": group_info["expert_action_ai"],
            "alternative_actions": group_info["alternatives"],
            "reflection": text,
            "gamefile": group_info["gamefile"],
        }
        return gid, entry

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_call_one, g): g["group_id"] for g in pending
        }
        for future in as_completed(future_map):
            try:
                gid, entry = future.result()
            except Exception:
                fid = future_map.get(future, "<unknown>")
                logger.error("分组 %s 处理时发生异常，跳过", fid, exc_info=True)
                continue
            result_map[gid] = entry
            completed_count += 1
            # 每完成 10 条自动保存一次（防止意外中断丢失进度）
            if completed_count % 10 == 0:
                ordered = sorted(result_map.values(), key=lambda r: r.get("idx", 0))
                save_json(ordered, output_file)
                logger.info(
                    "自动保存进度：已完成 %d/%d 条",
                    completed_count, total_count,
                )

    # 按 idx 排序最终结果
    results = sorted(result_map.values(), key=lambda r: r.get("idx", 0))

    # ------------------------------------------------------------------ #
    #  最终保存
    # ------------------------------------------------------------------ #
    save_json(results, output_file)
    logger.info(
        "D_refl 生成完毕。共 %d 条反思数据（每条含 3 个替代动作），"
        "%d 条因找不到专家后继状态使用占位文本",
        len(results), skipped_no_next_state,
    )
    return results


# --------------------------------------------------------------------------- #
#  命令行入口
# --------------------------------------------------------------------------- #

def parse_args():
    """解析命令行参数（仅 API 密钥等敏感配置项，路径已内置）。"""
    parser = argparse.ArgumentParser(
        description=(
            "利用强模型 API 生成 D_refl 反思数据集（chain-of-thought）。\n"
            "输入/输出路径已内置在脚本中，仅需提供 API 密钥。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help=(
            "强模型 API 密钥。\n"
            "默认读取环境变量 OPENAI_API_KEY，也可直接传入。\n"
            "示例: --api_key sk-xxxx"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            f"强模型名称（默认: {DEFAULT_MODEL}）。\n"
            "OpenAI  示例: gpt-4o, gpt-4-turbo\n"
            "DeepSeek示例: deepseek-chat, deepseek-reasoner"
        ),
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=DEFAULT_API_BASE,
        help=(
            f"API 基础 URL（默认: {DEFAULT_API_BASE}）。\n"
            "DeepSeek: https://api.deepseek.com/v1\n"
            "其他兼容 OpenAI 格式的 API 填写对应地址"
        ),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"每次 API 请求最大生成 token 数（默认: {DEFAULT_MAX_TOKENS}）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            f"生成采样温度（默认: {DEFAULT_TEMPERATURE}）。\n"
            "范围 0.0~1.0；低值使输出更稳定，高值增加多样性"
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"最大并发请求数（默认: {DEFAULT_MAX_WORKERS}）",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"API 调用失败最大重试次数（默认: {DEFAULT_MAX_RETRIES}）",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help=f"重试间隔秒数（默认: {DEFAULT_RETRY_DELAY}）",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="不使用断点续传，从头重新生成所有数据",
    )
    # 高级：允许用命令行覆盖内置路径（可选）
    parser.add_argument(
        "--rollout_file",
        type=str,
        default=DEFAULT_ROLLOUT_FILE,
        help=f"D_rollout.json 路径（默认: {DEFAULT_ROLLOUT_FILE}）",
    )
    parser.add_argument(
        "--dexpert_file",
        type=str,
        default=DEFAULT_DEXPERT_FILE,
        help=f"专家轨迹 JSON 路径（默认: {DEFAULT_DEXPERT_FILE}）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"输出 D_refl.json 路径（默认: {DEFAULT_OUTPUT_FILE}）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查 API 密钥
    if not args.api_key:
        raise ValueError(
            "未提供 API 密钥！\n"
            "方式一: export OPENAI_API_KEY=sk-xxxx\n"
            "方式二: python generate_d_refl.py --api_key sk-xxxx"
        )

    logger.info("=" * 60)
    logger.info("D_refl 反思数据生成任务启动")
    logger.info("  输入 D_rollout : %s", args.rollout_file)
    logger.info("  输入 D_expert  : %s", args.dexpert_file)
    logger.info("  输出 D_refl    : %s", args.output_file)
    logger.info("  模型           : %s", args.model)
    logger.info("  API Base       : %s", args.api_base)
    logger.info("  最大 token     : %d", args.max_tokens)
    logger.info("  采样温度       : %.2f", args.temperature)
    logger.info("  断点续传       : %s", not args.no_resume)
    logger.info("=" * 60)

    generate_d_refl(
        rollout_file=args.rollout_file,
        dexpert_file=args.dexpert_file,
        output_file=args.output_file,
        api_key=args.api_key,
        model=args.model,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
