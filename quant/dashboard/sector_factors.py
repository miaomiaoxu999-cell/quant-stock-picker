"""板块及因子页面 — AI 驱动的周期因子生成与管理"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from quant.llm.client import SiliconFlowClient, LLMConfig, LLMError
from quant.llm.prompts import build_factor_generation_prompt, build_factor_chat_messages, FACTOR_GENERATION_SYSTEM
from quant.dashboard.llm_settings import get_llm_config

# ==================== 常量 ====================

FACTORS_PATH = Path(__file__).parent.parent.parent / "data" / "sector_factors.json"

PRESET_SECTORS = [
    # 新能源
    "锂电池/锂盐", "光伏", "光伏玻璃", "风电", "储能",
    # 传统能源
    "煤炭", "石油开采", "炼化",
    # 金属
    "钢铁", "铜", "铝", "稀土",
    # 化工
    "基础化工", "磷化工", "氟化工", "钛白粉",
    # 建筑地产链
    "水泥", "浮法玻璃", "建筑工程", "建材", "房地产",
    # 农业
    "养殖(猪周期)", "白糖", "种植业",
    # 交运
    "干散货航运", "集装箱航运", "油运", "造纸",
    # 制造
    "工程机械", "重卡/商用车",
    # 科技
    "半导体", "面板/显示", "存储芯片", "消费电子",
    # 消费
    "白酒",
]


# ==================== 持久化 ====================

def _load_all_factors() -> dict:
    if FACTORS_PATH.exists():
        try:
            with open(FACTORS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_all_factors(data: dict) -> None:
    FACTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FACTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_sector(sector: str) -> dict:
    return _load_all_factors().get(sector, {})


def _save_sector(sector: str, sector_data: dict) -> None:
    all_data = _load_all_factors()
    sector_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    all_data[sector] = sector_data
    _save_all_factors(all_data)


# ==================== JSON 解析辅助 ====================

def extract_json_from_text(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON，支持 code fence / 裸 JSON / 正则"""
    # 1) ```json ... ``` 块
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) 裸 JSON 对象
    m = re.search(r"\{[\s\S]*\"factors\"[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_factors(data: dict) -> list[dict] | None:
    """校验因子数据，合法则返回 factors 列表"""
    if not isinstance(data, dict) or "factors" not in data:
        return None
    factors = data["factors"]
    if not isinstance(factors, list) or not (2 <= len(factors) <= 5):
        return None
    total = sum(f.get("weight", 0) for f in factors)
    if not (95 <= total <= 105):
        return None
    for f in factors:
        if not all(k in f for k in ("name", "weight", "description", "data_source")):
            return None
    return factors


# ==================== 权重归一化 ====================

def _normalize_weights(factors: list[dict], changed_idx: int) -> list[dict]:
    """拖动某个 slider 后，其余按比例缩放使总和 = 100"""
    n = len(factors)
    if n <= 1:
        factors[0]["weight"] = 100
        return factors

    changed_val = factors[changed_idx]["weight"]
    remaining = 100 - changed_val
    if remaining < 0:
        remaining = 0

    others_total = sum(
        factors[i]["weight"] for i in range(n) if i != changed_idx
    )

    for i in range(n):
        if i == changed_idx:
            continue
        if others_total > 0:
            factors[i]["weight"] = round(factors[i]["weight"] / others_total * remaining)
        else:
            factors[i]["weight"] = round(remaining / (n - 1))

    # 修正舍入误差
    current_total = sum(f["weight"] for f in factors)
    diff = 100 - current_total
    if diff != 0:
        for i in range(n):
            if i != changed_idx:
                factors[i]["weight"] += diff
                break

    return factors


# ==================== 渲染组件 ====================

def _render_factor_cards(sector: str, factors: list[dict]) -> list[dict] | None:
    """渲染因子卡片 + 权重 slider，返回修改后的 factors 或 None（无变化）"""
    cols = st.columns(min(len(factors), 3))
    changed = False
    changed_idx = -1

    for i, factor in enumerate(factors):
        with cols[i % 3]:
            st.markdown(f"**{factor['name']}**")
            st.metric("权重", f"{factor['weight']}%")
            st.caption(factor.get("description", ""))
            st.caption(f"数据来源: {factor.get('data_source', '未知')}")

            key = f"slider_{sector}_{i}"
            new_w = st.slider(
                f"调整权重",
                0, 100, factor["weight"],
                key=key,
                label_visibility="collapsed",
            )
            if new_w != factor["weight"]:
                factor["weight"] = new_w
                changed = True
                changed_idx = i

    if changed and changed_idx >= 0:
        factors = _normalize_weights(factors, changed_idx)
        return factors
    return None


def _render_chat(sector: str, llm_config: LLMConfig, sector_data: dict) -> None:
    """渲染对话区"""
    msg_key = f"sf_{sector}_messages"

    # 从持久化恢复对话历史
    if msg_key not in st.session_state:
        st.session_state[msg_key] = sector_data.get("conversation", [])

    messages: list[dict] = st.session_state[msg_key]

    # 展示历史消息
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    pending_key = f"_pending_sf_chat_{sector}"
    user_input = st.chat_input(f"讨论「{sector}」的因子...", key=f"chat_{sector}")
    if user_input:
        messages.append({"role": "user", "content": user_input})
        st.session_state[pending_key] = True
        st.rerun()

    if not st.session_state.get(pending_key):
        return

    # 构建 LLM 请求
    current_factors = sector_data.get("factors", [])
    factors_json = json.dumps({"factors": current_factors}, ensure_ascii=False, indent=2)
    last_user_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
    llm_messages = build_factor_chat_messages(sector, factors_json, messages[:-1], last_user_msg)

    from quant.dashboard.bg_task import bg_llm_stream, clear_task
    task_id = f"sf_chat_{sector}"
    with st.chat_message("assistant"):
        full_reply = bg_llm_stream(task_id, llm_config, llm_messages, retry_key=f"retry_sf_chat_{sector}")

    if full_reply is None:
        return

    # 保存 assistant 回复
    messages.append({"role": "assistant", "content": full_reply})

    # 截断到最近 20 轮（40 条消息）
    if len(messages) > 40:
        messages = messages[-40:]
    st.session_state[msg_key] = messages

    # 检查是否包含因子 JSON
    parsed = extract_json_from_text(full_reply)
    if parsed:
        valid = validate_factors(parsed)
        if valid:
            sector_data["factors"] = valid
            sector_data["conversation"] = messages
            _save_sector(sector, sector_data)
            st.session_state.pop(pending_key, None)
            clear_task(task_id)
            st.rerun()

    # 持久化对话
    sector_data["conversation"] = messages
    _save_sector(sector, sector_data)
    st.session_state.pop(pending_key, None)
    clear_task(task_id)


def _render_factor_generation_detailed(
    sector: str, llm_config: LLMConfig, sector_data: dict,
) -> None:
    """4 阶段详细因子生成：提示词 → 流式输出 → JSON 解析 → 校验保存"""

    # ---- Stage 1: 准备提示词 ----
    with st.status("准备提示词...") as s1:
        prompt_msgs = build_factor_generation_prompt(sector)
        st.markdown("**系统提示词摘要:**")
        st.caption(FACTOR_GENERATION_SYSTEM[:200] + "...")
        st.markdown(f"**用户请求:** 为「{sector}」板块生成核心周期驱动因子")
        s1.update(label="提示词已准备", state="complete")

    # ---- Stage 2: AI 流式生成 ----
    from quant.dashboard.bg_task import bg_llm_stream, clear_task
    task_id = f"sf_gen_{sector}"
    full_reply = bg_llm_stream(task_id, llm_config, prompt_msgs, retry_key=f"retry_sf_gen_{sector}")
    if full_reply is None:
        return

    # ---- Stage 3: JSON 解析 ----
    with st.status("解析因子 JSON...") as s3:
        parsed = extract_json_from_text(full_reply)
        if parsed:
            st.markdown("**提取到的 JSON:**")
            st.code(json.dumps(parsed, ensure_ascii=False, indent=2), language="json")
            s3.update(label="JSON 解析成功", state="complete")
        else:
            st.error("未能从 AI 回复中提取出有效 JSON")
            with st.expander("查看 AI 原始回复"):
                st.code(full_reply)
            s3.update(label="JSON 解析失败", state="error")
            clear_task(task_id)
            st.session_state.pop(f"_sf_gen_pending_{sector}", None)
            return

    # ---- Stage 4: 校验并保存 ----
    with st.status("校验因子...") as s4:
        valid = validate_factors(parsed)
        if valid:
            total_weight = sum(f["weight"] for f in valid)
            st.markdown(f"**生成 {len(valid)} 个因子，权重总和 = {total_weight}%**")
            for f in valid:
                st.caption(f"- {f['name']} ({f['weight']}%): {f['description']}")
            sector_data["factors"] = valid
            sector_data["conversation"] = []
            _save_sector(sector, sector_data)
            s4.update(label="因子已保存", state="complete")
            clear_task(task_id)
            st.session_state.pop(f"_sf_gen_pending_{sector}", None)
            st.rerun()
        else:
            st.error("因子格式校验失败（需 2-5 个因子，权重总和 ≈ 100）")
            with st.expander("查看解析结果"):
                st.code(json.dumps(parsed, ensure_ascii=False, indent=2), language="json")
            s4.update(label="校验失败", state="error")
            clear_task(task_id)
            st.session_state.pop(f"_sf_gen_pending_{sector}", None)


def _render_sector_tab(sector: str, llm_config: LLMConfig) -> None:
    """渲染单个板块的完整 tab 内容"""
    from quant.dashboard.bg_task import has_task

    sector_data = _load_sector(sector)
    factors = sector_data.get("factors", [])

    gen_task_id = f"sf_gen_{sector}"
    gen_active = has_task(gen_task_id)

    # ---- 因子展示区 ----
    if factors:
        st.subheader("核心驱动因子")
        updated = _render_factor_cards(sector, factors)
        if updated is not None:
            sector_data["factors"] = updated

        if st.button("保存权重", key=f"save_{sector}", type="primary"):
            _save_sector(sector, sector_data)
            st.success("权重已保存")
    else:
        st.info(f"「{sector}」尚未生成因子，点击下方按钮让 AI 生成。")
        if st.button("AI 生成因子", key=f"gen_{sector}", type="primary", disabled=gen_active):
            st.session_state[f"_sf_gen_pending_{sector}"] = True
            st.rerun()

    # 后台因子生成轮询（按钮外！任务存在就继续轮询）
    if st.session_state.get(f"_sf_gen_pending_{sector}") or gen_active:
        _render_factor_generation_detailed(sector, llm_config, _load_sector(sector))
        return  # 生成中不显示对话区

    # ---- 对话区 ----
    st.markdown("---")
    st.subheader("AI 对话")
    st.caption("你可以通过对话调整因子数量、权重，或提问关于该板块的问题。")
    _render_chat(sector, llm_config, sector_data)


# ==================== 主页面 ====================

def render_sector_factors_page() -> None:
    """渲染「板块及因子」页面"""
    st.header("板块及因子")
    st.markdown("选择周期性板块，AI 自动生成核心驱动因子并通过对话调整。")

    # 检查 LLM 配置
    llm_config = get_llm_config()
    if llm_config is None:
        st.warning("请先在「设置」页面配置 LLM API Key。")
        return

    # ---- 板块选择 ----
    st.subheader("选择板块")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.multiselect(
            "从预设板块中选择（最多 5 个）",
            PRESET_SECTORS,
            max_selections=5,
            key="sector_select",
        )
    with col2:
        custom = st.text_input("自定义板块", key="custom_sector", placeholder="输入板块名")

    # 合并
    sectors = list(selected)
    if custom and custom.strip():
        custom_name = custom.strip()
        if custom_name not in sectors:
            if len(sectors) >= 5:
                st.warning("最多选择 5 个板块，请先移除一个预设板块。")
            else:
                sectors.append(custom_name)

    # ---- 已分析板块 ----
    all_saved = _load_all_factors()
    saved_with_factors = {k: v for k, v in all_saved.items() if v.get("factors")}

    if saved_with_factors:
        st.markdown("---")
        st.subheader(f"已分析板块（{len(saved_with_factors)}）")
        for name, data in saved_with_factors.items():
            factors = data["factors"]
            updated = data.get("updated_at", "")
            summary = " / ".join(f"{f['name']} {f['weight']}%" for f in factors)
            with st.expander(f"**{name}**　—　{summary}"):
                cols = st.columns(min(len(factors), 3))
                for i, f in enumerate(factors):
                    with cols[i % 3]:
                        st.markdown(f"**{f['name']}**")
                        st.metric("权重", f"{f['weight']}%")
                        st.caption(f.get("description", ""))
                        st.caption(f"数据来源: {f.get('data_source', '未知')}")
                if updated:
                    st.caption(f"最后更新: {updated}")

    # ---- 当前选中板块分析 ----
    if not sectors:
        if not saved_with_factors:
            st.info("请选择至少一个板块开始分析。")
        return

    st.markdown("---")

    # ---- 多板块 tab / 单板块直接渲染 ----
    if len(sectors) == 1:
        _render_sector_tab(sectors[0], llm_config)
    else:
        tabs = st.tabs(sectors)
        for tab, sector in zip(tabs, sectors):
            with tab:
                _render_sector_tab(sector, llm_config)
