"""审计页面 — AI 红队审计周期分析结果"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from quant.llm.client import SiliconFlowClient, LLMConfig, LLMError
from quant.llm.prompts import (
    build_audit_prompt, build_audit_chat_messages,
    build_factor_chat_messages,
    build_cycle_chat_messages,
    build_audit_param_prompt,
)
from quant.dashboard.llm_settings import get_audit_llm_config
from quant.dashboard.sector_factors import extract_json_from_text, validate_factors

# ==================== 路径常量 ====================

_ROOT = Path(__file__).parent.parent.parent
CYCLE_DATA_PATH = _ROOT / "data" / "cycle_analysis.json"
FACTORS_PATH = _ROOT / "data" / "sector_factors.json"
PROFILES_DIR = _ROOT / "data" / "stock_profiles"
RESEARCH_DIR = _ROOT / "data" / "research"
AUDIT_RESULTS_PATH = _ROOT / "data" / "audit_results.json"

# 审计类型定义
_AUDIT_TYPES = {
    "factors": "因子审计",
    "cycle": "周期审计",
    "stock": "个股审计",
    "full": "全面审计",
}


# ==================== 持久化 ====================

def _load_audit_results() -> dict:
    """加载所有审计结果，自动迁移旧格式"""
    if not AUDIT_RESULTS_PATH.exists():
        return {}
    try:
        with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}

    migrated = False
    _KNOWN_TYPES = {"factors", "cycle", "stock", "full"}
    for sector, data in list(raw.items()):
        # 旧格式检测：有旧 key 且没有新类型 key
        has_old_keys = "report" in data or "raw_response" in data
        has_new_keys = bool(set(data.keys()) & _KNOWN_TYPES)
        if has_old_keys and not has_new_keys:
            raw[sector] = {"full": data}
            migrated = True

    if migrated:
        AUDIT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    return raw


def _save_audit_result(sector: str, audit_type: str, audit_data: dict) -> None:
    """保存单个板块某类型的审计结果"""
    all_results = _load_audit_results()
    audit_data["audited_at"] = datetime.now().isoformat(timespec="seconds")
    if sector not in all_results:
        all_results[sector] = {}
    all_results[sector][audit_type] = audit_data
    AUDIT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


# ==================== 数据加载 ====================

def _load_sector_complete_data(sector: str) -> dict:
    """加载板块的所有分析数据（因子+周期+个股+归档）"""
    data = {"sector": sector, "found": False}

    # 1. 因子配置
    if FACTORS_PATH.exists():
        try:
            with open(FACTORS_PATH, "r", encoding="utf-8") as f:
                all_factors = json.load(f)
                if sector in all_factors:
                    data["factors_config"] = all_factors[sector]
                    data["found"] = True
        except Exception:
            pass

    # 2. 周期分析结果
    if CYCLE_DATA_PATH.exists():
        try:
            with open(CYCLE_DATA_PATH, "r", encoding="utf-8") as f:
                all_cycles = json.load(f)
                if sector in all_cycles:
                    data["cycle_analysis"] = all_cycles[sector]
                    data["found"] = True
        except Exception:
            pass

    # 3. 个股档案
    safe_name = sector.replace("/", "_").replace("\\", "_")
    profile_path = PROFILES_DIR / f"{safe_name}.json"
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data["stock_profiles"] = json.load(f)
        except Exception:
            pass

    # 4. 归档研究数据（找最新的）
    sector_research_dir = RESEARCH_DIR / re.sub(r'[\\/:*?"<>|]', "_", sector)
    if sector_research_dir.exists():
        subdirs = sorted(
            [d for d in sector_research_dir.iterdir() if d.is_dir()], reverse=True,
        )
        if subdirs:
            latest = subdirs[0]
            archive_files = list(latest.glob("*"))
            data["archive_path"] = str(latest)
            data["archive_file_count"] = len(archive_files)

    return data


# ==================== JSON 解析 ====================

def _parse_audit_json(text: str) -> dict | None:
    """从 LLM 回复中提取审计报告 JSON"""
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 回退：用平衡括号提取 JSON 对象
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if "risk_level" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


# ==================== 渲染组件 ====================

_RISK_COLORS = {"low": "green", "medium": "orange", "high": "red", "critical": "red"}
_RISK_LABELS = {"low": "低风险", "medium": "中等风险", "high": "高风险", "critical": "严重风险"}


def _render_audit_report(report: dict) -> None:
    """渲染结构化审计报告"""
    overall_risk = report.get("risk_level", "medium")
    color = _RISK_COLORS.get(overall_risk, "gray")
    label = _RISK_LABELS.get(overall_risk, "未知")

    # 总体风险
    col1, col2 = st.columns([2, 2])
    with col1:
        st.markdown(f"### 整体风险: :{color}[**{label}**]")
    with col2:
        confidence = report.get("confidence_score", 0)
        st.metric("分析可信度", f"{confidence}%")

    summary = report.get("summary", "")
    if summary:
        st.markdown(summary)

    st.markdown("---")

    # 分项审计结果
    st.subheader("分项审计结果")
    items = report.get("audit_items", [])
    for item in items:
        category = item.get("category", "未知")
        finding = item.get("finding", "")
        risk = item.get("risk", "medium")
        recommendation = item.get("recommendation", "")
        risk_color = _RISK_COLORS.get(risk, "gray")
        risk_label = _RISK_LABELS.get(risk, "未知")

        with st.expander(
            f":{risk_color}[**{category}**] — {risk_label}",
            expanded=(risk in ("high", "critical")),
        ):
            st.markdown(f"**发现:** {finding}")
            if recommendation:
                st.markdown(f"**建议:** {recommendation}")

    # 红旗信号
    red_flags = report.get("red_flags", [])
    if red_flags:
        st.markdown("---")
        st.subheader("重点关注")
        for flag in red_flags:
            st.error(flag)

    # 数据质量问题
    data_issues = report.get("data_quality_issues", [])
    if data_issues:
        st.markdown("---")
        st.subheader("数据质量问题")
        for issue in data_issues:
            st.warning(issue)

    # LLM 幻觉指标
    hallucinations = report.get("llm_hallucination_indicators", [])
    if hallucinations:
        st.markdown("---")
        st.subheader("LLM 幻觉嫌疑")
        for h in hallucinations:
            st.warning(h)

    # 不同解读
    alt = report.get("alternative_interpretations", [])
    if alt:
        st.markdown("---")
        st.subheader("其他可能的解读")
        for a in alt:
            st.info(a)


def _render_audit_chat(
    sector: str, audit_type: str, llm_config: LLMConfig, audit_data: dict,
) -> None:
    """审计对话区 — 追问审计 agent"""
    msg_key = f"audit_chat_{sector}_{audit_type}_messages"

    if msg_key not in st.session_state:
        st.session_state[msg_key] = audit_data.get("conversation", [])

    messages: list[dict] = st.session_state[msg_key]

    # 展示历史
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending_key = f"_pending_audit_chat_{sector}_{audit_type}"
    user_input = st.chat_input(
        f"向审计 agent 提问...", key=f"audit_chat_input_{sector}_{audit_type}",
    )
    if user_input:
        messages.append({"role": "user", "content": user_input})
        st.session_state[pending_key] = True
        st.rerun()

    if not st.session_state.get(pending_key):
        return

    # 构建对话请求
    last_user_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
    llm_messages = build_audit_chat_messages(sector, audit_data, messages[:-1], last_user_msg)

    from quant.dashboard.bg_task import bg_llm_stream, clear_task
    task_id = f"audit_chat_{sector}_{audit_type}"
    with st.chat_message("assistant"):
        full_reply = bg_llm_stream(task_id, llm_config, llm_messages, retry_key=f"retry_audit_chat_{sector}_{audit_type}")

    if full_reply is None:
        return

    messages.append({"role": "assistant", "content": full_reply})

    if len(messages) > 40:
        messages = messages[-40:]
    st.session_state[msg_key] = messages

    # 持久化对话
    audit_data["conversation"] = messages
    _save_audit_result(sector, audit_type, audit_data)
    st.session_state.pop(pending_key, None)
    clear_task(task_id)


# ==================== 审计闭环反馈 ====================

_TYPE_LABELS = {"factors": "因子配置", "cycle": "周期分析", "stock": "个股档案"}


def _build_feedback_message(audit_type: str, report: dict) -> str:
    """从审计报告中提取 high/critical 发现，格式化为修正建议消息"""
    type_label = _TYPE_LABELS.get(audit_type, audit_type)
    risk_level = _RISK_LABELS.get(report.get("risk_level", "medium"), "中等风险")
    confidence = report.get("confidence_score", 0)

    lines = [
        f"以下是独立审计 agent 对本板块的{type_label}审计结果"
        f"（风险等级: {risk_level}，可信度: {confidence}%）：",
        "",
        "【审计发现】",
    ]

    for i, item in enumerate(report.get("audit_items", []), 1):
        category = item.get("category", "未知")
        finding = item.get("finding", "")
        recommendation = item.get("recommendation", "")
        lines.append(f"{i}. {category}: {finding}")
        if recommendation:
            lines.append(f"   建议: {recommendation}")

    red_flags = report.get("red_flags", [])
    if red_flags:
        lines.append("")
        lines.append("【红旗信号】")
        for flag in red_flags:
            lines.append(f"- {flag}")

    lines.append("")
    lines.append(
        "请根据以上审计意见，评估当前分析是否需要修正。"
        "如果需要修正，请输出修正后的完整 JSON。"
        "如果认为审计意见不合理，请说明理由。"
    )
    return "\n".join(lines)


def _inject_and_respond(
    sector: str, audit_type: str, feedback_msg: str, llm_config: LLMConfig,
) -> tuple[str | None, str]:
    """核心闭环函数：将审计反馈注入到源系统并获取 LLM 回复。

    Returns:
        (reply_text | None, status_msg)
    """
    client = SiliconFlowClient(llm_config)

    if audit_type == "factors":
        return _inject_factors(sector, feedback_msg, client)
    elif audit_type == "cycle":
        return _inject_cycle(sector, feedback_msg, client)
    elif audit_type == "stock":
        return _inject_stock(sector, feedback_msg)
    return None, "未知审计类型"


def _inject_factors(
    sector: str, feedback_msg: str, client: SiliconFlowClient,
) -> tuple[str | None, str]:
    """因子审计闭环：注入反馈到 sector_factors.json"""
    if not FACTORS_PATH.exists():
        return None, "因子配置文件不存在"

    with open(FACTORS_PATH, "r", encoding="utf-8") as f:
        all_factors = json.load(f)

    sector_data = all_factors.get(sector)
    if not sector_data:
        return None, f"板块「{sector}」无因子配置"

    factors_json = json.dumps(sector_data.get("factors", []), ensure_ascii=False, indent=2)
    history = sector_data.get("conversation", [])

    messages = build_factor_chat_messages(sector, factors_json, history, feedback_msg)
    try:
        reply = client.chat(messages)
    except LLMError as e:
        return None, f"LLM 调用失败: {e}"

    # 追加对话到 conversation
    history.append({"role": "user", "content": feedback_msg})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 40:
        history = history[-40:]
    sector_data["conversation"] = history

    # 尝试从回复中提取修正后的因子 JSON
    status_msg = "AI 已评估审计建议"
    parsed = extract_json_from_text(reply)
    if parsed:
        valid = validate_factors(parsed)
        if valid:
            sector_data["factors"] = valid
            sector_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
            status_msg = "已自动更新因子配置"

    all_factors[sector] = sector_data
    with open(FACTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_factors, f, ensure_ascii=False, indent=2)

    return reply, status_msg


def _inject_cycle(
    sector: str, feedback_msg: str, client: SiliconFlowClient,
) -> tuple[str | None, str]:
    """周期审计闭环：注入反馈到 cycle_analysis.json"""
    if not CYCLE_DATA_PATH.exists():
        return None, "周期分析文件不存在"

    with open(CYCLE_DATA_PATH, "r", encoding="utf-8") as f:
        all_cycles = json.load(f)

    sector_data = all_cycles.get(sector)
    if not sector_data:
        return None, f"板块「{sector}」无周期分析数据"

    history = sector_data.get("conversation", [])

    messages = build_cycle_chat_messages(sector, sector_data, history, feedback_msg)
    try:
        reply = client.chat(messages)
    except LLMError as e:
        return None, f"LLM 调用失败: {e}"

    # 追加对话
    history.append({"role": "user", "content": feedback_msg})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 40:
        history = history[-40:]
    sector_data["conversation"] = history

    # 尝试从回复中提取修正后的周期分析 JSON
    status_msg = "AI 已评估审计建议"
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", reply, re.DOTALL)
    candidate = None
    if m:
        try:
            candidate = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    if candidate is None:
        start = reply.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(reply)):
                if reply[i] == "{":
                    depth += 1
                elif reply[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            candidate = json.loads(reply[start : i + 1])
                        except json.JSONDecodeError:
                            pass
                        break

    if candidate:
        if "overall" in candidate:
            # LLM 返回了完整结构 {"overall": {...}, ...}
            preserved_keys = {"news", "archive_path", "analyzed_at", "factors", "conversation"}
            for key in preserved_keys:
                if key in sector_data and key not in candidate:
                    candidate[key] = sector_data[key]
            sector_data.update(candidate)
            status_msg = "已自动更新周期分析"
        elif "cycle_position" in candidate:
            # LLM 返回了 overall 内部的字段 {"cycle_position": ..., "reversal_probability": ...}
            # 需要清理 reversal_probability 可能为字符串（如 "60%"）的情况
            rp = candidate.get("reversal_probability", 0)
            if isinstance(rp, str):
                rp = int("".join(c for c in rp if c.isdigit()) or "0")
                candidate["reversal_probability"] = rp
            if "overall" not in sector_data:
                sector_data["overall"] = {}
            sector_data["overall"].update(candidate)
            status_msg = "已自动更新周期分析"

    all_cycles[sector] = sector_data
    with open(CYCLE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_cycles, f, ensure_ascii=False, indent=2)

    return reply, status_msg


def _inject_stock(sector: str, feedback_msg: str = "") -> tuple[None, str]:
    """个股审计闭环：删旧分析 + 清缓存 + 注入审计上下文"""
    from quant.data.cache import DataCache

    safe_name = sector.replace("/", "_").replace("\\", "_")
    profile_path = PROFILES_DIR / f"{safe_name}.json"

    # 清除 PB 缓存
    cleared_count = 0
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            cache = DataCache()
            for s in old_data.get("stocks", []):
                code = s.get("code", "")
                if code:
                    cache.clear_pattern(f"pb_%{code}%")
                    cleared_count += 1
        except Exception:
            pass
        profile_path.unlink()

    st.session_state["pending_stock_reanalysis"] = sector
    st.session_state["stock_reanalysis_context"] = feedback_msg  # 保存审计意见供重分析参考
    return None, f"已清除旧分析（{cleared_count} 只股票缓存），切换到「个股档案」将自动重分析（含审计意见）"


def _render_feedback_action(
    sector: str, audit_type: str, report: dict, llm_config: LLMConfig,
) -> None:
    """在审计报告下方渲染反馈区域"""
    from quant.dashboard.bg_task import has_task

    st.markdown("#### 审计建议回传")

    if audit_type == "full":
        # 全面审计显示三个独立按钮
        _render_full_feedback_buttons(sector, report, llm_config)
        return

    # 单类型审计
    feedback_msg = _build_feedback_message(audit_type, report)

    feedback_msg = st.text_area(
        "修正建议（可编辑）",
        value=feedback_msg,
        height=200,
        key=f"feedback_msg_{sector}_{audit_type}",
    )

    btn_labels = {
        "factors": "发送到「因子配置」修正",
        "cycle": "发送到「周期分析」修正",
        "stock": "触发个股重新分析",
    }
    btn_label = btn_labels.get(audit_type, "发送修正")

    fb_task_id = f"audit_fb_{audit_type}_{sector}"
    fb_active = has_task(fb_task_id)
    fb_pending_key = f"_fb_pending_{audit_type}_{sector}"

    # 活跃任务轮询（在按钮之前，防止 rerun 后丢失）
    if fb_active or st.session_state.get(fb_pending_key):
        _execute_feedback(sector, audit_type, feedback_msg, llm_config)
        return

    if st.button(btn_label, type="primary", key=f"feedback_btn_{sector}_{audit_type}"):
        st.session_state[fb_pending_key] = True
        st.rerun()


def _render_full_feedback_buttons(
    sector: str, report: dict, llm_config: LLMConfig,
) -> None:
    """全面审计 — 显示三个独立反馈按钮"""
    from quant.dashboard.bg_task import has_task

    feedback_msg = _build_feedback_message("full", report)

    feedback_msg = st.text_area(
        "修正建议（可编辑）",
        value=feedback_msg,
        height=200,
        key=f"feedback_msg_{sector}_full",
    )

    col1, col2, col3 = st.columns(3)

    # 检查各系统数据是否存在
    has_factors = FACTORS_PATH.exists() and sector in (
        json.loads(FACTORS_PATH.read_text(encoding="utf-8")) if FACTORS_PATH.exists() else {}
    )
    has_cycle = CYCLE_DATA_PATH.exists() and sector in (
        json.loads(CYCLE_DATA_PATH.read_text(encoding="utf-8")) if CYCLE_DATA_PATH.exists() else {}
    )
    safe_name = sector.replace("/", "_").replace("\\", "_")
    has_stock = (PROFILES_DIR / f"{safe_name}.json").exists()

    # 各反馈按钮的后台任务状态
    fb_types = [("factors", has_factors, col1), ("cycle", has_cycle, col2), ("stock", has_stock, col3)]
    fb_btn_labels = {
        "factors": "发送到「因子配置」修正",
        "cycle": "发送到「周期分析」修正",
        "stock": "触发个股重新分析",
    }

    # 先检查是否有活跃的反馈任务（在按钮之前轮询）
    any_active = False
    for target_type, has_data, _ in fb_types:
        if not has_data:
            continue
        fb_task_id = f"audit_fb_{target_type}_{sector}"
        fb_active = has_task(fb_task_id)
        fb_pending_key = f"_fb_pending_{target_type}_{sector}"
        if fb_active or st.session_state.get(fb_pending_key):
            _execute_feedback(sector, target_type, feedback_msg, llm_config)
            any_active = True

    if any_active:
        return

    # 无活跃任务时渲染按钮
    for target_type, has_data, col in fb_types:
        with col:
            if has_data:
                fb_pending_key = f"_fb_pending_{target_type}_{sector}"
                if st.button(fb_btn_labels[target_type], key=f"full_fb_{target_type}_{sector}"):
                    st.session_state[fb_pending_key] = True
                    st.rerun()
            else:
                st.caption(f"无{_TYPE_LABELS.get(target_type, '')}数据")


def _show_feedback_result(reply: str | None, status_msg: str, target_type: str = "") -> None:
    """展示反馈注入结果"""
    if reply:
        st.markdown("**AI 评估结果:**")
        st.markdown(reply)
        if "已自动更新" in status_msg:
            st.success(status_msg)
        else:
            st.info(status_msg)
    else:
        if target_type == "stock":
            st.success(status_msg)
        else:
            st.warning(status_msg)


def _generate_param_adjustments(
    sector: str, audit_text: str, llm_config: LLMConfig,
) -> dict:
    """调用 LLM 根据审计意见生成结构化参数调整建议"""
    from quant.analysis.stock_cycle_analyzer import load_cycle_analysis, _select_weights

    cycle_data = load_cycle_analysis()
    sector_info = cycle_data.get(sector, {})
    overall = sector_info.get("overall", {})
    cycle_position = overall.get("cycle_position", "未知")
    current_weights = _select_weights(cycle_position)
    top_n = 10

    messages = build_audit_param_prompt(sector, cycle_position, current_weights, top_n, audit_text)
    try:
        client = SiliconFlowClient(llm_config)
        reply = client.chat(messages)
    except LLMError:
        return {}

    # 提取 JSON
    parsed = _parse_audit_json(reply)
    if not parsed:
        # 回退：尝试通用 JSON 提取
        m = re.search(r"```json\s*\n?(.*?)\n?\s*```", reply, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                return {}
        else:
            start = reply.find("{")
            if start != -1:
                depth = 0
                for i in range(start, len(reply)):
                    if reply[i] == "{":
                        depth += 1
                    elif reply[i] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                parsed = json.loads(reply[start : i + 1])
                            except json.JSONDecodeError:
                                pass
                            break
    if not parsed:
        return {}

    # 验证 weights 之和 = 1.0
    weights = parsed.get("weights")
    if weights and isinstance(weights, dict):
        total = sum(weights.values())
        if abs(total - 1.0) > 0.05:
            # 自动归一化
            for k in weights:
                weights[k] = round(weights[k] / total, 2)

    # 验证 top_n 范围
    tn = parsed.get("top_n")
    if tn is not None:
        parsed["top_n"] = max(5, min(20, int(tn)))

    return parsed


def _execute_feedback(
    sector: str, target_type: str, feedback_msg: str, llm_config: LLMConfig,
) -> None:
    """执行反馈注入并展示结果"""
    # stock 类型：同步执行（_inject_stock 只做文件删除+缓存清理，毫秒级）
    if target_type == "stock":
        reply, status_msg = _inject_and_respond(sector, target_type, feedback_msg, llm_config)

        # LLM 生成参数调整建议
        with st.status("AI 生成参数调整建议..."):
            param_adjustments = _generate_param_adjustments(sector, feedback_msg, llm_config)

        if param_adjustments:
            st.session_state["stock_reanalysis_params"] = param_adjustments

        st.success(status_msg)
        st.session_state["_nav_redirect"] = "📑 个股档案"
        st.rerun()
        return

    # factors/cycle：走 bg_run（需要 LLM）
    from quant.dashboard.bg_task import bg_run, clear_task

    task_id = f"audit_fb_{target_type}_{sector}"

    result = bg_run(task_id, _inject_and_respond, sector, target_type, feedback_msg, llm_config)
    if result is None:
        return

    reply, status_msg = result
    clear_task(task_id)
    st.session_state.pop(f"_fb_pending_{target_type}_{sector}", None)
    _show_feedback_result(reply, status_msg, target_type)


# ==================== 审计执行 ====================

def _run_audit(sector: str, llm_config: LLMConfig, audit_type: str) -> None:
    """执行审计流程"""
    type_label = _AUDIT_TYPES.get(audit_type, audit_type)

    # Stage 1: 加载数据
    with st.status("加载板块数据...") as s1:
        data = _load_sector_complete_data(sector)

        if not data.get("found"):
            st.error(f"板块「{sector}」无任何分析数据")
            st.session_state.pop(f"_audit_pending_{audit_type}_{sector}", None)
            return

        data_summary = []
        if "factors_config" in data:
            factors = data["factors_config"].get("factors", [])
            data_summary.append(f"因子配置: {len(factors)} 个因子")
        if "cycle_analysis" in data:
            overall = data["cycle_analysis"].get("overall", {})
            data_summary.append(f"周期判断: {overall.get('cycle_position', 'N/A')}")
        if "stock_profiles" in data:
            stocks = data["stock_profiles"].get("stocks", [])
            data_summary.append(f"个股分析: {len(stocks)} 只")
        if "archive_path" in data:
            data_summary.append(f"归档: {data['archive_file_count']} 文件")

        for item in data_summary:
            st.caption(item)
        s1.update(label="数据加载完成", state="complete")

    # Stage 2: 构建审计提示词
    with st.status(f"准备{type_label}提示词...") as s2:
        messages = build_audit_prompt(sector, data, audit_type)
        # 估算 token 数
        total_chars = sum(len(m["content"]) for m in messages)
        st.markdown(f"**审计范围:** {len(messages)} 条消息，约 {total_chars} 字")
        st.caption(f"类型: {type_label}")
        s2.update(label="提示词已准备", state="complete")

    # Stage 3: AI 审计分析
    from quant.dashboard.bg_task import bg_llm_stream, clear_task
    audit_task_id = f"audit_{audit_type}_{sector}"
    full_reply = bg_llm_stream(audit_task_id, llm_config, messages, retry_key=f"retry_audit_{audit_type}_{sector}")
    if full_reply is None:
        return
    if not full_reply:
        st.error("LLM 返回空响应")
        clear_task(audit_task_id)
        st.session_state.pop(f"_audit_pending_{audit_type}_{sector}", None)
        return
    clear_task(audit_task_id)

    # Stage 4: 解析并保存
    _finalize_audit_result(sector, audit_type, full_reply, data)


def _finalize_audit_result(
    sector: str, audit_type: str, full_reply: str, preview_data: dict,
) -> None:
    """解析 LLM 审计回复，保存结果并刷新页面"""
    should_rerun = False
    with st.status("解析审计报告...") as s4:
        parsed = _parse_audit_json(full_reply)

        if parsed:
            st.markdown("**报告结构:**")
            st.caption(f"- 整体风险: {parsed.get('risk_level', 'N/A')}")
            st.caption(f"- 可信度: {parsed.get('confidence_score', 'N/A')}%")
            st.caption(f"- 审计项: {len(parsed.get('audit_items', []))} 项")
            st.caption(f"- 红旗信号: {len(parsed.get('red_flags', []))} 个")

            audit_result = {
                "report": parsed,
                "raw_response": full_reply,
                "conversation": [],
                "sector_data_snapshot": {
                    "cycle_position": preview_data.get("cycle_analysis", {}).get("overall", {}).get("cycle_position"),
                    "factors_count": len(preview_data.get("factors_config", {}).get("factors", [])),
                    "stocks_count": len(preview_data.get("stock_profiles", {}).get("stocks", [])),
                },
            }
            _save_audit_result(sector, audit_type, audit_result)
            s4.update(label="报告已保存", state="complete")
            should_rerun = True
        else:
            st.warning("未能解析出结构化报告，保存原始回复")
            audit_result = {
                "report": None,
                "raw_response": full_reply,
                "conversation": [],
            }
            _save_audit_result(sector, audit_type, audit_result)
            s4.update(label="已保存原始回复", state="complete")

    st.session_state.pop(f"_audit_pending_{audit_type}_{sector}", None)
    if should_rerun:
        st.rerun()


# ==================== Tab 内容渲染 ====================

def _render_tab_content(
    sector: str,
    audit_type: str,
    preview_data: dict,
    llm_config: LLMConfig,
    all_results: dict,
) -> None:
    """渲染单个审计 Tab 的内容"""
    type_label = _AUDIT_TYPES.get(audit_type, audit_type)

    # 数据完整度预览（按类型显示不同指标）
    _render_data_preview(audit_type, preview_data)

    # 检查是否满足审计条件
    can_audit = _check_audit_ready(audit_type, preview_data)

    # 已有审计结果
    sector_results = all_results.get(sector, {})
    existing = sector_results.get(audit_type)

    if existing:
        report = existing.get("report")
        audited_at = existing.get("audited_at", "")
        time_str = audited_at[:10] if audited_at else ""

        if report:
            risk = report.get("risk_level", "medium")
            risk_label = _RISK_LABELS.get(risk, "未知")
            confidence = report.get("confidence_score", 0)
            exp_label = f"上次{type_label}结果 — 风险: {risk_label} · 可信度 {confidence}% · {time_str}"
        else:
            exp_label = f"上次{type_label}结果 — (原始文本) · {time_str}"

        with st.expander(exp_label, expanded=True):
            if report:
                _render_audit_report(report)
            else:
                st.markdown("**原始审计回复:**")
                raw = existing.get("raw_response", "")
                st.markdown(raw)

            # 反馈区域（仅 medium/high/critical 时显示）
            if report and report.get("risk_level") in ("medium", "high", "critical"):
                st.markdown("---")
                _render_feedback_action(sector, audit_type, report, llm_config)

            st.markdown("---")
            st.markdown("##### 对话追问")
            _render_audit_chat(sector, audit_type, llm_config, existing)

    # 审计任务轮询（在按钮之前检查，防止 rerun 后任务丢失）
    if can_audit:
        from quant.dashboard.bg_task import has_task
        audit_task_id = f"audit_{audit_type}_{sector}"
        audit_active = has_task(audit_task_id)
        audit_pending_key = f"_audit_pending_{audit_type}_{sector}"

        # 活跃任务轮询（优先于按钮渲染）
        if audit_active or st.session_state.get(audit_pending_key):
            _run_audit(sector, llm_config, audit_type)
            return  # 运行中不渲染按钮

        # 开始审计按钮
        btn_label = f"重新{type_label}" if existing else f"开始{type_label}"
        if st.button(btn_label, type="primary", key=f"start_{audit_type}_{sector}"):
            st.session_state[audit_pending_key] = True
            st.rerun()
    else:
        st.info(f"数据不足，无法执行{type_label}。请先完成相关分析步骤。")


def _render_data_preview(audit_type: str, preview_data: dict) -> None:
    """按审计类型显示数据完整度指标"""
    if audit_type == "factors":
        col1, col2 = st.columns(2)
        has_factors = "factors_config" in preview_data
        factors_count = len(preview_data.get("factors_config", {}).get("factors", [])) if has_factors else 0
        col1.metric("因子数量", factors_count if has_factors else "无")
        updated = preview_data.get("factors_config", {}).get("updated_at", "无")
        col2.metric("更新时间", updated[:10] if updated and updated != "无" else "无")

    elif audit_type == "cycle":
        col1, col2, col3 = st.columns(3)
        has_cycle = "cycle_analysis" in preview_data
        overall = preview_data.get("cycle_analysis", {}).get("overall", {})
        col1.metric("周期位置", overall.get("cycle_position", "无") if has_cycle else "无")
        col2.metric("反转概率", f"{overall.get('reversal_probability', 0)}%" if has_cycle else "无")
        has_factors = "factors_config" in preview_data
        factors_count = len(preview_data.get("factors_config", {}).get("factors", [])) if has_factors else 0
        col3.metric("因子数量", factors_count if has_factors else "无")

    elif audit_type == "stock":
        col1, col2 = st.columns(2)
        has_stocks = "stock_profiles" in preview_data
        stock_count = len(preview_data.get("stock_profiles", {}).get("stocks", [])) if has_stocks else 0
        col1.metric("个股数量", stock_count if has_stocks else "无")
        # 检查是否有相关性数据
        has_corr = False
        if has_stocks:
            for s in preview_data.get("stock_profiles", {}).get("stocks", []):
                if s.get("correlation"):
                    has_corr = True
                    break
        col2.metric("相关性数据", "有" if has_corr else "无")

    else:  # full
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("因子配置", "有" if "factors_config" in preview_data else "无")
        col2.metric("周期分析", "有" if "cycle_analysis" in preview_data else "无")
        col3.metric("个股档案", "有" if "stock_profiles" in preview_data else "无")
        col4.metric("归档数据", f"{preview_data.get('archive_file_count', 0)} 文件")


def _check_audit_ready(audit_type: str, preview_data: dict) -> bool:
    """检查数据是否满足开始审计的条件"""
    if audit_type == "factors":
        return "factors_config" in preview_data
    elif audit_type == "cycle":
        return "cycle_analysis" in preview_data
    elif audit_type == "stock":
        return "stock_profiles" in preview_data
    else:  # full
        return preview_data.get("found", False)


# ==================== 主页面 ====================

def render_audit_page() -> None:
    """渲染审计页面"""
    st.header("审计 — AI 红队质疑")
    st.markdown("独立审计 agent 深度质疑分析结论，识别数据风险与逻辑漏洞。")

    # 检查审计 LLM 配置
    audit_config = get_audit_llm_config()
    if audit_config is None:
        st.warning("请先在「设置」页面配置 LLM API Key（主 LLM 或独立审计模型均可）。")
        st.info("建议使用 DeepSeek-V3 或其他强推理模型作为审计 agent。")
        return

    # 加载已分析板块
    sectors = []
    if CYCLE_DATA_PATH.exists():
        try:
            with open(CYCLE_DATA_PATH, "r", encoding="utf-8") as f:
                cycle_data = json.load(f)
                sectors = list(cycle_data.keys())
        except Exception:
            pass

    # 也检查只有因子配置的板块
    if FACTORS_PATH.exists():
        try:
            with open(FACTORS_PATH, "r", encoding="utf-8") as f:
                factor_sectors = list(json.load(f).keys())
                for s in factor_sectors:
                    if s not in sectors:
                        sectors.append(s)
        except Exception:
            pass

    if not sectors:
        st.info("暂无已分析的板块。请先到「因子配置」或「周期分析」页面分析至少一个板块。")
        return

    # 板块选择
    selected_sector = st.selectbox(
        "选择要审计的板块",
        sectors,
        help="从已完成分析的板块中选择",
    )

    if not selected_sector:
        return

    # 加载数据
    preview_data = _load_sector_complete_data(selected_sector)
    all_results = _load_audit_results()

    # Tab 布局
    tab_factors, tab_cycle, tab_stock, tab_full = st.tabs([
        "因子审计", "周期审计", "个股审计", "全面审计",
    ])

    with tab_factors:
        _render_tab_content(selected_sector, "factors", preview_data, audit_config, all_results)

    with tab_cycle:
        _render_tab_content(selected_sector, "cycle", preview_data, audit_config, all_results)

    with tab_stock:
        _render_tab_content(selected_sector, "stock", preview_data, audit_config, all_results)

    with tab_full:
        _render_tab_content(selected_sector, "full", preview_data, audit_config, all_results)
