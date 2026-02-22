"""周期分析页面 — AI 驱动的真实数据周期研判"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from quant.llm.client import SiliconFlowClient, LLMConfig, LLMError
from quant.llm.prompts import (
    build_cycle_analysis_prompt,
    build_cycle_chat_messages,
)
from quant.dashboard.llm_settings import (
    get_llm_config,
    get_tavily_key,
    get_jina_key,
    get_apify_key,
)
from quant.research.data_fetcher import DataFetcher, _dedupe_news as dedupe_news
from quant.research.tavily_client import TavilyClient
from quant.research.jina_reader import JinaReader
from quant.research.apify_client import ApifyClient

# ==================== 路径常量 ====================

_ROOT = Path(__file__).parent.parent.parent
CYCLE_DATA_PATH = _ROOT / "data" / "cycle_analysis.json"
FACTORS_PATH = _ROOT / "data" / "sector_factors.json"
RESEARCH_DIR = _ROOT / "data" / "research"


# ==================== 持久化 ====================

def _load_all_cycles() -> dict:
    if CYCLE_DATA_PATH.exists():
        try:
            with open(CYCLE_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_all_cycles(data: dict) -> None:
    CYCLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CYCLE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_cycle(sector: str, cycle_data: dict) -> None:
    all_data = _load_all_cycles()
    all_data[sector] = cycle_data
    _save_all_cycles(all_data)


def _load_all_factors() -> dict:
    """从 sector_factors.json 加载所有板块因子"""
    if FACTORS_PATH.exists():
        try:
            with open(FACTORS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _safe_dirname(name: str) -> str:
    """板块名转安全目录名"""
    return re.sub(r'[\\/:*?"<>|]', "_", name)


# ==================== JSON 提取 ====================

def extract_json_from_text(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON"""
    # 1) ```json ... ```
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2) 裸 JSON
    m = re.search(r"\{[\s\S]*\"overall\"[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ==================== 图表 ====================

def _safe_float(val) -> float | None:
    """安全转 float，非数字返回 None"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # 去掉常见干扰字符，只保留数字和小数点
        cleaned = re.sub(r"[^\d.\-]", "", val)
        if cleaned:
            try:
                return float(cleaned)
            except ValueError:
                pass
    return None


def _quarter_to_datetime(q_str: str) -> datetime | None:
    """'2022-Q1' → datetime(2022, 2, 15)，支持 YYYY-QN 格式"""
    try:
        parts = q_str.split("-")
        year = int(parts[0])
        q = int(parts[1][1]) if len(parts) > 1 and parts[1].startswith("Q") else 1
        month = {1: 2, 2: 5, 3: 8, 4: 11}.get(q, 2)
        return datetime(year, month, 15)
    except (ValueError, IndexError, KeyError):
        return None


def _build_factor_chart(factor: dict) -> go.Figure:
    """构建因子周期图表 — 关键节点连线，X轴等比时间"""
    fig = go.Figure()
    all_points = []  # (date_str, datetime, value, type)

    for cycle in factor.get("cycle_data", []):
        peak = cycle.get("peak", {})
        trough = cycle.get("trough", {})
        peak_val = _safe_float(peak.get("value"))
        trough_val = _safe_float(trough.get("value"))
        if peak.get("date") and peak_val is not None:
            all_points.append((peak["date"], peak_val, "peak"))
        if trough.get("date") and trough_val is not None:
            all_points.append((trough["date"], trough_val, "trough"))

        # 提取 key_points（中间关键节点）
        for kp in cycle.get("key_points", []):
            kp_val = _safe_float(kp.get("value"))
            if kp.get("date") and kp_val is not None:
                all_points.append((kp["date"], kp_val, "normal"))

    if not all_points:
        return fig

    # 去重（同日期同值只保留优先级最高的类型：peak > trough > normal）
    seen = {}
    priority = {"peak": 0, "trough": 1, "normal": 2}
    for date_str, val, ptype in all_points:
        key = date_str
        if key not in seen or priority.get(ptype, 2) < priority.get(seen[key][2], 2):
            seen[key] = (date_str, val, ptype)
    all_points = list(seen.values())

    # 按日期排序
    all_points.sort(key=lambda x: x[0])

    # 转换为 datetime 用于等比 X 轴
    dt_points = []
    for date_str, val, ptype in all_points:
        dt = _quarter_to_datetime(date_str)
        if dt:
            dt_points.append((dt, date_str, val, ptype))

    if not dt_points:
        return fig

    dates_dt = [p[0] for p in dt_points]
    date_labels = [p[1] for p in dt_points]
    values = [p[2] for p in dt_points]
    types = [p[3] for p in dt_points]

    # 主折线
    fig.add_trace(go.Scatter(
        x=dates_dt, y=values,
        mode="lines+markers",
        name="周期走势",
        line=dict(color="#5B8DEF", width=2),
        marker=dict(size=6),
        text=date_labels,
        hovertemplate="%{text}<br>%{y}<extra></extra>",
    ))

    # 高点标注（红色倒三角）
    peaks = [(dt, lbl, v) for dt, lbl, v, t in dt_points if t == "peak"]
    if peaks:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in peaks],
            y=[p[2] for p in peaks],
            mode="markers+text",
            name="高点",
            marker=dict(color="red", size=12, symbol="triangle-down"),
            text=[f"{v}" for _, _, v in peaks],
            textposition="top center",
            textfont=dict(size=11, color="red"),
        ))

    # 低点标注（绿色正三角）
    troughs = [(dt, lbl, v) for dt, lbl, v, t in dt_points if t == "trough"]
    if troughs:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in troughs],
            y=[p[2] for p in troughs],
            mode="markers+text",
            name="低点",
            marker=dict(color="green", size=12, symbol="triangle-up"),
            text=[f"{v}" for _, _, v in troughs],
            textposition="bottom center",
            textfont=dict(size=11, color="green"),
        ))

    unit = factor.get("unit", "")
    fig.update_layout(
        title=f"{factor['name']}" + (f" ({unit})" if unit else ""),
        yaxis_title=unit,
        hovermode="x unified",
        showlegend=False,
        height=350,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    fig.update_xaxes(dtick="M12", tickformat="%Y")
    return fig


# ==================== 渲染组件 ====================

def _render_cycle_result(sector: str, data: dict) -> None:
    """渲染分析结果：综合判断 + 因子图表 + 行业资讯"""
    overall = data.get("overall", {})

    # 综合判断面板
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {overall.get('cycle_position', '未知')}")
        st.caption("周期位置")
    with col2:
        prob = overall.get("reversal_probability", 0)
        timeframe = overall.get("probability_timeframe", "12个月")
        rationale = overall.get("probability_rationale", "")
        st.metric(f"反转概率({timeframe})", f"{prob}%")
        if rationale:
            st.caption(rationale)
    with col3:
        st.markdown("**关键信号**")
        for s in overall.get("key_signals", []):
            st.markdown(f"- {s}")

    summary = overall.get("summary", "")
    if summary:
        st.markdown(summary)

    # 每个因子的图表
    for factor in data.get("factors", []):
        st.markdown(f"#### {factor['name']}")

        if factor.get("cycle_data"):
            fig = _build_factor_chart(factor)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"「{factor['name']}」数据不足，未能生成周期图表。")

        # 因子小结
        cols = st.columns([2, 2, 1])
        with cols[0]:
            avg = factor.get("avg_cycle_length_months")
            if avg:
                st.caption(f"平均周期长度: {avg} 个月")
            current_pos = factor.get("current_position", "未知")
            current_val = factor.get("current_value", "")
            if current_val:
                st.caption(f"当前值: {current_val} | 位置: {current_pos}")
            else:
                st.caption(f"当前位置: {current_pos}")
        with cols[1]:
            confidence = factor.get("data_confidence", "unknown")
            confidence_label = {"high": "高", "medium": "中", "low": "低"}.get(
                confidence, "未知"
            )
            source_url = factor.get("data_source_url", "")
            st.caption(f"数据可信度: {confidence_label}")
            if source_url:
                short_url = source_url[:50] + ("..." if len(source_url) > 50 else "")
                st.caption(f"[{short_url}]({source_url})")
        with cols[2]:
            archive = data.get("archive_path", "")
            if archive:
                st.caption(f"原始数据已归档")

        analysis = factor.get("analysis", "")
        if analysis:
            st.markdown(analysis)

    # 行业资讯
    news = data.get("news", [])
    if news:
        st.markdown("#### 行业资讯")
        for item in news[:8]:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("snippet", "")[:100]
            if title and url:
                st.markdown(f"- [{title}]({url})" + (f" — {snippet}" if snippet else ""))


def _render_cycle_chat(
    sector: str, llm_config: LLMConfig, cycle_data: dict,
) -> None:
    """对话区：讨论/调整周期判断"""
    msg_key = f"cycle_chat_{sector}_messages"

    if msg_key not in st.session_state:
        st.session_state[msg_key] = cycle_data.get("conversation", [])

    messages: list[dict] = st.session_state[msg_key]

    # 展示历史
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending_key = f"_pending_cycle_chat_{sector}"
    user_input = st.chat_input(
        f"讨论「{sector}」的周期...", key=f"cycle_chat_input_{sector}",
    )
    if user_input:
        messages.append({"role": "user", "content": user_input})
        st.session_state[pending_key] = True
        st.rerun()

    if not st.session_state.get(pending_key):
        return

    # 构建 LLM 请求
    last_user_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
    llm_messages = build_cycle_chat_messages(sector, cycle_data, messages[:-1], last_user_msg)

    llm_cfg = LLMConfig(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        model=llm_config.model,
        max_tokens=4096,
    )

    from quant.dashboard.bg_task import bg_llm_stream, clear_task
    task_id = f"cycle_chat_{sector}"
    with st.chat_message("assistant"):
        full_reply = bg_llm_stream(task_id, llm_cfg, llm_messages, retry_key=f"retry_cycle_chat_{sector}")

    if full_reply is None:
        return

    messages.append({"role": "assistant", "content": full_reply})

    if len(messages) > 40:
        messages = messages[-40:]
    st.session_state[msg_key] = messages

    # 检查是否包含更新的 JSON
    parsed = extract_json_from_text(full_reply)
    if parsed and "overall" in parsed:
        parsed["conversation"] = messages
        parsed["analyzed_at"] = cycle_data.get("analyzed_at", "")
        parsed["archive_path"] = cycle_data.get("archive_path", "")
        parsed["news"] = cycle_data.get("news", [])
        _save_cycle(sector, parsed)
        st.session_state.pop(pending_key, None)
        clear_task(task_id)
        st.rerun()

    # 持久化对话
    cycle_data["conversation"] = messages
    _save_cycle(sector, cycle_data)
    st.session_state.pop(pending_key, None)
    clear_task(task_id)


# ==================== 分步渲染 ====================

def _render_step(step: str, status: str, detail) -> None:
    """渲染数据获取的单步结果（LemonAI 风格）"""

    if step == "guidance" and status == "start":
        st.caption("🤖 询问 AI 数据去哪找...")

    elif step == "guidance" and status == "done":
        with st.expander("🤖 AI 数据源指导", expanded=False):
            if detail.get("search_queries"):
                st.markdown("**搜索关键词:**")
                for q in detail["search_queries"]:
                    st.markdown(f"- `{q}`")
            if detail.get("suggested_urls"):
                st.markdown("**建议数据源:**")
                for url in detail["suggested_urls"]:
                    st.markdown(f"- {url}")
            if detail.get("akshare_api"):
                st.markdown(f"**AKShare 接口:** `{detail['akshare_api']}`")

    elif step == "akshare" and status == "start":
        st.caption(f"📡 尝试 AKShare: `{detail.get('api', '')}`...")

    elif step == "akshare" and status == "done":
        n = detail.get("records", 0)
        st.markdown(f"📡 AKShare: ✅ 获取到 {n} 条记录")

    elif step == "akshare" and status == "fail":
        st.markdown("📡 AKShare: ❌ 未找到对应数据")

    elif step == "akshare" and status == "skip":
        pass  # 静默跳过

    elif step == "tavily" and status == "start":
        queries = detail.get("queries", []) if isinstance(detail, dict) else []
        st.caption(f"🔍 搜索中... ({len(queries)} 组关键词)")

    elif step == "tavily" and status == "done":
        results = detail if isinstance(detail, list) else []
        st.markdown(f"🔍 搜索结果: 找到 **{len(results)}** 条")
        if results:
            with st.expander(f"查看搜索结果 ({len(results)} 条)", expanded=False):
                for j, r in enumerate(results[:8]):
                    title = r.get("title", "无标题")
                    url = r.get("url", "")
                    snippet = (r.get("content", "") or "")[:120]
                    st.markdown(f"**[{j+1}]** [{title}]({url})")
                    if snippet:
                        st.caption(snippet)

    elif step == "tavily" and status == "fail":
        st.markdown("🔍 搜索结果: ❌ 未找到相关数据")

    elif step == "jina" and status == "start":
        url = (detail.get("url", "") if isinstance(detail, dict) else "")[:50]
        st.caption(f"📄 网页抓取: {url}...")

    elif step == "jina" and status == "done":
        chars = detail.get("chars", 0) if isinstance(detail, dict) else 0
        url = (detail.get("url", "") if isinstance(detail, dict) else "")[:50]
        st.markdown(f"📄 网页抓取: ✅ 成功 ({chars} 字) — {url}...")

    elif step == "jina" and status == "fail":
        url = (detail.get("url", "") if isinstance(detail, dict) else "")[:50]
        st.markdown(f"📄 网页抓取: ❌ 失败 — {url}")

    elif step == "apify" and status == "start":
        url = (detail.get("url", "") if isinstance(detail, dict) else "")[:50]
        st.caption(f"🕷️ Apify 爬虫: {url}...")

    elif step == "apify" and status == "done":
        chars = detail.get("chars", 0) if isinstance(detail, dict) else 0
        st.markdown(f"🕷️ Apify 爬虫: ✅ 成功 ({chars} 字)")

    elif step == "apify" and status == "fail":
        st.markdown("🕷️ Apify 爬虫: ❌ 失败")


# ==================== 后台分析流程 ====================

def _bg_analysis_worker(
    sector: str,
    factors: list[dict],
    llm_config: LLMConfig,
    progress: dict,
    cancel_event: threading.Event,
) -> None:
    """后台线程执行分析 — 禁止任何 st.* 调用"""
    try:
        tavily_key = get_tavily_key()
        jina_key = get_jina_key()
        apify_key = get_apify_key()

        top_factors = sorted(
            factors, key=lambda f: f.get("weight", 0), reverse=True,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = RESEARCH_DIR / _safe_dirname(sector) / timestamp

        fetcher = DataFetcher(
            llm_client=SiliconFlowClient(llm_config),
            tavily=TavilyClient(tavily_key) if tavily_key else None,
            jina=JinaReader(jina_key) if jina_key else None,
            apify=ApifyClient(apify_key) if apify_key else None,
        )

        factor_data_map = {}
        all_news = []

        for i, f in enumerate(top_factors):
            if cancel_event.is_set():
                progress["status"] = "cancelled"
                return

            progress["current_factor"] = f["name"]
            progress["factor_index"] = i + 1
            progress["factor_total"] = len(top_factors)
            progress["current_step"] = "fetch"
            progress["log"].append(
                f"({i+1}/{len(top_factors)}) 开始获取「{f['name']}」数据..."
            )

            def on_step(step, status, detail, _name=f["name"]):
                if step == "guidance" and status == "done":
                    progress["log"].append(f"  AI 数据源指导完成")
                elif step == "tavily" and status == "done":
                    n = len(detail) if isinstance(detail, list) else 0
                    progress["log"].append(f"  搜索完成: {n} 条结果")
                elif step == "jina" and status == "done":
                    chars = detail.get("chars", 0) if isinstance(detail, dict) else 0
                    progress["log"].append(f"  网页抓取: {chars} 字")
                elif step == "akshare" and status == "done":
                    n = detail.get("records", 0)
                    progress["log"].append(f"  AKShare: {n} 条记录")

            result = fetcher.fetch_factor_data(sector, f, progress_callback=on_step)
            factor_data_map[f["name"]] = result
            all_news.extend(result.news)

            # 记录因子详情供 UI 展示
            progress["factor_details"].append({
                "name": f["name"],
                "weight": f.get("weight", 0),
                "status": "completed" if result.found else "failed",
                "source": result.source,
                "guidance": result.guidance,
                "search_results": result.search_results_raw[:5],
                "fetch_log": result.fetch_log,
            })

            if result.found:
                progress["log"].append(
                    f"「{f['name']}」数据获取完成 — {result.source}"
                )
            else:
                progress["log"].append(f"「{f['name']}」未找到数据")

        if cancel_event.is_set():
            progress["status"] = "cancelled"
            return

        fetcher.save_archive(archive_dir)

        has_data = any(r.found for r in factor_data_map.values())
        if not has_data:
            progress["status"] = "failed"
            progress["error"] = "所有因子均未找到可用数据，无法进行周期分析。"
            return

        # LLM 分析
        progress["current_step"] = "llm"
        progress["log"].append("AI 正在分析周期...")
        messages = build_cycle_analysis_prompt(sector, top_factors, factor_data_map)

        client = SiliconFlowClient(LLMConfig(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
            max_tokens=4096,
        ))

        full_reply = client.chat(messages)

        if cancel_event.is_set():
            progress["status"] = "cancelled"
            return

        # 保存 LLM 原文
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            (archive_dir / "llm_response.txt").write_text(full_reply, encoding="utf-8")
        except Exception:
            pass

        parsed = extract_json_from_text(full_reply)
        if parsed and "overall" in parsed:
            parsed["news"] = dedupe_news(all_news)
            parsed["conversation"] = []
            parsed["analyzed_at"] = datetime.now().isoformat(timespec="seconds")
            parsed["archive_path"] = str(archive_dir)
            _save_cycle(sector, parsed)
            progress["result"] = parsed
            progress["status"] = "completed"
            progress["log"].append("分析完成，结果已保存。")
        else:
            progress["status"] = "failed"
            progress["error"] = "AI 未返回有效的分析结果"
            progress["llm_tail"] = full_reply[-500:] if full_reply else ""
            progress["log"].append("AI 未返回有效结果")

    except Exception as e:
        progress["status"] = "failed"
        progress["error"] = str(e)
        progress["log"].append(f"分析出错: {e}")


def _start_bg_analysis(
    sector: str, factors: list[dict], llm_config: LLMConfig,
) -> None:
    """启动后台分析线程"""
    cancel_event = threading.Event()
    progress = {
        "sector": sector,
        "status": "running",
        "current_step": "init",
        "current_factor": "",
        "factor_index": 0,
        "factor_total": 0,
        "log": [],
        "factor_details": [],
        "result": None,
        "error": None,
        "llm_tail": "",
        "started_at": datetime.now().strftime("%H:%M:%S"),
    }

    thread = threading.Thread(
        target=_bg_analysis_worker,
        args=(sector, factors, llm_config, progress, cancel_event),
        daemon=True,
    )

    st.session_state["bg_analysis"] = {
        "thread": thread,
        "cancel_event": cancel_event,
        "progress": progress,
    }

    thread.start()


def _render_bg_status() -> bool:
    """渲染后台分析状态，返回是否正在运行"""
    bg = st.session_state.get("bg_analysis")
    if not bg:
        return False

    progress = bg["progress"]
    status = progress["status"]
    sector = progress["sector"]

    if status == "running":
        with st.container(border=True):
            # 标题行 + 终止按钮
            col_title, col_btn = st.columns([4, 1])
            with col_title:
                if progress["current_step"] == "llm":
                    step_text = "AI 正在分析周期..."
                elif progress["factor_total"] > 0:
                    step_text = (
                        f"获取数据 ({progress['factor_index']}/{progress['factor_total']})"
                        f" — {progress.get('current_factor', '')}"
                    )
                else:
                    step_text = "初始化..."
                st.markdown(f"**正在分析「{sector}」** — {step_text}")
            with col_btn:
                if st.button("终止分析", type="secondary", key="stop_analysis"):
                    bg["cancel_event"].set()
                    progress["status"] = "cancelling"
                    st.rerun()

            # 进度日志（最近 10 条）
            for msg in progress["log"][-10:]:
                st.caption(msg)

            # 因子获取详情卡片
            factor_details = progress.get("factor_details", [])
            if factor_details:
                st.markdown("**各因子获取详情:**")
                for detail in factor_details:
                    status_icon = "✅" if detail["status"] == "completed" else "❌"
                    with st.expander(
                        f"{status_icon} {detail['name']} ({detail['weight']}%) — "
                        f"{detail.get('source', '未获取数据') or '未获取数据'}",
                        expanded=False,
                    ):
                        # AI 数据源指导
                        guidance = detail.get("guidance", {})
                        if guidance:
                            st.markdown("**AI 建议的搜索关键词:**")
                            for q in guidance.get("search_queries", []):
                                st.code(q, language="text")
                            if guidance.get("suggested_urls"):
                                st.markdown("**建议数据源:**")
                                for url in guidance["suggested_urls"]:
                                    st.markdown(f"- {url}")
                            if guidance.get("akshare_api"):
                                st.code(f"AKShare: {guidance['akshare_api']}", language="python")

                        # 搜索结果
                        search_results = detail.get("search_results", [])
                        if search_results:
                            st.markdown(f"**搜索结果:** {len(search_results)} 条")
                            for j, r in enumerate(search_results[:3]):
                                title = r.get("title", "无标题")
                                url = r.get("url", "")
                                snippet = (r.get("content", "") or "")[:120]
                                st.markdown(f"**[{j+1}]** [{title}]({url})")
                                if snippet:
                                    st.caption(snippet)

                        # 获取日志（成功/失败记录）
                        fetch_log = detail.get("fetch_log", [])
                        if fetch_log:
                            st.markdown("**数据获取记录:**")
                            for log_item in fetch_log:
                                icon = "✅" if log_item.get("success") else "❌"
                                method = log_item.get("method", "unknown")
                                url = (log_item.get("url", "") or "")[:60]
                                chars = log_item.get("chars", 0)
                                st.caption(f"{icon} {method} — {url} ({chars} 字)")

        return True

    if status == "cancelling":
        st.warning(f"正在终止「{sector}」分析，请稍候...")
        # 检查线程是否已结束
        if not bg["thread"].is_alive():
            progress["status"] = "cancelled"
            st.rerun()
        return True

    if status == "completed":
        st.success(f"「{sector}」分析完成！结果已保存。")
        if st.button("关闭提示", key="dismiss_complete"):
            del st.session_state["bg_analysis"]
            st.rerun()
        return False

    if status == "cancelled":
        st.warning(f"「{sector}」分析已终止。")
        if st.button("关闭", key="dismiss_cancel"):
            del st.session_state["bg_analysis"]
            st.rerun()
        return False

    if status == "failed":
        st.error(f"「{sector}」分析失败: {progress.get('error', '未知错误')}")
        if progress.get("llm_tail"):
            with st.expander("查看 AI 原始输出"):
                st.code(progress["llm_tail"])
        if st.button("关闭", key="dismiss_fail"):
            del st.session_state["bg_analysis"]
            st.rerun()
        return False

    return False


# ==================== 主入口 ====================

def render_cycle_analysis_page() -> None:
    """渲染「周期分析」页面"""
    st.header("周期分析")

    llm_config = get_llm_config()
    if not llm_config:
        st.warning("请先在「设置」页面配置 LLM API Key。")
        return

    # 后台分析状态（顶部横幅）
    is_running = _render_bg_status()

    # 1. 已分析周期列表
    _render_saved_cycles(llm_config)

    # 2. 新建分析
    st.markdown("---")
    st.subheader("新建周期分析")

    # 从 sector_factors.json 加载有因子的板块
    all_factors = _load_all_factors()
    sectors_with_factors = {
        k: v for k, v in all_factors.items() if v.get("factors")
    }

    if not sectors_with_factors:
        st.info("暂无已分析的板块因子。请先在「板块及因子」页面生成因子。")
        return

    sector_names = list(sectors_with_factors.keys())
    selected_sector = st.selectbox(
        "选择板块",
        sector_names,
        help="从「板块及因子」页已分析的板块中选择",
    )

    if selected_sector:
        factors = sectors_with_factors[selected_sector].get("factors", [])
        # 显示将分析的因子
        top_factors = sorted(factors, key=lambda f: f.get("weight", 0), reverse=True)
        factor_summary = " / ".join(
            f"{f['name']} ({f['weight']}%)" for f in top_factors
        )
        st.caption(f"将分析因子: {factor_summary}")

        # 检查是否有已有分析
        saved = _load_all_cycles()
        if selected_sector in saved:
            st.caption(
                f"该板块已有分析结果（{saved[selected_sector].get('analyzed_at', '')[:10]}），"
                "重新分析将覆盖旧数据。"
            )

        # TODO 清单 — 展示分析计划
        with st.container(border=True):
            st.markdown("**分析计划 (TODO)**")
            st.markdown(f"1. 询问 AI 每个因子的数据源建议")
            for j, tf in enumerate(top_factors):
                st.markdown(f"2.{j+1} 获取「{tf['name']}」({tf['weight']}%) 数据 — AKShare → Tavily → Jina → Apify")
            st.markdown(f"{len(top_factors)+2}. AI 综合分析所有因子周期")
            st.markdown(f"{len(top_factors)+3}. 保存结果到 cycle_analysis.json")

        if st.button("开始分析", type="primary", disabled=is_running):
            _start_bg_analysis(selected_sector, factors, llm_config)
            st.rerun()

    # 后台运行中 → 定时刷新
    if is_running:
        time.sleep(2)
        st.rerun()


def _render_data_integration_summary(data: dict) -> None:
    """展示 AI 如何整合多源数据的概览"""
    factors = data.get("factors", [])
    if not factors:
        return

    st.markdown("#### 数据整合概览")
    for f in factors:
        confidence = f.get("data_confidence", "unknown")
        confidence_label = {"high": "高", "medium": "中", "low": "低"}.get(confidence, "未知")
        confidence_color = {"high": "green", "medium": "orange", "low": "red"}.get(confidence, "gray")

        with st.expander(
            f":{confidence_color}[{confidence_label}可信度] **{f['name']}** — {f.get('current_position', '未知')}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("数据可信度", confidence_label)
            col2.metric("周期数量", len(f.get("cycle_data", [])))
            avg_len = f.get("avg_cycle_length_months", 0)
            col3.metric("平均周期长度", f"{avg_len} 月" if avg_len else "N/A")

            current_val = f.get("current_value", "")
            if current_val:
                st.markdown(f"**当前值:** {current_val} {f.get('unit', '')}")

            source_url = f.get("data_source_url", "")
            if source_url:
                st.markdown(f"**数据来源:** [{source_url[:50]}...]({source_url})")

            analysis = f.get("analysis", "")
            if analysis:
                st.markdown("**AI 分析推理:**")
                st.markdown(analysis)

    # 整体判断依据
    overall = data.get("overall", {})
    rationale = overall.get("probability_rationale", "")
    key_signals = overall.get("key_signals", [])
    if rationale or key_signals:
        st.markdown("#### 综合判断依据")
        if rationale:
            st.markdown(f"**反转概率依据:** {rationale}")
        if key_signals:
            st.markdown("**关键信号:**")
            for signal in key_signals:
                st.markdown(f"- {signal}")


def _render_saved_cycles(llm_config: LLMConfig) -> None:
    """渲染已分析周期 expander 列表"""
    saved = _load_all_cycles()
    if not saved:
        return

    st.subheader(f"已分析周期（{len(saved)}）")

    for sector, data in saved.items():
        overall = data.get("overall", {})
        pos = overall.get("cycle_position", "未知")
        prob = overall.get("reversal_probability", 0)
        analyzed_at = data.get("analyzed_at", "")
        time_str = analyzed_at[:10] if analyzed_at else ""

        label = f"**{sector}** — {pos} · 反转概率 {prob}% · {time_str}"
        with st.expander(label):
            _render_cycle_result(sector, data)
            st.markdown("---")
            _render_data_integration_summary(data)
            st.markdown("---")
            st.markdown("##### 对话讨论")
            _render_cycle_chat(sector, llm_config, data)
