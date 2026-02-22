"""投资顾问页面 — 5 步 LLM 驱动向导"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from quant.llm.client import SiliconFlowClient, LLMError
from quant.dashboard.llm_settings import get_llm_config
from quant.llm.advisor_prompts import (
    build_diagnosis_messages,
    build_allocation_messages,
    build_prediction_messages,
    build_risk_plan_messages,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SESSION_PATH = DATA_DIR / "advisor_session.json"

STEP_LABELS = ["信息收集", "AI 诊断", "仓位配置", "收益预测", "风险监控"]


# ==================== 数据加载 ====================

def _load_latest_analysis() -> pd.DataFrame:
    path = DATA_DIR / "latest_analysis.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"code": str})
    return pd.DataFrame()


def _load_portfolio_state() -> dict:
    path = DATA_DIR / "portfolio_state.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_portfolio_state(state: dict) -> None:
    path = DATA_DIR / "portfolio_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_cycle_data() -> dict:
    path = DATA_DIR / "cycle_analysis.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_sector_factors() -> dict:
    path = DATA_DIR / "sector_factors.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _load_stock_profiles() -> list[dict]:
    """加载 stock_profiles 目录下的动态分析股票"""
    profiles_dir = DATA_DIR / "stock_profiles"
    all_stocks = []
    if not profiles_dir.exists():
        return all_stocks
    for fp in profiles_dir.glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("stocks", []):
                all_stocks.append(s)
        except Exception:
            continue
    return all_stocks


def _derive_valuation_status(valuation: dict) -> str:
    """根据 current_pb / cycle_peak_pb 推算估值状态（与 valuation.py 逻辑一致）"""
    current_pb = valuation.get("current_pb", 0)
    peak_pb = valuation.get("cycle_peak_pb", 0)
    if not current_pb or not peak_pb or current_pb <= 0 or peak_pb <= 0:
        return ""
    ratio = current_pb / peak_pb
    if ratio <= 0.30:
        return "严重低估"
    elif ratio <= 0.50:
        return "低估"
    elif ratio <= 0.67:
        return "合理"
    else:
        return "高估"


def _load_all_available_stocks() -> list[dict]:
    """合并 latest_analysis.csv + stock_profiles/*.json，去重"""
    stocks = []
    seen_codes = set()

    # 1) CSV 核心股
    analysis = _load_latest_analysis()
    if not analysis.empty:
        for _, row in analysis.iterrows():
            code = str(row.get("code", "")).zfill(6)
            if code in seen_codes:
                continue
            seen_codes.add(code)
            stocks.append({
                "code": code,
                "name": row.get("name", ""),
                "price": row.get("price"),
                "pb": row.get("pb"),
                "pe_ttm": row.get("pe_ttm"),
                "valuation_status": row.get("valuation_status", ""),
                "potential_upside": row.get("potential_upside"),
                "cycle_position": row.get("cycle_position", ""),
                "advice": row.get("advice", ""),
                "industry": row.get("industry", ""),
                "role": row.get("role", ""),
                "source": "core",
            })

    # 2) 动态分析股
    for s in _load_stock_profiles():
        code = str(s.get("code", "")).zfill(6)
        if code in seen_codes:
            continue
        seen_codes.add(code)
        stocks.append({
            "code": code,
            "name": s.get("name", ""),
            "price": s.get("price"),
            "pb": s.get("pb"),
            "pe_ttm": s.get("pe_ttm"),
            "valuation_status": _derive_valuation_status(s.get("valuation", {})),
            "potential_upside": s.get("valuation", {}).get("upside_to_peak"),
            "cycle_position": s.get("cycle_position", ""),
            "total_score": s.get("total_score"),
            "source": "profile",
        })

    return stocks


def _load_advisor_session() -> dict | None:
    if SESSION_PATH.exists():
        try:
            with open(SESSION_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_advisor_session(session: dict) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    session["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


# ==================== JSON 提取 ====================

def _extract_json_from_text(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON"""
    # 1) ```json ... ``` 代码块
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 2) 裸 JSON — 平衡大括号解析（避免贪婪正则）
    return _find_balanced_json(text)


def _find_balanced_json(text: str) -> dict | None:
    """从文本中找到第一个平衡的 JSON 对象"""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ==================== 进度条 ====================

def _render_progress_bar(current_step: int) -> None:
    """渲染 5 步进度条"""
    cols = st.columns(5)
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS)):
        step_num = i + 1
        with col:
            if step_num < current_step:
                st.markdown(f"**:green[{step_num}. {label}]**")
            elif step_num == current_step:
                st.markdown(f"**:blue[{step_num}. {label}]**")
            else:
                st.markdown(f":gray[{step_num}. {label}]")


# ==================== 归一化 ====================

def _normalize_allocations(allocations: list[dict], changed_idx: int) -> list[dict]:
    """调整一个比例后，其余等比缩放使总和 = 100%"""
    n = len(allocations)
    if n == 0:
        return allocations
    if n == 1:
        allocations[0]["ratio"] = 1.0
        return allocations

    # 钳位到 [0, 1]
    changed_val = min(max(allocations[changed_idx]["ratio"], 0.0), 1.0)
    allocations[changed_idx]["ratio"] = changed_val
    remaining = 1.0 - changed_val

    others_total = sum(
        allocations[i]["ratio"] for i in range(n) if i != changed_idx
    )

    for i in range(n):
        if i == changed_idx:
            continue
        if others_total > 0:
            allocations[i]["ratio"] = allocations[i]["ratio"] / others_total * remaining
        else:
            allocations[i]["ratio"] = remaining / (n - 1)

    # 修正舍入
    total = sum(a["ratio"] for a in allocations)
    diff = 1.0 - total
    if abs(diff) > 0.001:
        for i in range(n):
            if i != changed_idx:
                allocations[i]["ratio"] += diff
                break

    return allocations


# ==================== LLM 流式调用 ====================

def _call_llm_stream(llm_config, messages: list[dict], retry_key: str = "btn_retry", task_id: str = "advisor_llm") -> str | None:
    """后台 LLM 流式调用，切页不中断"""
    from quant.dashboard.bg_task import bg_llm_stream
    return bg_llm_stream(task_id, llm_config, messages, retry_key=retry_key)


# ==================== Step 1: 信息收集 ====================

def _render_step1() -> None:
    st.subheader("Step 1: 信息收集")
    st.caption("设定投资资金、选择看好的板块和个股")

    # 总资金
    total = st.number_input(
        "总投资资金 (元)",
        min_value=10000,
        step=10000,
        value=st.session_state.get("advisor_total_capital", 500000),
        key="input_total_capital",
    )

    # 看好板块
    sector_factors = _load_sector_factors()
    sector_options = list(sector_factors.keys()) if sector_factors else []
    bullish = st.multiselect(
        "看好板块",
        sector_options,
        default=st.session_state.get("advisor_bullish_sectors", []),
        key="input_bullish_sectors",
    )

    # 看好个股
    all_stocks = _load_all_available_stocks()
    stock_options = {}
    for s in all_stocks:
        code = s["code"]
        name = s.get("name", "")
        parts = [f"{name}({code})"]
        status = s.get("valuation_status", "")
        if status:
            parts.append(status)
        advice = s.get("advice", "")
        score = s.get("total_score")
        if advice:
            parts.append(advice)
        elif isinstance(score, (int, float)):
            parts.append(f"总分 {score:.0f}")
        label = " | ".join(parts)
        stock_options[label] = code

    # 先还原已选中的标签
    prev_codes = st.session_state.get("advisor_favored_stock_codes", [])
    prev_labels = [
        lbl for lbl, c in stock_options.items() if c in prev_codes
    ]

    selected_labels = st.multiselect(
        "看好个股（从已分析股票中选择）",
        list(stock_options.keys()),
        default=prev_labels,
        key="input_favored_stocks",
    )

    # 现有持仓展示
    portfolio = _load_portfolio_state()
    if portfolio:
        st.markdown("---")
        st.subheader("现有持仓")
        analysis = _load_latest_analysis()
        rows = []
        for code, h in portfolio.items():
            name = code
            current_price = 0
            if not analysis.empty:
                match = analysis[analysis["code"] == code]
                if not match.empty:
                    name = match.iloc[0].get("name", code)
                    current_price = match.iloc[0].get("price", 0)
            market_val = current_price * h.get("shares", 0) if current_price else 0
            pnl = 0
            if h.get("avg_cost", 0) > 0 and current_price > 0:
                pnl = (current_price - h["avg_cost"]) / h["avg_cost"]
            rows.append({
                "股票": f"{name}({code})",
                "均价": f"{h.get('avg_cost', 0):.2f}",
                "数量": h.get("shares", 0),
                "现价": f"{current_price:.2f}" if current_price else "N/A",
                "市值": f"{market_val:,.0f}" if market_val else "N/A",
                "盈亏": f"{pnl:+.1%}" if current_price else "N/A",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无持仓数据（可在「仓位管理」页面录入）")

    # 下一步按钮
    st.markdown("---")
    if st.button("下一步: AI 诊断", type="primary", key="btn_step1_next"):
        if not selected_labels:
            st.warning("请至少选择一只股票")
            return

        # 保存到 session_state
        selected_codes = [stock_options[lbl] for lbl in selected_labels]
        st.session_state["advisor_total_capital"] = total
        st.session_state["advisor_bullish_sectors"] = bullish
        st.session_state["advisor_favored_stock_codes"] = selected_codes

        # 构建 favored_stocks 详细数据
        favored = []
        for s in all_stocks:
            if s["code"] in selected_codes:
                favored.append(s)
        st.session_state["advisor_favored_stocks"] = favored

        st.session_state["advisor_step"] = 2
        # 清空后续步骤缓存
        for k in ["advisor_diagnosis_response", "advisor_diagnosis_json",
                   "advisor_confirmed_codes", "advisor_allocation_response",
                   "advisor_allocation_json", "advisor_prediction_response",
                   "advisor_prediction_json", "advisor_risk_response",
                   "advisor_risk_json", "advisor_diagnosis_messages",
                   "advisor_allocation_messages"]:
            st.session_state.pop(k, None)
        from quant.dashboard.bg_task import clear_task
        for tid in ["advisor_diag", "advisor_diag_chat", "advisor_alloc", "advisor_alloc_chat", "advisor_pred", "advisor_risk"]:
            clear_task(tid)
        st.rerun()


# ==================== Step 2: AI 诊断 ====================

def _render_step2(llm_config) -> None:
    st.subheader("Step 2: AI 诊断")
    st.caption("AI 分析你选中的股票，给出买入/持有/减仓建议")

    total_capital = st.session_state.get("advisor_total_capital", 500000)
    favored = st.session_state.get("advisor_favored_stocks", [])
    bullish = st.session_state.get("advisor_bullish_sectors", [])
    portfolio = _load_portfolio_state()
    all_stocks = _load_all_available_stocks()
    cycle_data = _load_cycle_data()

    # 初始化对话历史
    if "advisor_diagnosis_messages" not in st.session_state:
        st.session_state["advisor_diagnosis_messages"] = []

    messages_history = st.session_state["advisor_diagnosis_messages"]

    # 首次进入：调用 LLM
    if "advisor_diagnosis_response" not in st.session_state:
        llm_msgs = build_diagnosis_messages(
            total_capital, bullish, favored, portfolio, all_stocks, cycle_data,
        )
        with st.chat_message("assistant"):
            reply = _call_llm_stream(llm_config, llm_msgs, "btn_retry_diag", task_id="advisor_diag")
        if reply:
            st.session_state["advisor_diagnosis_response"] = reply
            messages_history.append({"role": "assistant", "content": reply})
            parsed = _extract_json_from_text(reply)
            if parsed and "recommended_stocks" in parsed:
                st.session_state["advisor_diagnosis_json"] = parsed
            st.rerun()
        return

    # 显示历史消息
    for msg in messages_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 提取推荐列表
    diag_json = st.session_state.get("advisor_diagnosis_json", {})
    recommended = diag_json.get("recommended_stocks", [])

    # 确认清单
    st.markdown("---")
    st.subheader("确认最终股票清单")

    # 合并推荐 + 用户已选
    all_available = _load_all_available_stocks()
    code_name_map = {s["code"]: s.get("name", s["code"]) for s in all_available}

    recommend_codes = [r["code"] for r in recommended if "code" in r]
    favored_codes = st.session_state.get("advisor_favored_stock_codes", [])
    default_codes = list(dict.fromkeys(recommend_codes + favored_codes))

    confirm_options = {
        f"{code_name_map.get(c, c)}({c})": c for c in code_name_map
    }

    prev_confirmed = st.session_state.get("advisor_confirmed_codes", default_codes)
    default_labels = [
        lbl for lbl, c in confirm_options.items() if c in prev_confirmed
    ]

    confirmed_labels = st.multiselect(
        "选择最终清单（可增删）",
        list(confirm_options.keys()),
        default=default_labels,
        key="input_confirmed_stocks",
    )

    # 追问
    user_input = st.chat_input("追问 AI（可选）...", key="chat_diagnosis")
    if user_input:
        messages_history.append({"role": "user", "content": user_input})
        st.session_state["_pending_advisor_chat_step2"] = True
        st.rerun()

    if st.session_state.get("_pending_advisor_chat_step2"):
        with st.chat_message("assistant"):
            llm_msgs = build_diagnosis_messages(
                total_capital, bullish, favored, portfolio, all_stocks, cycle_data,
            )
            llm_msgs.extend(messages_history[-20:])
            reply = _call_llm_stream(llm_config, llm_msgs, task_id="advisor_diag_chat")
        if reply:
            messages_history.append({"role": "assistant", "content": reply})
            parsed = _extract_json_from_text(reply)
            if parsed and "recommended_stocks" in parsed:
                st.session_state["advisor_diagnosis_json"] = parsed
            st.session_state.pop("_pending_advisor_chat_step2", None)
            from quant.dashboard.bg_task import clear_task
            clear_task("advisor_diag_chat")
            st.rerun()

    # 导航
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("上一步", key="btn_step2_prev"):
            st.session_state["advisor_step"] = 1
            st.rerun()
    with col2:
        if st.button("重新开始", key="btn_step2_reset"):
            _reset_session()
            st.rerun()
    with col3:
        if st.button("确认清单，进入仓位配置", type="primary", key="btn_step2_next"):
            if not confirmed_labels:
                st.warning("请至少确认一只股票")
                return
            confirmed_codes = [confirm_options[lbl] for lbl in confirmed_labels]
            st.session_state["advisor_confirmed_codes"] = confirmed_codes

            # 构建确认股票详细信息
            confirmed_stocks = []
            for s in all_available:
                if s["code"] in confirmed_codes:
                    confirmed_stocks.append(s)
            st.session_state["advisor_confirmed_stocks"] = confirmed_stocks

            st.session_state["advisor_step"] = 3
            for k in ["advisor_allocation_response", "advisor_allocation_json",
                       "advisor_prediction_response", "advisor_prediction_json",
                       "advisor_risk_response", "advisor_risk_json",
                       "advisor_allocation_messages"]:
                st.session_state.pop(k, None)
            from quant.dashboard.bg_task import clear_task
            for tid in ["advisor_alloc", "advisor_alloc_chat", "advisor_pred", "advisor_risk"]:
                clear_task(tid)
            st.rerun()


# ==================== Step 3: 仓位配置 ====================

def _render_step3(llm_config) -> None:
    st.subheader("Step 3: 仓位配置")
    st.caption("AI 建议仓位比例，你可以手动调整（自动归一化）")

    total_capital = st.session_state.get("advisor_total_capital", 500000)
    confirmed = st.session_state.get("advisor_confirmed_stocks", [])
    portfolio = _load_portfolio_state()

    if "advisor_allocation_messages" not in st.session_state:
        st.session_state["advisor_allocation_messages"] = []

    messages_history = st.session_state["advisor_allocation_messages"]

    # 首次进入：调用 LLM
    if "advisor_allocation_response" not in st.session_state:
        llm_msgs = build_allocation_messages(total_capital, confirmed, portfolio)
        with st.chat_message("assistant"):
            reply = _call_llm_stream(llm_config, llm_msgs, "btn_retry_alloc", task_id="advisor_alloc")
        if reply:
            st.session_state["advisor_allocation_response"] = reply
            messages_history.append({"role": "assistant", "content": reply})
            parsed = _extract_json_from_text(reply)
            if parsed and "allocations" in parsed:
                st.session_state["advisor_allocation_json"] = parsed
            st.rerun()
        return

    # 显示历史消息
    for msg in messages_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 可编辑配置表
    alloc_json = st.session_state.get("advisor_allocation_json", {})
    allocations = alloc_json.get("allocations", [])
    cash_reserve = alloc_json.get("cash_reserve", {"ratio": 0.10, "amount": 0, "reason": ""})

    if allocations:
        st.markdown("---")
        st.subheader("配置调整")

        # 用 session_state 管理可编辑比例
        if "advisor_editable_alloc" not in st.session_state:
            st.session_state["advisor_editable_alloc"] = copy.deepcopy(allocations)
            st.session_state["advisor_cash_reserve"] = copy.deepcopy(cash_reserve)

        editable = st.session_state["advisor_editable_alloc"]
        cash = st.session_state["advisor_cash_reserve"]

        changed = False
        changed_idx = -1

        for i, a in enumerate(editable):
            col1, col2, col3, col4 = st.columns([2, 1.5, 1, 1])
            with col1:
                st.markdown(f"**{a.get('name', '')}** `{a.get('code', '')}`")
                st.caption(f"操作: {a.get('action', '')} | 价格区间: {a.get('price_range', 'N/A')}")
            with col2:
                new_ratio = st.number_input(
                    "比例%",
                    min_value=0.0, max_value=100.0,
                    value=round(a.get("ratio", 0) * 100, 1),
                    step=1.0,
                    key=f"alloc_ratio_{i}",
                    label_visibility="collapsed",
                )
                new_ratio_dec = new_ratio / 100.0
                if abs(new_ratio_dec - a.get("ratio", 0)) > 0.001:
                    a["ratio"] = new_ratio_dec
                    changed = True
                    changed_idx = i
            with col3:
                amount = total_capital * a.get("ratio", 0)
                a["amount"] = amount
                st.metric("金额", f"{amount:,.0f}")
            with col4:
                price = a.get("price_range", "").split("-")[0]
                try:
                    price_val = float(price)
                    shares = int(amount / price_val / 100) * 100 if price_val > 0 else 0
                except (ValueError, TypeError):
                    shares = 0
                a["shares"] = shares
                st.metric("估算股数", f"{shares}")

        # 自动归一化（股票之间等比缩放，现金 = 剩余）
        if changed and changed_idx >= 0:
            _normalize_allocations(editable, changed_idx)
            stock_total = sum(a.get("ratio", 0) for a in editable)
            cash["ratio"] = max(0.0, 1.0 - stock_total)
            st.rerun()

        # 现金行
        st.markdown("---")
        cash_pct = cash.get("ratio", 0.1) * 100
        cash_amount = total_capital * cash.get("ratio", 0.1)
        st.markdown(f"**现金储备**: {cash_pct:.1f}% = {cash_amount:,.0f} 元")
        st.caption(cash.get("reason", ""))

        # 汇总
        total_alloc = sum(a.get("ratio", 0) for a in editable)
        total_pct = (total_alloc + cash.get("ratio", 0)) * 100
        st.markdown(f"**合计**: 股票 {total_alloc:.1%} + 现金 {cash.get('ratio', 0):.1%} = {total_pct:.1f}%")

    # 追问
    user_input = st.chat_input("追问 AI 调整配置...", key="chat_allocation")
    if user_input:
        messages_history.append({"role": "user", "content": user_input})
        st.session_state["_pending_advisor_chat_step3"] = True
        st.rerun()

    if st.session_state.get("_pending_advisor_chat_step3"):
        with st.chat_message("assistant"):
            llm_msgs = build_allocation_messages(
                total_capital, confirmed, portfolio, messages_history[-20:],
            )
            reply = _call_llm_stream(llm_config, llm_msgs, task_id="advisor_alloc_chat")
        if reply:
            messages_history.append({"role": "assistant", "content": reply})
            parsed = _extract_json_from_text(reply)
            if parsed and "allocations" in parsed:
                st.session_state["advisor_allocation_json"] = parsed
                st.session_state.pop("advisor_editable_alloc", None)
                st.session_state.pop("advisor_cash_reserve", None)
            st.session_state.pop("_pending_advisor_chat_step3", None)
            from quant.dashboard.bg_task import clear_task
            clear_task("advisor_alloc_chat")
            st.rerun()

    # 导航
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("上一步", key="btn_step3_prev"):
            st.session_state["advisor_step"] = 2
            st.rerun()
    with col2:
        if st.button("重新开始", key="btn_step3_reset"):
            _reset_session()
            st.rerun()
    with col3:
        if st.button("确认配置，查看收益预测", type="primary", key="btn_step3_next"):
            # 保存最终配置
            final_alloc = st.session_state.get("advisor_editable_alloc", allocations)
            for a in final_alloc:
                a["amount"] = total_capital * a.get("ratio", 0)
            st.session_state["advisor_final_allocation"] = final_alloc
            st.session_state["advisor_final_cash"] = st.session_state.get(
                "advisor_cash_reserve", cash_reserve,
            )
            st.session_state["advisor_step"] = 4
            for k in ["advisor_prediction_response", "advisor_prediction_json",
                       "advisor_risk_response", "advisor_risk_json"]:
                st.session_state.pop(k, None)
            from quant.dashboard.bg_task import clear_task
            for tid in ["advisor_pred", "advisor_risk"]:
                clear_task(tid)
            st.rerun()


# ==================== Step 4: 收益预测 ====================

def _render_step4(llm_config) -> None:
    st.subheader("Step 4: 收益预测")
    st.caption("3 时间段 x 3 情景的收益预测（含交易成本和股息）")

    total_capital = st.session_state.get("advisor_total_capital", 500000)
    allocation = st.session_state.get("advisor_final_allocation", [])
    cycle_data = _load_cycle_data()

    # 首次进入
    if "advisor_prediction_response" not in st.session_state:
        llm_msgs = build_prediction_messages(total_capital, allocation, cycle_data)
        with st.chat_message("assistant"):
            reply = _call_llm_stream(llm_config, llm_msgs, "btn_retry_pred", task_id="advisor_pred")
        if reply:
            st.session_state["advisor_prediction_response"] = reply
            parsed = _extract_json_from_text(reply)
            if parsed and "predictions" in parsed:
                st.session_state["advisor_prediction_json"] = parsed
            st.rerun()
        return

    # 显示分析文字
    st.markdown(st.session_state["advisor_prediction_response"])

    # 渲染图表
    pred_json = st.session_state.get("advisor_prediction_json", {})
    predictions = pred_json.get("predictions", {})

    if predictions:
        st.markdown("---")
        st.subheader("收益预测图表")

        # Plotly 分组柱状图
        periods = ["3m", "6m", "12m"]
        period_labels = ["3个月", "6个月", "12个月"]
        scenarios = ["pessimistic", "baseline", "optimistic"]
        scenario_labels = ["悲观", "基准", "乐观"]
        colors = ["#FF6B6B", "#FFD700", "#00CC00"]

        fig = go.Figure()
        for scenario, label, color in zip(scenarios, scenario_labels, colors):
            values = []
            for p in periods:
                p_data = predictions.get(p, {}).get(scenario, {})
                values.append(p_data.get("total_value", total_capital))

            fig.add_trace(go.Bar(
                name=label,
                x=period_labels,
                y=values,
                marker_color=color,
                text=[f"{v:,.0f}" for v in values],
                textposition="outside",
            ))

        fig.add_hline(
            y=total_capital,
            line_dash="dash", line_color="white",
            annotation_text=f"本金 {total_capital:,.0f}",
        )
        fig.update_layout(
            barmode="group",
            title="不同情景下的预期总资产",
            yaxis_title="总资产 (元)",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 明细表
        st.subheader("收益明细")
        detail_rows = []
        for p, p_label in zip(periods, period_labels):
            for scenario, s_label in zip(scenarios, scenario_labels):
                d = predictions.get(p, {}).get(scenario, {})
                detail_rows.append({
                    "时间段": p_label,
                    "情景": s_label,
                    "收益率": f"{d.get('return_rate', 0):.1%}",
                    "总资产": f"{d.get('total_value', 0):,.0f}",
                    "股息收入": f"{d.get('dividend', 0):,.0f}",
                    "交易成本": f"{d.get('cost', 0):,.0f}",
                })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

        # 假设说明
        assumptions = pred_json.get("assumptions", {})
        if assumptions:
            with st.expander("情景假设说明"):
                for k, v in assumptions.items():
                    label_map = {"optimistic": "乐观", "baseline": "基准", "pessimistic": "悲观"}
                    st.markdown(f"**{label_map.get(k, k)}**: {v}")

    # 导航
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("上一步", key="btn_step4_prev"):
            st.session_state["advisor_step"] = 3
            st.rerun()
    with col2:
        if st.button("重新开始", key="btn_step4_reset"):
            _reset_session()
            st.rerun()
    with col3:
        if st.button("下一步: 风险监控计划", type="primary", key="btn_step4_next"):
            st.session_state["advisor_step"] = 5
            for k in ["advisor_risk_response", "advisor_risk_json"]:
                st.session_state.pop(k, None)
            from quant.dashboard.bg_task import clear_task
            clear_task("advisor_risk")
            st.rerun()


# ==================== Step 5: 风险监控 ====================

def _render_step5(llm_config) -> None:
    st.subheader("Step 5: 风险监控计划")
    st.caption("个股风险、板块风险、宏观风险的系统性监控方案")

    allocation = st.session_state.get("advisor_final_allocation", [])
    cycle_data = _load_cycle_data()

    # 首次进入
    if "advisor_risk_response" not in st.session_state:
        llm_msgs = build_risk_plan_messages(allocation, cycle_data)
        with st.chat_message("assistant"):
            reply = _call_llm_stream(llm_config, llm_msgs, "btn_retry_risk", task_id="advisor_risk")
        if reply:
            st.session_state["advisor_risk_response"] = reply
            parsed = _extract_json_from_text(reply)
            if parsed:
                st.session_state["advisor_risk_json"] = parsed
            st.rerun()
        return

    # 显示分析文字
    st.markdown(st.session_state["advisor_risk_response"])

    # 渲染结构化风险表
    risk_json = st.session_state.get("advisor_risk_json", {})

    # 个股风险
    stock_risks = risk_json.get("stock_risks", [])
    if stock_risks:
        st.markdown("---")
        st.subheader("个股风险")
        for sr in stock_risks:
            with st.expander(f"**{sr.get('name', '')}** ({sr.get('code', '')})"):
                for r in sr.get("risks", []):
                    cols = st.columns([3, 1, 2, 2, 2])
                    with cols[0]:
                        st.markdown(f"**{r.get('signal', '')}**")
                    with cols[1]:
                        st.markdown(f"频率: {r.get('frequency', '')}")
                    with cols[2]:
                        st.markdown(f"阈值: {r.get('threshold', '')}")
                    with cols[3]:
                        st.markdown(f"应对: {r.get('action', '')}")
                    with cols[4]:
                        st.caption(f"来源: {r.get('source', '')}")

    # 板块风险
    sector_risks = risk_json.get("sector_risks", [])
    if sector_risks:
        st.markdown("---")
        st.subheader("板块风险")
        rows = []
        for r in sector_risks:
            rows.append({
                "板块": r.get("sector", ""),
                "风险信号": r.get("signal", ""),
                "频率": r.get("frequency", ""),
                "阈值": r.get("threshold", ""),
                "应对": r.get("action", ""),
                "来源": r.get("source", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 宏观风险
    macro_risks = risk_json.get("macro_risks", [])
    if macro_risks:
        st.markdown("---")
        st.subheader("宏观风险")
        rows = []
        for r in macro_risks:
            rows.append({
                "风险信号": r.get("signal", ""),
                "频率": r.get("frequency", ""),
                "阈值": r.get("threshold", ""),
                "应对": r.get("action", ""),
                "来源": r.get("source", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 同步持仓 & 保存
    st.markdown("---")
    sync = st.checkbox("同步更新仓位管理的持仓数据", key="advisor_sync_portfolio")

    col1, col2, col3, col4 = st.columns([1, 1, 1.5, 1.5])
    with col1:
        if st.button("上一步", key="btn_step5_prev"):
            st.session_state["advisor_step"] = 4
            st.rerun()
    with col2:
        if st.button("重新开始", key="btn_step5_reset"):
            _reset_session()
            st.rerun()
    with col3:
        if st.button("保存投资计划", type="primary", key="btn_step5_save"):
            # 同步持仓
            if sync:
                _sync_portfolio()

            # 保存 session
            session = _build_session_data()
            _save_advisor_session(session)
            st.success("投资计划已保存!")

    with col4:
        session = _build_session_data()
        st.download_button(
            "下载计划 JSON",
            data=json.dumps(session, ensure_ascii=False, indent=2),
            file_name=f"advisor_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            key="btn_download",
        )


# ==================== 辅助函数 ====================

def _reset_session() -> None:
    """清空所有 advisor_ 前缀的 session_state"""
    from quant.dashboard.bg_task import clear_task
    keys_to_remove = [k for k in st.session_state if k.startswith("advisor_")]
    for k in keys_to_remove:
        del st.session_state[k]
    st.session_state["advisor_step"] = 1
    # 清除后台任务
    for tid in ["advisor_diag", "advisor_diag_chat", "advisor_alloc", "advisor_alloc_chat", "advisor_pred", "advisor_risk"]:
        clear_task(tid)


def _sync_portfolio() -> None:
    """将最终配置同步到 portfolio_state.json"""
    allocation = st.session_state.get("advisor_final_allocation", [])
    if not allocation:
        return

    portfolio = _load_portfolio_state()
    for a in allocation:
        code = a.get("code", "")
        if not code:
            continue
        action = a.get("action", "")
        if action in ("buy", "add"):
            existing = portfolio.get(code, {})
            old_shares = existing.get("shares", 0)
            old_cost = existing.get("avg_cost", 0)
            new_shares = a.get("shares", 0)
            try:
                new_price = float(a.get("price_range", "0").split("-")[0])
            except (ValueError, TypeError):
                new_price = 0
            total_shares = old_shares + new_shares
            if total_shares > 0 and new_price > 0:
                avg_cost = (old_cost * old_shares + new_price * new_shares) / total_shares
            else:
                avg_cost = old_cost or new_price
            portfolio[code] = {
                "avg_cost": avg_cost,
                "shares": total_shares,
                "weight": a.get("ratio", 0),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        elif action == "sell":
            portfolio.pop(code, None)
        elif action == "reduce":
            if code in portfolio:
                portfolio[code]["shares"] = max(
                    0, portfolio[code].get("shares", 0) - abs(a.get("shares", 0))
                )
                portfolio[code]["weight"] = a.get("ratio", 0)
                portfolio[code]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        elif action == "hold":
            if code in portfolio:
                portfolio[code]["weight"] = a.get("ratio", 0)
                portfolio[code]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    _save_portfolio_state(portfolio)


def _build_session_data() -> dict:
    """构建要保存的 session 数据"""
    return {
        "created_at": st.session_state.get(
            "advisor_created_at",
            datetime.now().isoformat(timespec="seconds"),
        ),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "current_step": st.session_state.get("advisor_step", 1),
        "inputs": {
            "total_capital": st.session_state.get("advisor_total_capital", 0),
            "bullish_sectors": st.session_state.get("advisor_bullish_sectors", []),
            "favored_stocks": [
                {"code": s.get("code", ""), "name": s.get("name", "")}
                for s in st.session_state.get("advisor_favored_stocks", [])
            ],
        },
        "confirmed_stocks": st.session_state.get("advisor_confirmed_codes", []),
        "allocation": st.session_state.get("advisor_final_allocation", []),
        "cash_reserve": st.session_state.get("advisor_final_cash", {}),
        "prediction": st.session_state.get("advisor_prediction_json", {}),
        "risk_plan": st.session_state.get("advisor_risk_json", {}),
        "conversations": {
            "diagnosis": st.session_state.get("advisor_diagnosis_messages", []),
            "allocation": st.session_state.get("advisor_allocation_messages", []),
        },
    }


# ==================== 主入口 ====================

def render_advisor_page() -> None:
    """渲染投资顾问页面"""
    st.title("投资顾问")

    # 检查 LLM 配置
    llm_config = get_llm_config()
    if llm_config is None:
        st.warning("请先在「设置」页面配置 LLM API Key。")
        return

    # 检查是否有上次保存的计划
    if "advisor_step" not in st.session_state:
        saved = _load_advisor_session()
        if saved:
            st.info("发现上次保存的投资计划")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("继续上次计划", type="primary", key="btn_resume"):
                    # 恢复 session_state
                    inputs = saved.get("inputs", {})
                    st.session_state["advisor_step"] = saved.get("current_step", 1)
                    st.session_state["advisor_total_capital"] = inputs.get("total_capital", 500000)
                    st.session_state["advisor_bullish_sectors"] = inputs.get("bullish_sectors", [])
                    st.session_state["advisor_favored_stock_codes"] = [
                        s.get("code", "") for s in inputs.get("favored_stocks", [])
                    ]
                    st.session_state["advisor_confirmed_codes"] = saved.get("confirmed_stocks", [])
                    st.session_state["advisor_created_at"] = saved.get("created_at", "")

                    # 从 codes 重建完整股票数据
                    all_stocks = _load_all_available_stocks()
                    fav_codes = set(st.session_state.get("advisor_favored_stock_codes", []))
                    if fav_codes:
                        st.session_state["advisor_favored_stocks"] = [
                            s for s in all_stocks if s["code"] in fav_codes
                        ]
                    conf_codes = set(st.session_state.get("advisor_confirmed_codes", []))
                    if conf_codes:
                        st.session_state["advisor_confirmed_stocks"] = [
                            s for s in all_stocks if s["code"] in conf_codes
                        ]

                    # 恢复 LLM 结果
                    if saved.get("allocation"):
                        st.session_state["advisor_final_allocation"] = saved["allocation"]
                    if saved.get("cash_reserve"):
                        st.session_state["advisor_final_cash"] = saved["cash_reserve"]
                    if saved.get("prediction"):
                        st.session_state["advisor_prediction_json"] = saved["prediction"]
                    if saved.get("risk_plan"):
                        st.session_state["advisor_risk_json"] = saved["risk_plan"]

                    # 恢复对话
                    convs = saved.get("conversations", {})
                    if convs.get("diagnosis"):
                        st.session_state["advisor_diagnosis_messages"] = convs["diagnosis"]
                        # 恢复最后一条 assistant 消息作为 response
                        for msg in reversed(convs["diagnosis"]):
                            if msg["role"] == "assistant":
                                st.session_state["advisor_diagnosis_response"] = msg["content"]
                                parsed = _extract_json_from_text(msg["content"])
                                if parsed and "recommended_stocks" in parsed:
                                    st.session_state["advisor_diagnosis_json"] = parsed
                                break
                    if convs.get("allocation"):
                        st.session_state["advisor_allocation_messages"] = convs["allocation"]
                        for msg in reversed(convs["allocation"]):
                            if msg["role"] == "assistant":
                                st.session_state["advisor_allocation_response"] = msg["content"]
                                parsed = _extract_json_from_text(msg["content"])
                                if parsed and "allocations" in parsed:
                                    st.session_state["advisor_allocation_json"] = parsed
                                break

                    st.rerun()
            with col2:
                if st.button("重新开始", key="btn_fresh"):
                    st.session_state["advisor_step"] = 1
                    st.session_state["advisor_created_at"] = datetime.now().isoformat(
                        timespec="seconds",
                    )
                    st.rerun()
            return
        else:
            st.session_state["advisor_step"] = 1
            st.session_state["advisor_created_at"] = datetime.now().isoformat(
                timespec="seconds",
            )

    current_step = st.session_state.get("advisor_step", 1)
    _render_progress_bar(current_step)
    st.markdown("---")

    if current_step == 1:
        _render_step1()
    elif current_step == 2:
        _render_step2(llm_config)
    elif current_step == 3:
        _render_step3(llm_config)
    elif current_step == 4:
        _render_step4(llm_config)
    elif current_step == 5:
        _render_step5(llm_config)
