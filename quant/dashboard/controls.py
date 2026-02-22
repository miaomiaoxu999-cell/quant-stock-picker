"""策略调参控制面板 — 周期底部龙头策略"""

import copy
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# 行业中文名
IND_CN = {
    "lithium": "锂盐",
    "phosphorus": "磷化工",
    "basic_chem": "基础化工",
}


def init_session_state(config: dict):
    """从 config 初始化 session_state（仅首次调用生效）"""
    if "ctrl_initialized" in st.session_state:
        return

    # 行业权重
    industries = config.get("industries", {})
    st.session_state.industry_weights = {}
    for ind_key, ind_config in industries.items():
        st.session_state.industry_weights[ind_key] = ind_config.get("target_weight", 0)

    # 现金备用
    st.session_state.cash_reserve = config.get("cash_reserve", 0.10)

    # 买入策略
    buy = config.get("buy_strategy", {})
    st.session_state.initial_ratio = buy.get("initial_ratio", 0.30)
    st.session_state.dip_buy_threshold = buy.get("dip_buy_threshold", 0.08)
    st.session_state.max_chase_pct = buy.get("max_chase_pct", 0.10)

    # 卖出策略
    sell = config.get("sell_strategy", {})
    st.session_state.double_sell_ratio = sell.get("double_sell_ratio", 0.50)
    st.session_state.pb_target_sell = sell.get("pb_target_sell", 0.50)

    # 止损策略
    stop = config.get("stop_loss", {})
    st.session_state.l1_drawdown = stop.get("level_1_drawdown", 0.15)
    st.session_state.l1_action = stop.get("level_1_action", 0.50)
    st.session_state.l2_drawdown = stop.get("level_2_drawdown", 0.25)

    # 估值标准
    val = config.get("valuation", {})
    st.session_state.undervalued_ratio = val.get("undervalued_ratio", 0.50)
    st.session_state.fair_ratio = val.get("fair_ratio", 0.67)

    # PB历史高点（可手动修正）
    st.session_state.pb_peaks = {}
    for ind_key, ind_config in industries.items():
        for stock in ind_config.get("stocks", []):
            code = str(stock["code"]).zfill(6)
            st.session_state.pb_peaks[code] = {
                "pb_peak_2022": stock.get("pb_peak_2022"),
                "pb_peak_2017": stock.get("pb_peak_2017"),
            }

    # 持仓状态输入
    st.session_state.holdings = {}

    # 保存状态
    st.session_state.config_saved = True
    st.session_state._original = _snapshot_state()
    st.session_state.ctrl_initialized = True


def _snapshot_state() -> dict:
    """当前调参状态快照"""
    return {
        "industry_weights": dict(st.session_state.industry_weights),
        "cash_reserve": st.session_state.cash_reserve,
        "initial_ratio": st.session_state.initial_ratio,
        "dip_buy_threshold": st.session_state.dip_buy_threshold,
        "max_chase_pct": st.session_state.max_chase_pct,
        "double_sell_ratio": st.session_state.double_sell_ratio,
        "pb_target_sell": st.session_state.pb_target_sell,
        "l1_drawdown": st.session_state.l1_drawdown,
        "l2_drawdown": st.session_state.l2_drawdown,
        "undervalued_ratio": st.session_state.undervalued_ratio,
        "fair_ratio": st.session_state.fair_ratio,
    }


def detect_changes() -> list[str]:
    """对比 session_state vs 原始值，返回变更描述列表"""
    changes = []
    orig = st.session_state.get("_original", {})
    if not orig:
        return changes

    for ind, w in st.session_state.industry_weights.items():
        ow = orig.get("industry_weights", {}).get(ind, 0)
        if abs(w - ow) > 0.001:
            changes.append(f"{IND_CN.get(ind, ind)} {ow:.0%} -> {w:.0%}")

    for key, label in [
        ("cash_reserve", "现金备用"),
        ("initial_ratio", "首次买入比例"),
        ("dip_buy_threshold", "补仓阈值"),
        ("l1_drawdown", "一级止损"),
        ("l2_drawdown", "二级止损"),
        ("undervalued_ratio", "低估阈值"),
        ("fair_ratio", "合理阈值"),
    ]:
        nv = getattr(st.session_state, key, None)
        ov = orig.get(key)
        if nv is not None and ov is not None and abs(nv - ov) > 0.001:
            changes.append(f"{label} {ov:.0%} -> {nv:.0%}")

    return changes


def build_config_from_session_state(original_config: dict) -> dict:
    """从 session_state 组装完整 config dict"""
    config = copy.deepcopy(original_config)

    # 行业权重
    for ind_key in config.get("industries", {}):
        if ind_key in st.session_state.industry_weights:
            config["industries"][ind_key]["target_weight"] = st.session_state.industry_weights[ind_key]

    config["cash_reserve"] = st.session_state.cash_reserve

    # 买入策略
    config["buy_strategy"] = {
        "initial_ratio": st.session_state.initial_ratio,
        "dip_buy_threshold": st.session_state.dip_buy_threshold,
        "max_chase_pct": st.session_state.max_chase_pct,
    }

    # 卖出策略
    config["sell_strategy"] = {
        "double_sell_ratio": st.session_state.double_sell_ratio,
        "pb_target_sell": st.session_state.pb_target_sell,
    }

    # 止损
    config["stop_loss"] = {
        "level_1_drawdown": st.session_state.l1_drawdown,
        "level_1_action": st.session_state.l1_action,
        "level_2_drawdown": st.session_state.l2_drawdown,
        "level_2_action": 0.00,
    }

    # 估值
    config["valuation"] = {
        "undervalued_ratio": st.session_state.undervalued_ratio,
        "fair_ratio": st.session_state.fair_ratio,
        "overvalued_ratio": st.session_state.fair_ratio,
    }

    # PB高点
    for ind_key in config.get("industries", {}):
        for stock in config["industries"][ind_key].get("stocks", []):
            code = str(stock["code"]).zfill(6)
            if code in st.session_state.pb_peaks:
                peaks = st.session_state.pb_peaks[code]
                if peaks.get("pb_peak_2022") is not None:
                    stock["pb_peak_2022"] = peaks["pb_peak_2022"]
                if peaks.get("pb_peak_2017") is not None:
                    stock["pb_peak_2017"] = peaks["pb_peak_2017"]

    return config


def save_config_to_yaml(original_config: dict):
    """保存当前参数到 config.yaml"""
    new_config = build_config_from_session_state(original_config)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    st.session_state.config_saved = True
    st.session_state._original = _snapshot_state()
