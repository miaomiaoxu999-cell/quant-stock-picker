"""个股档案 v2 — 周期关联分析 + 估值排名 + 旧档案"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.analysis.stock_cycle_analyzer import (
    StockCycleAnalyzer,
    get_analyzed_sectors,
    load_cycle_analysis,
    resolve_sector_boards,
    PROFILES_DIR,
    _select_weights,
)
from quant.data.akshare_provider import AKShareProvider
from quant.data.cache import DataCache
from quant.utils.constants import STOCK_PROFILES, IND_CN

# 路径常量
_ROOT = Path(__file__).parent.parent.parent


def render_stock_profile_page():
    """个股档案页面主入口"""
    st.title("个股档案")

    # 审计触发的自动重分析 — 在 tabs 之前执行，让用户直接看到进度
    pending = st.session_state.get("pending_stock_reanalysis")
    if pending:
        _handle_audit_reanalysis(pending)
        return  # reanalysis 内部会 st.rerun()

    # 顶部：自选股摘要
    _render_watchlist_summary()

    # 三个 Tab
    tab_summary, tab_cycle, tab_legacy = st.tabs([
        "跨板块汇总", "周期关联分析", "个股基本档案"
    ])

    with tab_summary:
        _render_cross_sector_summary()

    with tab_cycle:
        _render_cycle_correlation_tab()

    with tab_legacy:
        _render_legacy_profiles()


# ==================== 审计触发重分析 ====================

def _handle_audit_reanalysis(pending_sector: str):
    """审计触发的自动重分析 — 在页面顶层执行"""
    cycle_data = load_cycle_analysis()
    sector_info = cycle_data.get(pending_sector, {})
    overall = sector_info.get("overall", {})
    cycle_position = overall.get("cycle_position", "未知")

    if not sector_info:
        st.error(f"板块「{pending_sector}」无周期分析数据，无法重分析")
        st.session_state.pop("pending_stock_reanalysis", None)
        st.session_state.pop("stock_reanalysis_context", None)
        st.session_state.pop("stock_reanalysis_params", None)
        return

    # 消费 pending 标记
    st.session_state.pop("pending_stock_reanalysis", None)
    audit_context = st.session_state.pop("stock_reanalysis_context", "")
    audit_params = st.session_state.pop("stock_reanalysis_params", {})
    audit_top_n = audit_params.get("top_n", 10)

    st.subheader(f"审计触发重分析：{pending_sector}")
    st.info(f"周期位置：{cycle_position} | 分析个股数：{audit_top_n}")

    if audit_context:
        with st.expander("审计建议参考", expanded=True):
            st.markdown(audit_context)

    # 展示 AI 参数调整建议
    if audit_params:
        with st.expander("AI 参数调整建议", expanded=True):
            if audit_params.get("notes"):
                st.markdown(f"**调整说明:** {audit_params['notes']}")
            if audit_params.get("weights"):
                w = audit_params["weights"]
                st.markdown(
                    f"权重: 上行{w.get('upside', 0):.0%} / "
                    f"对齐{w.get('alignment', 0):.0%} / "
                    f"估值{w.get('valuation', 0):.0%} / "
                    f"动量{w.get('momentum', 0):.0%}"
                )
            if audit_params.get("excluded_codes"):
                reasons = audit_params.get("excluded_reasons", {})
                for code in audit_params["excluded_codes"]:
                    st.markdown(f"- 排除 {code}: {reasons.get(code, '审计建议排除')}")

    _redo_analysis(
        pending_sector, audit_top_n, cycle_position,
        audit_context=audit_context, audit_params=audit_params,
    )


# ==================== 自选列表摘要 ====================

def _render_watchlist_summary():
    """页面顶部展示已自选股票"""
    watchlist = StockCycleAnalyzer.load_watchlist()
    if not watchlist:
        return

    with st.expander(f"自选股票 ({len(watchlist)} 只)", expanded=False):
        cols = st.columns([1, 2, 2, 2, 2, 1])
        cols[0].markdown("**序号**")
        cols[1].markdown("**名称**")
        cols[2].markdown("**代码**")
        cols[3].markdown("**板块**")
        cols[4].markdown("**加入时间**")
        cols[5].markdown("**操作**")

        for i, item in enumerate(watchlist):
            cols = st.columns([1, 2, 2, 2, 2, 1])
            cols[0].write(i + 1)
            cols[1].write(item["name"])
            cols[2].write(item["code"])
            cols[3].write(item.get("sector", ""))
            added = item.get("added_at", "")
            if added:
                try:
                    added = datetime.fromisoformat(added).strftime("%m-%d %H:%M")
                except Exception:
                    pass
            cols[4].write(added)
            if cols[5].button("移除", key=f"rm_watch_{item['code']}"):
                StockCycleAnalyzer.remove_from_watchlist(item["code"])
                st.rerun()


# ==================== Tab 1: 跨板块汇总 ====================

def _get_historical_extremes(code: str) -> dict:
    """从实际历史数据中获取股价最高点和最低点（价格、PB、时间）"""
    provider = AKShareProvider(cache=DataCache())
    pb_data = provider.get_pb_history_long(code, years=15)
    result = {
        "high_price": "-", "high_pb": "-", "high_date": "-",
        "low_price": "-", "low_pb": "-", "low_date": "-",
    }
    if pb_data.empty or "close" not in pb_data.columns:
        return result

    df = pb_data.dropna(subset=["close", "pb"]).copy()
    df = df[df["close"] > 0]
    if df.empty:
        return result

    df["date"] = pd.to_datetime(df["date"])

    # 历史最高价
    idx_max = df["close"].idxmax()
    row_max = df.loc[idx_max]
    result["high_price"] = round(float(row_max["close"]), 2)
    result["high_pb"] = round(float(row_max["pb"]), 2)
    result["high_date"] = row_max["date"].strftime("%Y-%m")

    # 历史最低价
    idx_min = df["close"].idxmin()
    row_min = df.loc[idx_min]
    result["low_price"] = round(float(row_min["close"]), 2)
    result["low_pb"] = round(float(row_min["pb"]), 2)
    result["low_date"] = row_min["date"].strftime("%Y-%m")

    return result


def _render_cross_sector_summary():
    """所有已分析板块的 top 3 个股汇总"""
    all_analyses = StockCycleAnalyzer.load_all_analyses()

    if not all_analyses:
        st.info("暂无已分析板块。请先到「周期关联分析」Tab 分析至少一个板块。")
        return

    watchlist_codes = {w["code"] for w in StockCycleAnalyzer.load_watchlist()}

    # 每个板块取 top 3
    rows = []
    for analysis in all_analyses:
        sector = analysis.get("sector", "")
        cycle_pos = analysis.get("cycle_position", "-")
        stocks = analysis.get("stocks", [])
        # 按 total_score 排序取 top 3（排除 PB 异常）
        valid_stocks = [s for s in stocks if not s.get("pb_anomaly", False)]
        valid_stocks.sort(key=lambda s: s.get("total_score", 0), reverse=True)
        for stock in valid_stocks[:3]:
            val = stock.get("valuation", {})
            cur_price = stock.get("price")
            cur_pb = val.get("current_pb") or stock.get("pb")

            # 从实际历史数据取真实高低点
            ext = _get_historical_extremes(stock.get("code", ""))

            rows.append({
                "板块": sector,
                "排名": stock.get("rank", "-"),
                "名称": stock.get("name", ""),
                "代码": stock.get("code", ""),
                "当前股价": round(cur_price, 2) if cur_price else "-",
                "当前PB": round(cur_pb, 2) if cur_pb else "-",
                "周期位置": cycle_pos,
                "高点价格": ext["high_price"],
                "高点PB": ext["high_pb"],
                "高点时间": ext["high_date"],
                "低点价格": ext["low_price"],
                "低点PB": ext["low_pb"],
                "低点时间": ext["low_date"],
                "上行空间%": val.get("upside_to_peak", "-"),
                "总分": round(stock.get("total_score", 0), 1),
                "_code": stock.get("code", ""),
                "_name": stock.get("name", ""),
                "_sector": sector,
            })

    if not rows:
        st.info("所有板块均无有效分析结果。")
        return

    # 按总分全局排序
    rows.sort(key=lambda r: r.get("总分", 0), reverse=True)

    # 展示表格
    df = pd.DataFrame(rows)
    display_cols = ["板块", "排名", "名称", "代码", "当前股价", "当前PB", "周期位置",
                    "高点价格", "高点PB", "高点时间", "低点价格", "低点PB", "低点时间",
                    "上行空间%", "总分"]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # 加入自选按钮
    st.markdown("---")
    cols = st.columns(4)
    for i, row in enumerate(rows):
        col = cols[i % 4]
        code = row["_code"]
        if code in watchlist_codes:
            col.success(f"{row['_name']} 已在自选")
        else:
            if col.button(f"加入自选: {row['_name']}", key=f"sum_watch_{code}"):
                StockCycleAnalyzer.add_to_watchlist(code, row["_name"], row["_sector"])
                st.rerun()


# ==================== Tab 2: 周期关联分析 ====================

def _render_cycle_correlation_tab():
    """核心分析 Tab"""
    sectors = get_analyzed_sectors()
    if not sectors:
        st.warning("尚未完成任何板块的周期分析。请先到「周期分析」页面分析板块。")
        return

    # 控制面板
    col1, col2 = st.columns([3, 1])
    with col1:
        sector = st.selectbox("选择已分析板块", sectors, key="sp_sector")
    with col2:
        top_n = st.select_slider("分析个股数", options=[5, 10, 15], value=10, key="sp_topn")

    # 显示板块周期状态
    cycle_data = load_cycle_analysis()
    sector_info = cycle_data.get(sector, {})
    overall = sector_info.get("overall", {})
    cycle_position = overall.get("cycle_position", "未知")
    reversal_prob = overall.get("reversal_probability", 0)

    col_a, col_b = st.columns(2)
    col_a.metric("周期位置", cycle_position)
    col_b.metric("反转概率", f"{reversal_prob}%")

    # 检查板块映射
    boards = resolve_sector_boards(sector)
    if not boards:
        st.warning(f"板块 '{sector}' 未能自动匹配，请在 SECTOR_BOARD_MAP 中手动配置。")
        return

    # 已有结果展示
    existing = StockCycleAnalyzer.load_analysis(sector)
    if existing:
        analyzed_at = existing.get("analyzed_at", "")
        try:
            analyzed_at = datetime.fromisoformat(analyzed_at).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        st.caption(f"上次分析: {analyzed_at}")

    # 后台任务状态检测
    from quant.dashboard.bg_task import has_task
    task_id = f"sp_analysis_{sector}"
    is_analyzing = has_task(task_id)

    # 分析按钮（两列布局）
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("分析板块个股", type="primary", key="sp_analyze", disabled=is_analyzing):
            _run_analysis(sector, top_n, cycle_position)
    with btn_col2:
        if existing:
            if st.button("重做分析（清缓存）", type="secondary", key="sp_redo", disabled=is_analyzing):
                _redo_analysis(sector, top_n, cycle_position)

    # 后台分析轮询（按钮外！任务存在就继续轮询）
    if is_analyzing:
        _run_analysis(sector, top_n, cycle_position)

    # 展示结果
    analysis = StockCycleAnalyzer.load_analysis(sector)
    if analysis and analysis.get("stocks"):
        _render_analysis_results(analysis, sector_info)
    elif not existing and not is_analyzing:
        st.info("点击上方按钮开始分析")


def _redo_analysis(
    sector: str, top_n: int, cycle_position: str,
    audit_context: str = "", audit_params: dict | None = None,
):
    """重做分析：删除旧 JSON + 清 PB 缓存 + 重新分析"""
    cache = DataCache()

    if audit_context:
        with st.expander("审计建议参考", expanded=False):
            st.markdown(audit_context)

    # 1. 读取旧分析中的股票代码，清除 PB 缓存
    old = StockCycleAnalyzer.load_analysis(sector)
    if old and old.get("stocks"):
        for s in old["stocks"]:
            code = s.get("code", "")
            if code:
                cache.clear_pattern(f"pb_%{code}%")
        st.toast(f"已清除 {len(old['stocks'])} 只股票的 PB 缓存")

    # 2. 删除旧分析 JSON
    safe_name = sector.replace("/", "_").replace("\\", "_")
    old_path = PROFILES_DIR / f"{safe_name}.json"
    if old_path.exists():
        old_path.unlink()

    # 3. 重新分析（后台）
    _run_analysis(sector, top_n, cycle_position, audit_params=audit_params)


def _analysis_worker(
    sector: str, top_n: int, cycle_position: str,
    audit_params: dict | None = None, _cancel_event=None,
):
    """后台分析 worker — 不调用任何 st.* 函数"""
    from quant.analysis.stock_cycle_analyzer import (
        StockCycleAnalyzer, load_cycle_analysis,
    )
    from quant.data.akshare_provider import AKShareProvider
    from quant.data.cache import DataCache

    analyzer = StockCycleAnalyzer()

    # 阶段1：获取成分股
    stocks = analyzer.get_sector_top_stocks(sector, top_n)
    if not stocks:
        raise ValueError("未获取到板块成分股")

    # 应用审计排除列表
    if audit_params and audit_params.get("excluded_codes"):
        excluded = set(audit_params["excluded_codes"])
        before = len(stocks)
        stocks = [s for s in stocks if s["code"] not in excluded]
        # 如果排除后数量不足，保持原样
        if not stocks:
            stocks = analyzer.get_sector_top_stocks(sector, top_n)

    if _cancel_event and _cancel_event.is_set():
        return None

    # 阶段2：获取历史数据
    cycle_data = load_cycle_analysis()
    sector_info = cycle_data.get(sector, {})
    factors = sector_info.get("factors", [])

    for stock in stocks:
        if _cancel_event and _cancel_event.is_set():
            return None
        pb_data = analyzer.provider.get_pb_history_long(stock["code"])
        stock["pb_data"] = pb_data
        stock["pb_months"] = len(pb_data)
        recent_change = analyzer._get_recent_change(stock["code"], months=6)
        stock["recent_6m_change"] = recent_change

    if _cancel_event and _cancel_event.is_set():
        return None

    # 阶段3：计算相关性与排名
    for stock in stocks:
        pb_data = stock.get("pb_data", __import__("pandas").DataFrame())
        correlation = {}
        for factor in factors:
            factor_name = factor.get("name", "unknown")
            if not pb_data.empty:
                corr = analyzer.compute_correlation(pb_data, factor)
            else:
                corr = {"pearson": None, "confidence": "low", "alignment": "weak",
                        "peak_match_rate": 0, "trough_match_rate": 0}
            correlation[factor_name] = corr
        stock["correlation"] = correlation

        all_factor_cycles = [f.get("cycle_data", []) for f in factors]
        valuation = analyzer.compute_valuation_position(pb_data, all_factor_cycles)
        stock["valuation"] = valuation

    # 应用审计自定义权重
    custom_weights = audit_params.get("weights") if audit_params else None
    analyzer.rank_stocks(stocks, cycle_position, weights=custom_weights)

    # 清理大对象
    for stock in stocks:
        if "pb_data" in stock:
            del stock["pb_data"]

    # 保存
    analyzer.save_analysis(sector, stocks, cycle_position, top_n)

    return {"sector": sector, "stocks": stocks, "cycle_position": cycle_position, "top_n": top_n}


def _run_analysis(
    sector: str, top_n: int, cycle_position: str,
    audit_params: dict | None = None,
):
    """执行分析流程（后台线程）"""
    from quant.dashboard.bg_task import bg_run, clear_task

    task_id = f"sp_analysis_{sector}"
    result = bg_run(task_id, _analysis_worker, sector, top_n, cycle_position, audit_params=audit_params)
    if result is None:
        return

    clear_task(task_id)
    st.success("分析完成！结果已保存。")
    st.rerun()


def _render_score_breakdown(stock: dict, cycle_position: str = "") -> None:
    """渲染 4 维度评分明细"""
    scores = stock.get("scores", {})
    if not scores:
        st.info("该股票暂无评分数据")
        return

    w = _select_weights(cycle_position) if cycle_position else {"upside": 0.40, "alignment": 0.30, "valuation": 0.15, "momentum": 0.15}
    dim_config = [
        ("upside", "上行空间", w["upside"]),
        ("alignment", "周期对齐", w["alignment"]),
        ("valuation", "估值位置", w["valuation"]),
        ("momentum", "近期动量", w["momentum"]),
    ]

    for dim_key, dim_name, weight in dim_config:
        raw = scores.get(dim_key, 0)
        weighted = raw * weight
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.markdown(f"**{dim_name}** (权重 {weight*100:.0f}%)")
        col2.metric("原始分", f"{raw:.1f}")
        col3.metric("加权分", f"{weighted:.1f}")

        # 解释
        if dim_key == "upside":
            val = stock.get("valuation", {})
            upside = val.get("upside_to_peak", 0)
            st.caption(f"上行空间 {upside:.1f}% → 分数 {raw:.1f}")
        elif dim_key == "alignment":
            corr = stock.get("correlation", {})
            pearson_vals = [abs(c.get("pearson") or 0) for c in corr.values()]
            avg_p = sum(pearson_vals) / max(len(pearson_vals), 1)
            st.caption(f"平均|Pearson| {avg_p:.2f} → 分数 {raw:.1f}")
        elif dim_key == "valuation":
            val = stock.get("valuation", {})
            cur_pb = val.get("current_pb", 0)
            st.caption(f"当前 PB {cur_pb:.2f} 相对历史位置 → 分数 {raw:.1f}")
        elif dim_key == "momentum":
            change = stock.get("recent_6m_change", 0) or 0
            st.caption(f"6月涨幅 {change*100:.1f}% → 分数 {raw:.1f}")

    st.markdown("---")
    total = stock.get("total_score", 0)
    st.metric("总分", f"{total:.1f}")


def _render_ranking_reasoning(stocks: list, cycle_position: str) -> None:
    """解释每只股票为什么排在这个位置"""
    w = _select_weights(cycle_position)
    st.markdown(f"**板块周期位置:** {cycle_position}")
    st.markdown(
        f"**排名依据:** 上行空间({w['upside']*100:.0f}%) + "
        f"周期对齐({w['alignment']*100:.0f}%) + "
        f"估值位置({w['valuation']*100:.0f}%) + "
        f"动量({w['momentum']*100:.0f}%)"
    )

    valid = [s for s in stocks if not s.get("pb_anomaly", False)]
    valid.sort(key=lambda s: s.get("total_score", 0), reverse=True)

    for i, stock in enumerate(valid[:5], 1):
        name = stock.get("name", "")
        total = stock.get("total_score", 0)
        scores = stock.get("scores", {})

        with st.expander(f"#{i} {name} (总分 {total:.1f})", expanded=(i == 1)):
            weighted = {
                "上行空间": scores.get("upside", 0) * w["upside"],
                "周期对齐": scores.get("alignment", 0) * w["alignment"],
                "估值位置": scores.get("valuation", 0) * w["valuation"],
                "近期动量": scores.get("momentum", 0) * w["momentum"],
            }
            sorted_dims = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
            st.markdown("**得分贡献排序:**")
            for dim, score in sorted_dims:
                st.markdown(f"- {dim}: **{score:.1f}** 分")

            if stock.get("pb_anomaly"):
                st.warning("PB 数据异常，可能影响准确性")


def _render_correlation_details_tab(stock: dict, factors: list) -> None:
    """展示个股与因子的相关性详情"""
    correlations = stock.get("correlation", {})
    if not correlations:
        st.info("暂无相关性数据")
        return

    for fname, corr in correlations.items():
        with st.expander(f"{fname}", expanded=False):
            col1, col2, col3 = st.columns(3)
            pearson = corr.get("pearson")
            lag = corr.get("best_lag_months", 0)
            alignment = corr.get("alignment", "weak")

            col1.metric("Pearson", f"{pearson:.3f}" if pearson else "N/A")
            col2.metric("最佳滞后", f"{lag} 月")
            col3.metric("对齐度", alignment)

            st.markdown("**解读:**")
            if pearson and abs(pearson) > 0.6:
                direction = "正相关" if pearson > 0 else "负相关"
                st.markdown(f"- 强{direction}，股价跟随因子波动")
            elif pearson and abs(pearson) > 0.3:
                st.markdown("- 中等相关性，部分跟随")
            else:
                st.markdown("- 弱相关或不相关")

            if lag and lag > 0:
                st.markdown(f"- 股价滞后因子约 {lag} 个月")

            peak_match = corr.get("peak_match_rate", 0)
            trough_match = corr.get("trough_match_rate", 0)
            st.markdown(f"- 峰值匹配率: {peak_match*100:.0f}%")
            st.markdown(f"- 谷值匹配率: {trough_match*100:.0f}%")


def _render_data_quality_tab(stocks: list) -> None:
    """展示数据质量和 PB 异常"""
    anomaly_stocks = [s for s in stocks if s.get("pb_anomaly")]

    if anomaly_stocks:
        st.warning(f"{len(anomaly_stocks)} 只股票 PB 数据异常")
        for stock in anomaly_stocks:
            name = stock.get("name", "")
            code = stock.get("code", "")
            pb_months = stock.get("pb_months", 0)
            with st.expander(f"{name} ({code})", expanded=False):
                st.markdown("**异常原因:**")
                if pb_months and pb_months < 12:
                    st.markdown(f"- 历史数据不足（仅 {pb_months} 个月，需至少 12 个月）")
                val = stock.get("valuation", {})
                cur_pb = val.get("current_pb")
                if cur_pb is None or cur_pb <= 0:
                    st.markdown("- 当前 PB 为空或负值（可能亏损或数据源问题）")
                st.markdown("**影响:** 该股票已从排名中降权")
    else:
        st.success("所有股票 PB 数据正常")

    st.markdown("---")
    st.markdown("#### 数据覆盖统计")
    quality_rows = []
    for s in stocks:
        quality_rows.append({
            "名称": s.get("name", ""),
            "代码": s.get("code", ""),
            "PB数据月数": s.get("pb_months", 0),
            "因子相关数": len(s.get("correlation", {})),
            "数据状态": "异常" if s.get("pb_anomaly") else "正常",
        })
    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)


def _render_analysis_results(analysis: dict, sector_info: dict):
    """展示分析结果的五个 SubTab"""
    stocks = analysis.get("stocks", [])
    factors = sector_info.get("factors", [])
    watchlist_codes = {w["code"] for w in StockCycleAnalyzer.load_watchlist()}

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "排名总览", "评分明细", "相关性分析", "估值对比", "数据质量",
    ])

    # ------ SubTab 1: 排名总览 ------
    with sub1:
        cycle_pos = analysis.get("cycle_position", "-")
        rows = []
        for s in stocks:
            val = s.get("valuation", {})
            cur_price = s.get("price")
            cur_pb = val.get("current_pb") or s.get("pb")

            ext = _get_historical_extremes(s.get("code", ""))

            rows.append({
                "排名": s.get("rank", "-"),
                "名称": s.get("name", ""),
                "代码": s.get("code", ""),
                "当前股价": round(cur_price, 2) if cur_price else "-",
                "当前PB": round(cur_pb, 2) if cur_pb else "-",
                "周期位置": cycle_pos,
                "高点价格": ext["high_price"],
                "高点PB": ext["high_pb"],
                "高点时间": ext["high_date"],
                "低点价格": ext["low_price"],
                "低点PB": ext["low_pb"],
                "低点时间": ext["low_date"],
                "上行空间%": val.get("upside_to_peak", "-"),
                "6月涨幅%": round(s.get("recent_6m_change", 0) * 100, 1),
                "总分": round(s.get("total_score", 0), 1),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # 加入自选按钮
        st.markdown("##### 加入自选")
        btn_cols = st.columns(5)
        for i, s in enumerate(stocks):
            col = btn_cols[i % 5]
            code = s.get("code", "")
            name = s.get("name", "")
            if code in watchlist_codes:
                col.caption(f"{name} (已自选)")
            else:
                if col.button(f"{name}", key=f"watch_{code}"):
                    StockCycleAnalyzer.add_to_watchlist(code, name, analysis.get("sector", ""))
                    st.rerun()

    # ------ SubTab 2: 评分明细 ------
    with sub2:
        cycle_pos = analysis.get("cycle_position", "-")
        _render_ranking_reasoning(stocks, cycle_pos)

        st.markdown("---")
        stock_names_detail = [f"{s['name']}({s['code']})" for s in stocks if not s.get("pb_anomaly")]
        valid_stocks_detail = [s for s in stocks if not s.get("pb_anomaly")]
        if valid_stocks_detail:
            selected_detail_idx = st.selectbox(
                "选择个股查看评分明细",
                range(len(stock_names_detail)),
                format_func=lambda i: stock_names_detail[i],
                key="sp_score_stock",
            )
            _render_score_breakdown(valid_stocks_detail[selected_detail_idx], cycle_pos)

    # ------ SubTab 3: 相关性分析 ------
    with sub3:
        stock_names = [f"{s['name']}({s['code']})" for s in stocks]
        selected_idx = st.selectbox(
            "选择个股", range(len(stock_names)),
            format_func=lambda i: stock_names[i],
            key="sp_corr_stock"
        )
        selected_stock = stocks[selected_idx]
        correlations = selected_stock.get("correlation", {})

        if not correlations:
            st.info("该股票暂无相关性数据")
        else:
            # 相关性表格
            corr_rows = []
            for fname, corr in correlations.items():
                corr_rows.append({
                    "因子": fname,
                    "Pearson相关": corr.get("pearson", "-"),
                    "最佳滞后(月)": corr.get("best_lag_months", "-"),
                    "数据点数": corr.get("data_points", 0),
                    "周期对齐": corr.get("alignment", "weak"),
                    "Peak匹配率": corr.get("peak_match_rate", 0),
                    "Trough匹配率": corr.get("trough_match_rate", 0),
                })
            st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

            # 图1：双轴图（因子 vs 个股PB）
            _render_correlation_chart(selected_stock, factors)

        # 相关性详情（expander 方式）
        if correlations:
            st.markdown("---")
            st.markdown("##### 相关性详情")
            _render_correlation_details_tab(selected_stock, factors)

    # ------ SubTab 4: 估值对比 ------
    with sub4:
        _render_valuation_comparison(stocks)

    # ------ SubTab 5: 数据质量 ------
    with sub5:
        _render_data_quality_tab(stocks)


def _render_correlation_chart(stock: dict, factors: list):
    """Plotly 三轴图 — 因子插值曲线 vs 个股 PB vs 股价，图例可点击切换显示"""
    # 尝试重新加载个股 PB 数据（从缓存）
    from quant.data.akshare_provider import AKShareProvider
    from quant.data.cache import DataCache
    provider = AKShareProvider(cache=DataCache())
    pb_data = provider.get_pb_history_long(stock["code"])

    if pb_data.empty:
        st.info("无法加载该股票的 PB 历史数据")
        return

    pb_data = pb_data.copy()
    pb_data["date"] = pd.to_datetime(pb_data["date"])

    # 收集所有 trace 名称，用于构建 multiselect
    all_traces = []
    trace_name_pb = f"{stock['name']} PB"
    all_traces.append(trace_name_pb)
    trace_name_price = f"{stock['name']} 股价"
    if "close" in pb_data.columns and pb_data["close"].notna().any():
        all_traces.append(trace_name_price)
    for factor in factors:
        cycle_data = factor.get("cycle_data", [])
        points = [c for c in cycle_data
                  if (c.get("peak", {}).get("date") and c.get("peak", {}).get("value") is not None)
                  or (c.get("trough", {}).get("date") and c.get("trough", {}).get("value") is not None)]
        if len(points) >= 1:
            all_traces.append(f"{factor['name']}({factor.get('unit', '')})")

    # 线条选择器
    selected = st.multiselect(
        "选择显示的线条",
        options=all_traces,
        default=all_traces,
        key=f"corr_lines_{stock['code']}",
    )

    fig = go.Figure()

    # 个股 PB（右轴 y2）
    fig.add_trace(go.Scatter(
        x=pb_data["date"],
        y=pb_data["pb"],
        name=trace_name_pb,
        yaxis="y2",
        line=dict(color="#2196F3", width=2),
        visible=True if trace_name_pb in selected else "legendonly",
    ))

    # 股价（右轴 y3，独立刻度）
    if "close" in pb_data.columns:
        close_data = pb_data.dropna(subset=["close"])
        if not close_data.empty:
            fig.add_trace(go.Scatter(
                x=close_data["date"],
                y=close_data["close"],
                name=trace_name_price,
                yaxis="y3",
                line=dict(color="#00BCD4", width=1.8),
                visible=True if trace_name_price in selected else "legendonly",
            ))

    # 每个因子的插值曲线（左轴 y）
    colors = ["#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]
    for i, factor in enumerate(factors):
        cycle_data = factor.get("cycle_data", [])
        points = []
        for cycle in cycle_data:
            peak = cycle.get("peak", {})
            trough = cycle.get("trough", {})
            if peak.get("date") and peak.get("value") is not None:
                points.append((StockCycleAnalyzer._parse_quarter_date(peak["date"]), float(peak["value"])))
            if trough.get("date") and trough.get("value") is not None:
                points.append((StockCycleAnalyzer._parse_quarter_date(trough["date"]), float(trough["value"])))

        if len(points) < 2:
            continue

        points.sort(key=lambda x: x[0])
        dates = [p[0] for p in points]
        values = [p[1] for p in points]

        trace_name = f"{factor['name']}({factor.get('unit', '')})"
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            name=trace_name,
            line=dict(color=colors[i % len(colors)], width=1.5, dash="dot"),
            yaxis="y",
            visible=True if trace_name in selected else "legendonly",
        ))

    fig.update_layout(
        title=f"{stock['name']} — 因子与PB·股价走势对比",
        yaxis=dict(title="因子值", side="left"),
        yaxis2=dict(title="PB", side="right", overlaying="y"),
        yaxis3=dict(
            title="股价(元)", side="right", overlaying="y",
            anchor="free", position=1.0,
            showgrid=False,
        ),
        height=500,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25,
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        margin=dict(b=80, r=80),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"corr_chart_{stock['code']}")


def _render_valuation_comparison(stocks: list):
    """PB + PS 柱状图对比 — 使用真实历史数据"""
    valid_stocks = [s for s in stocks
                    if not s.get("pb_anomaly", False)
                    and s.get("valuation", {}).get("current_pb") is not None]
    if not valid_stocks:
        st.info("无有效估值数据（所有股票的 PB 均为空，请检查数据源是否正常）")
        return

    provider = AKShareProvider(cache=DataCache())

    # 从真实历史数据获取 PB 高低点
    names = []
    current_pbs = []
    hist_high_pbs = []
    hist_low_pbs = []
    has_ps = False
    ps_current = []

    for s in valid_stocks:
        code = s.get("code", "")
        names.append(s["name"])
        current_pbs.append(s.get("valuation", {}).get("current_pb") or 0)

        pb_data = provider.get_pb_history_long(code, years=15)
        if not pb_data.empty and "pb" in pb_data.columns:
            pb_vals = pb_data["pb"].dropna()
            pb_positive = pb_vals[pb_vals > 0]
            if len(pb_positive) > 0:
                hist_high_pbs.append(round(float(pb_positive.max()), 2))
                hist_low_pbs.append(round(float(pb_positive.min()), 2))
            else:
                hist_high_pbs.append(0)
                hist_low_pbs.append(0)

            # PS 数据
            if "ps" in pb_data.columns:
                ps_vals = pb_data["ps"].dropna()
                if len(ps_vals) > 0:
                    has_ps = True
                    ps_current.append(float(ps_vals.iloc[-1]))
                else:
                    ps_current.append(0)
            else:
                ps_current.append(0)
        else:
            hist_high_pbs.append(0)
            hist_low_pbs.append(0)
            ps_current.append(0)

    # PB 柱状图
    fig_pb = go.Figure()
    fig_pb.add_trace(go.Bar(
        name="历史最低PB", x=names, y=hist_low_pbs,
        marker_color="#4CAF50",
        text=hist_low_pbs, textposition="outside",
    ))
    fig_pb.add_trace(go.Bar(
        name="当前PB", x=names, y=current_pbs,
        marker_color="#2196F3",
        text=current_pbs, textposition="outside",
    ))
    fig_pb.add_trace(go.Bar(
        name="历史最高PB", x=names, y=hist_high_pbs,
        marker_color="#F44336",
        text=hist_high_pbs, textposition="outside",
    ))
    fig_pb.update_layout(
        title="PB 估值对比（历史最低 / 当前 / 历史最高）",
        barmode="group",
        height=450,
        yaxis_title="PB",
    )
    st.plotly_chart(fig_pb, use_container_width=True)

    if has_ps:
        fig_ps = go.Figure()
        fig_ps.add_trace(go.Bar(
            name="当前PS", x=names, y=ps_current,
            marker_color="#FF9800",
            text=[round(v, 2) for v in ps_current], textposition="outside",
        ))
        fig_ps.update_layout(
            title="PS 辅助参考",
            height=350,
            yaxis_title="PS",
        )
        st.plotly_chart(fig_ps, use_container_width=True)


# ==================== Tab 3: 个股基本档案（综合图表 + 估值信息） ====================

# 核心龙头的 industry → 周期分析板块名映射
_INDUSTRY_TO_SECTOR = {
    "lithium": "锂电池/锂盐",
    "phosphorus": "磷化工",
    "basic_chem": "基础化工",
}


def _render_legacy_profiles():
    """核心龙头档案 + 动态分析个股，每只股票含估值 metrics + 综合图表"""
    # 加载分析数据
    analysis_path = _ROOT / "data" / "latest_analysis.csv"
    if analysis_path.exists():
        try:
            analysis_df = pd.read_csv(analysis_path)
            analysis_df["code"] = analysis_df["code"].astype(str).str.zfill(6)
        except Exception:
            analysis_df = pd.DataFrame()
    else:
        analysis_df = pd.DataFrame()

    cycle_data = load_cycle_analysis()
    provider = AKShareProvider(cache=DataCache())

    # --- Part 1: 核心龙头档案 ---
    st.subheader("核心龙头档案")
    for code, profile in STOCK_PROFILES.items():
        ind_cn = IND_CN.get(profile.get("industry", ""), "")
        sector_key = _INDUSTRY_TO_SECTOR.get(profile.get("industry", ""))
        sector_info = cycle_data.get(sector_key, {}) if sector_key else {}

        # 从 latest_analysis.csv 获取数据
        stock_data = {}
        if not analysis_df.empty:
            match = analysis_df[analysis_df["code"] == code]
            if not match.empty:
                stock_data = match.iloc[0].to_dict()

        with st.expander(
            f"**{profile['name']}** ({code}) — {ind_cn}",
            expanded=False,
        ):
            # 左列：定性信息 | 右列：估值 metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**核心优势**: {profile.get('core_advantage', 'N/A')}")
                st.markdown(f"**成本优势**: {profile.get('cost_advantage', 'N/A')}")
                st.markdown("**风险点**:")
                for risk in profile.get("risks", []):
                    st.markdown(f"- {risk}")

            with col2:
                # 估值 metrics
                current_pb = stock_data.get("pb") if stock_data else None
                pb_peak_2022 = stock_data.get("pb_peak_2022") if stock_data else None
                pb_peak_2017 = stock_data.get("pb_peak_2017") if stock_data else None
                status = stock_data.get("valuation_status", "N/A") if stock_data else "N/A"
                upside = stock_data.get("potential_upside") if stock_data else None

                mc1, mc2, mc3 = st.columns(3)
                if current_pb is not None and pd.notna(current_pb):
                    mc1.metric("当前PB", f"{current_pb:.2f}")
                else:
                    mc1.metric("当前PB", "N/A")
                if pb_peak_2022 is not None and pd.notna(pb_peak_2022):
                    mc2.metric("2022高点PB", f"{float(pb_peak_2022):.1f}")
                else:
                    mc2.metric("历史高点PB", "N/A")
                if pb_peak_2017 is not None and pd.notna(pb_peak_2017):
                    mc3.metric("2017高点PB", f"{float(pb_peak_2017):.1f}")
                else:
                    mc3.metric("历史低点PB", "N/A")

                st.markdown(f"**估值状态**: {status}")
                if upside is not None and pd.notna(upside):
                    st.markdown(f"**上涨空间**: {upside:.0%}")

            # 综合图表 — 核心龙头用 latest_analysis.csv 的高低点
            hist_peak = None
            hist_trough = None
            if stock_data:
                p22 = stock_data.get("pb_peak_2022")
                p17 = stock_data.get("pb_peak_2017")
                peaks = [float(v) for v in [p22, p17] if v is not None and pd.notna(v)]
                if peaks:
                    hist_peak = max(peaks)
                    hist_trough = min(peaks)

            # 找该股票对应板块的最相关因子
            best_factor, best_factor_data = _find_best_factor_for_stock_in_sector(
                code, sector_info
            )

            _render_combined_chart(
                code, profile["name"], provider, sector_info,
                best_factor, best_factor_data,
                hist_peak_pb=hist_peak, hist_trough_pb=hist_trough,
            )

    # --- Part 2: 动态分析板块个股 ---
    profile_dir = _ROOT / "data" / "stock_profiles"
    if not profile_dir.exists():
        return

    profile_files = sorted(profile_dir.glob("*.json"))
    if not profile_files:
        return

    fixed_codes = set(STOCK_PROFILES.keys())

    st.markdown("---")
    st.subheader("动态分析个股")

    for pf in profile_files:
        try:
            sector_data = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue

        sector_name = sector_data.get("sector", pf.stem)
        sector_info = cycle_data.get(sector_name, {})
        stocks = sector_data.get("stocks", [])
        new_stocks = [s for s in stocks if s.get("code", "") not in fixed_codes]
        if not new_stocks:
            continue

        st.markdown(f"#### {sector_name} — 动态分析")
        for s in new_stocks:
            code = s.get("code", "")
            name = s.get("name", "")
            val = s.get("valuation", {})
            current_pb = val.get("current_pb")
            peak_pb = val.get("cycle_peak_pb")
            trough_pb = val.get("cycle_trough_pb")
            upside = val.get("upside_to_peak")
            score = s.get("total_score", 0)
            confidence = val.get("confidence", "low")
            mcap = s.get("market_cap", 0)

            with st.expander(
                f"**{name}** ({code}) — 总分 {score:.1f} | 置信度 {confidence}",
                expanded=False,
            ):
                # 估值 metrics 行
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("当前PB", f"{current_pb:.2f}" if current_pb is not None else "N/A")
                c2.metric("周期峰值PB", f"{peak_pb:.2f}" if peak_pb is not None else "N/A")
                c3.metric("上行空间", f"{upside:.1f}%" if upside is not None else "N/A")
                c4.metric("总市值", f"{mcap/1e8:.0f}亿" if mcap and mcap > 1e8 else "N/A")

                # 找最相关因子
                best_factor, best_factor_data = _find_best_factor_for_dynamic_stock(
                    s, sector_info
                )

                # 动态股票高低点从 BaoStock 15 年数据计算
                _render_combined_chart(
                    code, name, provider, sector_info,
                    best_factor, best_factor_data,
                    hist_peak_pb=None, hist_trough_pb=None,
                )

                st.caption(f"来源板块: {sector_name}")


# ==================== 综合图表辅助函数 ====================

def _find_best_factor_for_stock_in_sector(code: str, sector_info: dict):
    """从周期分析的板块 JSON 中，找到对应股票 Pearson 最高的因子

    用于核心龙头 — 先在 stock_profiles/*.json 里匹配该 code 的 correlation，
    如果找不到就直接返回板块第一个因子。

    Returns:
        (best_factor_name: str | None, best_factor_obj: dict | None)
    """
    factors = sector_info.get("factors", [])
    if not factors:
        return None, None

    # 尝试从动态分析结果中找该股票的 correlation
    profile_dir = _ROOT / "data" / "stock_profiles"
    if profile_dir.exists():
        for pf in profile_dir.glob("*.json"):
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for s in data.get("stocks", []):
                if s.get("code") == code:
                    correlation = s.get("correlation", {})
                    if correlation:
                        best_name = max(
                            correlation,
                            key=lambda k: abs(correlation[k].get("pearson") or 0),
                        )
                        # 匹配 factor 对象
                        for f in factors:
                            if f.get("name") == best_name:
                                return best_name, f
                        return best_name, factors[0]

    # 默认返回第一个因子
    return factors[0].get("name"), factors[0]


def _find_best_factor_for_dynamic_stock(stock: dict, sector_info: dict):
    """从动态分析股票的 correlation 中选 abs(pearson) 最大的因子

    Returns:
        (best_factor_name: str | None, best_factor_obj: dict | None)
    """
    factors = sector_info.get("factors", [])
    correlation = stock.get("correlation", {})

    if not correlation or not factors:
        if factors:
            return factors[0].get("name"), factors[0]
        return None, None

    best_name = max(
        correlation,
        key=lambda k: abs(correlation[k].get("pearson") or 0),
    )
    for f in factors:
        if f.get("name") == best_name:
            return best_name, f
    return best_name, factors[0] if factors else None


def _render_combined_chart(
    code: str,
    name: str,
    provider: AKShareProvider,
    sector_info: dict,
    best_factor_name: str | None,
    best_factor_obj: dict | None,
    hist_peak_pb: float | None = None,
    hist_trough_pb: float | None = None,
):
    """渲染综合图表：上方股价 + 下方 PB/PS + 最相关因子 + 高低点参考线

    Args:
        hist_peak_pb: 核心龙头从 latest_analysis.csv 传入；None 表示自动计算
        hist_trough_pb: 同上
    """
    from plotly.subplots import make_subplots

    pb_data = provider.get_pb_history_long(code, years=15)
    if pb_data.empty:
        st.info(f"{name} 暂无历史数据")
        return

    pb_data = pb_data.copy()
    pb_data["date"] = pd.to_datetime(pb_data["date"])

    # 动态股票：从 15 年 PB 数据自动计算历史 max/min
    if hist_peak_pb is None:
        pb_vals = pb_data["pb"].dropna()
        pb_positive = pb_vals[pb_vals > 0]
        if len(pb_positive) > 0:
            hist_peak_pb = float(pb_positive.max())
            hist_trough_pb = float(pb_positive.min())

    # 创建 2 行共享 X 轴的子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.4, 0.6],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # === 上方面板：月度收盘价 ===
    if "close" in pb_data.columns:
        close_data = pb_data.dropna(subset=["close"])
        fig.add_trace(
            go.Scatter(
                x=close_data["date"], y=close_data["close"],
                name="收盘价", line=dict(color="#2196F3", width=1.5),
            ),
            row=1, col=1,
        )

    # === 下方面板：PB（实线）===
    pb_clean = pb_data.dropna(subset=["pb"])
    fig.add_trace(
        go.Scatter(
            x=pb_clean["date"], y=pb_clean["pb"],
            name="PB", line=dict(color="#FF6B6B", width=2),
        ),
        row=2, col=1, secondary_y=False,
    )

    # PS（细线，半透明）
    if "ps" in pb_data.columns:
        ps_clean = pb_data.dropna(subset=["ps"])
        if not ps_clean.empty:
            fig.add_trace(
                go.Scatter(
                    x=ps_clean["date"], y=ps_clean["ps"],
                    name="PS", line=dict(color="#FF9800", width=1),
                    opacity=0.5,
                ),
                row=2, col=1, secondary_y=False,
            )

    # === 下方叠加：最相关因子（虚线，独立 Y 轴）===
    if best_factor_obj and best_factor_name:
        cycle_points = best_factor_obj.get("cycle_data", [])
        factor_pts = []
        for cycle in cycle_points:
            for point_type in ["peak", "trough"]:
                pt = cycle.get(point_type, {})
                if pt.get("date") and pt.get("value") is not None:
                    try:
                        parsed = StockCycleAnalyzer._parse_quarter_date(pt["date"])
                        factor_pts.append((parsed, float(pt["value"])))
                    except Exception:
                        pass

        if len(factor_pts) >= 2:
            factor_pts.sort(key=lambda x: x[0])
            f_dates = [p[0] for p in factor_pts]
            f_values = [p[1] for p in factor_pts]
            unit = best_factor_obj.get("unit", "")
            fig.add_trace(
                go.Scatter(
                    x=f_dates, y=f_values,
                    name=f"{best_factor_name}({unit})",
                    line=dict(color="#4CAF50", width=1.5, dash="dash"),
                ),
                row=2, col=1, secondary_y=True,
            )

    # === 下方参考线：历史高低点 + 低估线 ===
    if hist_peak_pb is not None:
        fig.add_hline(
            y=hist_peak_pb, line_dash="dash", line_color="red",
            annotation_text=f"历史最高PB {hist_peak_pb:.1f}",
            annotation_position="top left",
            row=2, col=1,
        )
    if hist_trough_pb is not None:
        fig.add_hline(
            y=hist_trough_pb, line_dash="dash", line_color="green",
            annotation_text=f"历史最低PB {hist_trough_pb:.1f}",
            annotation_position="bottom left",
            row=2, col=1,
        )
    if hist_peak_pb is not None:
        undervalued_line = hist_peak_pb * 0.5
        fig.add_hline(
            y=undervalued_line, line_dash="dot", line_color="limegreen",
            annotation_text=f"低估线 {undervalued_line:.1f}",
            annotation_position="bottom right",
            row=2, col=1,
        )

    # 布局
    fig.update_layout(
        title=f"{name}({code}) — 综合走势",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
        margin=dict(b=60, t=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="价格(元)", row=1, col=1)
    fig.update_yaxes(title_text="PB / PS", row=2, col=1, secondary_y=False)
    if best_factor_name:
        fig.update_yaxes(
            title_text=best_factor_name, row=2, col=1, secondary_y=True,
        )

    st.plotly_chart(fig, use_container_width=True, key=f"chart_{code}")
