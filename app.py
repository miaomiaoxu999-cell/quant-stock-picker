"""Streamlit 交互式看板 — 周期底部龙头投资策略"""

import json
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import streamlit as st

from quant.dashboard.controls import (
    init_session_state, detect_changes, save_config_to_yaml,
    build_config_from_session_state, IND_CN,
)
from quant.utils.constants import STOCK_PROFILES, INDUSTRY_MAP
from quant.dashboard.llm_settings import render_settings_page
from quant.dashboard.sector_factors import render_sector_factors_page

# ==================== 配置 ====================

st.set_page_config(
    page_title="周期底部龙头策略",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_latest_analysis():
    path = DATA_DIR / "latest_analysis.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"code": str})
    return pd.DataFrame()


def load_portfolio_state():
    path = DATA_DIR / "portfolio_state.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_portfolio_state(state):
    path = DATA_DIR / "portfolio_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)



# ==================== 侧边栏 ====================

st.sidebar.title("📊 周期底部龙头策略")
st.sidebar.markdown("---")

# 处理页面跳转请求（必须在 radio 渲染前）
_redirect = st.session_state.pop("_nav_redirect", None)
if _redirect:
    st.session_state["nav_page"] = _redirect

page = st.sidebar.radio(
    "导航",
    [
        "🧩 板块及因子",
        "🔄 周期分析",
        "📑 个股档案",
        "🔍 审计",
        "🤖 投资顾问",
        "💰 仓位管理",
        "⚠️ 风险监控",
        "⚙️ 设置",
    ],
    key="nav_page",
)

from quant.dashboard.bg_task import render_running_indicator
render_running_indicator()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**策略核心**
- PB估值 + 周期底部布局
- AI 驱动选股与仓位配置
- 分批买入，纪律止损
""")

config = load_config()
init_session_state(config)

# ==================== 板块及因子 ====================

if page == "🧩 板块及因子":
    render_sector_factors_page()

# ==================== 设置 ====================

elif page == "⚙️ 设置":
    render_settings_page()

# ==================== 投资顾问 ====================

elif page == "🤖 投资顾问":
    from quant.dashboard.advisor import render_advisor_page
    render_advisor_page()


# ==================== 周期分析 ====================

elif page == "🔄 周期分析":
    from quant.dashboard.cycle_analysis import render_cycle_analysis_page
    render_cycle_analysis_page()


# ==================== 个股档案 ====================

elif page == "📑 个股档案":
    from quant.dashboard.stock_profile import render_stock_profile_page
    render_stock_profile_page()

# ==================== 审计 ====================

elif page == "🔍 审计":
    from quant.dashboard.audit import render_audit_page
    render_audit_page()


# ==================== 仓位管理 ====================

elif page == "💰 仓位管理":
    st.title("仓位管理")

    analysis = load_latest_analysis()
    portfolio_state = load_portfolio_state()

    # 目标仓位展示
    st.subheader("目标仓位 vs 实际仓位")

    industries = config.get("industries", {})
    target_rows = []
    for ind_key, ind_config in industries.items():
        for stock in ind_config.get("stocks", []):
            code = str(stock["code"]).zfill(6)
            holding = portfolio_state.get(code, {})
            target_rows.append({
                "股票": f"{stock['name']}({code})",
                "行业": IND_CN.get(ind_key, ind_key),
                "目标仓位": f"{stock.get('weight', 0):.0%}",
                "实际仓位": f"{holding.get('weight', 0):.0%}",
                "持仓均价": f"{holding.get('avg_cost', 0):.2f}" if holding.get("avg_cost") else "无持仓",
                "持仓数量": holding.get("shares", 0),
                "差异": f"{stock.get('weight', 0) - holding.get('weight', 0):+.0%}",
            })

    st.dataframe(pd.DataFrame(target_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 持仓录入
    st.subheader("录入/更新持仓")
    st.caption("手动输入你的实际持仓信息（持仓成本、数量），系统会据此计算止损线和操作建议")

    with st.form("holding_form"):
        stock_options = []
        for ind_config in industries.values():
            for stock in ind_config.get("stocks", []):
                code = str(stock["code"]).zfill(6)
                stock_options.append(f"{stock['name']}({code})")

        selected = st.selectbox("选择股票", stock_options)

        col1, col2, col3 = st.columns(3)
        with col1:
            avg_cost = st.number_input("持仓均价", min_value=0.0, step=0.01, format="%.2f")
        with col2:
            shares = st.number_input("持仓数量（股）", min_value=0, step=100)
        with col3:
            weight = st.number_input("仓位占比 (%)", min_value=0.0, max_value=100.0, step=1.0)

        submitted = st.form_submit_button("保存持仓")
        if submitted and selected:
            code = selected.split("(")[-1].rstrip(")")
            portfolio_state[code] = {
                "avg_cost": avg_cost,
                "shares": shares,
                "weight": weight / 100,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            save_portfolio_state(portfolio_state)
            st.success(f"已保存 {selected} 的持仓信息")
            st.rerun()


# ==================== 风险监控 ====================

elif page == "⚠️ 风险监控":
    st.title("风险监控")

    analysis = load_latest_analysis()
    portfolio_state = load_portfolio_state()

    # 止损线监控
    st.subheader("止损线监控")

    l1_drawdown = config.get("stop_loss", {}).get("level_1_drawdown", 0.15)
    l2_drawdown = config.get("stop_loss", {}).get("level_2_drawdown", 0.25)

    st.markdown(f"""
    | 级别 | 回撤阈值 | 操作 |
    |------|----------|------|
    | 一级止损 | 回撤 {l1_drawdown:.0%} | 减仓50% |
    | 二级止损 | 回撤 {l2_drawdown:.0%} | 全部止损 |
    """)

    if portfolio_state:
        st.markdown("---")
        st.subheader("当前持仓回撤检测")

        for code, holding in portfolio_state.items():
            avg_cost = holding.get("avg_cost", 0)
            if avg_cost <= 0:
                continue

            # 从分析结果获取当前价格
            current_price = 0
            name = code
            if not analysis.empty:
                match = analysis[analysis["code"] == code]
                if not match.empty:
                    current_price = match.iloc[0].get("price", 0)
                    name = match.iloc[0].get("name", code)

            if current_price > 0:
                drawdown = (avg_cost - current_price) / avg_cost
                profit = (current_price - avg_cost) / avg_cost

                if drawdown >= l2_drawdown:
                    st.error(f"**{name}({code})**: 回撤 {drawdown:.1%} — 触发二级止损！全部卖出")
                elif drawdown >= l1_drawdown:
                    st.warning(f"**{name}({code})**: 回撤 {drawdown:.1%} — 触发一级止损！减仓50%")
                elif drawdown > 0:
                    st.info(f"**{name}({code})**: 回撤 {drawdown:.1%} (安全)")
                else:
                    st.success(f"**{name}({code})**: 盈利 {profit:.1%}")

                # 止损线价格
                l1_price = avg_cost * (1 - l1_drawdown)
                l2_price = avg_cost * (1 - l2_drawdown)
                st.caption(f"成本 {avg_cost:.2f} | 一级止损价 {l1_price:.2f} | 二级止损价 {l2_price:.2f}")
    else:
        st.info("暂无持仓数据。请在「仓位管理」页面录入持仓信息。")

    st.markdown("---")

    # 行业风险提示
    st.subheader("行业风险提示")

    risk_data = {
        "锂盐": [
            "碳酸锂价格持续下跌，行业产能严重过剩",
            "下游新能源车增速放缓",
            "钠电池等替代技术对锂需求的冲击",
            "海外锂矿政策风险（澳大利亚、智利、阿根廷）",
        ],
        "磷化工": [
            "农业周期下行影响磷肥需求",
            "环保政策趋严导致产能受限",
            "磷酸铁锂需求增速不及预期",
        ],
        "基础化工": [
            "MDI/化工品价格周期波动",
            "原油/煤炭等原材料成本波动",
            "全球经济下行压缩化工品需求",
            "产能过剩导致价格战",
        ],
    }

    for ind, risks in risk_data.items():
        with st.expander(f"{ind} 风险因素"):
            for risk in risks:
                st.markdown(f"- {risk}")

    # 宏观风险
    st.subheader("宏观风险因素")
    st.markdown("""
    - **利率风险**: 美联储货币政策转向影响全球资金流动
    - **地缘政治**: 中美关系、俄乌冲突对大宗商品的影响
    - **国内经济**: 房地产下行拖累内需，影响化工品需求
    - **汇率风险**: 人民币汇率波动影响出口型企业利润
    """)
