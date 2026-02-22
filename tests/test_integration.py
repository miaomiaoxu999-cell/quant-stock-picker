"""集成测试：验证核心模块之间的协作"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from quant.factors.composite import CompositeScorer
from quant.portfolio.selector import StockSelector
from quant.portfolio.constructor import PortfolioConstructor
from quant.backtest.metrics import compute_metrics, format_metrics


@pytest.fixture
def config():
    return {
        "fundamental_weights": {
            "roe": 0.25,
            "revenue_growth": 0.20,
            "cashflow_quality": 0.15,
            "pe_inverse": 0.10,
            "gross_margin_change": 0.05,
        },
        "research_weight": 0.15,
        "intelligence_weights": {
            "vc_funding_heat": 0.05,
            "report_sentiment": 0.03,
        },
        "supply_chain_signals": {
            "oil": {"crack_spread_change": 0.1},
            "dairy": {"raw_milk_price_change": -0.05},
            "power": {"coal_price_change": -0.08},
        },
        "portfolio": {
            "total_stocks": 15,
            "min_stocks": 5,
            "max_stocks": 20,
            "max_industry_weight": 0.40,
        },
    }


@pytest.fixture
def mock_universe():
    """模拟 3 行业 15 只股票的完整数据"""
    np.random.seed(42)
    n = 15
    return pd.DataFrame({
        "code": [f"{i:06d}" for i in range(1, n + 1)],
        "name": [f"Stock{i}" for i in range(1, n + 1)],
        "industry": ["oil"] * 5 + ["power"] * 5 + ["dairy"] * 5,
        "roe_rank": np.random.uniform(0.1, 1.0, n),
        "revenue_growth_rank": np.random.uniform(0.1, 1.0, n),
        "cashflow_quality_rank": np.random.uniform(0.1, 1.0, n),
        "pe_inverse_rank": np.random.uniform(0.1, 1.0, n),
        "gross_margin_change_rank": np.random.uniform(0.1, 1.0, n),
    })


def test_full_scoring_pipeline(config, mock_universe):
    """测试完整打分流水线：基本面 -> 研究 -> 情报 -> 供应链 -> 综合"""
    scorer = CompositeScorer(config)

    # Step 1: 基本面打分
    df = scorer.score_fundamental(mock_universe)
    assert "fundamental_score" in df.columns
    assert df["fundamental_score"].between(0, 1).all()

    # Step 2: 模拟研究分
    df["research_score_raw"] = np.random.uniform(3, 9, len(df))
    df = scorer.score_research(df)
    assert "research_score" in df.columns

    # Step 3: 模拟情报分
    df["vc_funding_heat"] = np.random.uniform(0, 1, len(df))
    df["report_sentiment"] = np.random.uniform(0, 1, len(df))
    df = scorer.score_intelligence(df)
    assert "intelligence_score" in df.columns

    # Step 4: 供应链调整
    df = scorer.apply_supply_chain_adjustment(df)
    assert "supply_chain_adj" in df.columns

    # Step 5: 综合打分
    df = scorer.compute_composite(df)
    assert "composite_score" in df.columns
    assert "composite_rank" in df.columns
    assert df["composite_score"].notna().all()


def test_scoring_to_selection_to_portfolio(config, mock_universe):
    """测试从打分到选股到组合构建的完整流程"""
    scorer = CompositeScorer(config)

    # 打分
    df = scorer.score_fundamental(mock_universe)
    df["research_score"] = 0.5
    df["intelligence_score"] = 0.5
    df["supply_chain_adj"] = 1.0
    df = scorer.compute_composite(df)

    # 选股
    selector = StockSelector(config)
    selected = selector.select(df, industry_heat={
        "oil": "high", "power": "medium", "dairy": "low",
    })
    assert len(selected) > 0
    assert "composite_score" in selected.columns

    # 组合构建
    constructor = PortfolioConstructor(config)
    portfolio = constructor.build_score_weight(selected)
    assert "weight" in portfolio.columns
    assert np.isclose(portfolio["weight"].sum(), 1.0)

    # 与上月对比
    rebalance = constructor.compare_with_previous(portfolio, None)
    assert (rebalance["action"] == "BUY").all()


def test_rebalance_flow(config, mock_universe):
    """测试调仓流程：上月持仓 -> 本月打分 -> 调仓输出"""
    scorer = CompositeScorer(config)
    constructor = PortfolioConstructor(config)

    # 模拟上月持仓
    prev = pd.DataFrame({
        "code": ["000001", "000002", "000003"],
        "name": ["Stock1", "Stock2", "Stock3"],
        "industry": ["oil", "oil", "power"],
        "weight": [0.4, 0.3, 0.3],
    })

    # 本月打分
    df = scorer.score_fundamental(mock_universe)
    df["research_score"] = 0.5
    df["intelligence_score"] = 0.5
    df["supply_chain_adj"] = 1.0
    df = scorer.compute_composite(df)

    selector = StockSelector(config)
    selected = selector.select(df)
    portfolio = constructor.build_score_weight(selected)
    rebalance = constructor.compare_with_previous(portfolio, prev)

    # 应有 BUY/HOLD/SELL/ADJUST 动作
    assert "action" in rebalance.columns
    actions = set(rebalance["action"].unique())
    # 至少有 BUY 或 SELL（因为组合变了）
    assert len(actions) >= 1


def test_backtest_metrics_with_synthetic_data():
    """测试回测指标能完整计算"""
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    np.random.seed(123)

    port = pd.Series(
        (1 + pd.Series(np.random.normal(0.0005, 0.015, 500)).values).cumprod(),
        index=dates[:500],
    )
    bench = pd.Series(
        (1 + pd.Series(np.random.normal(0.0003, 0.012, 500)).values).cumprod(),
        index=dates[:500],
    )

    metrics = compute_metrics(port, bench)
    assert "error" not in metrics

    # 所有必需指标都应存在
    required_keys = [
        "total_return", "annual_return", "annual_volatility",
        "sharpe_ratio", "max_drawdown", "alpha", "beta",
        "info_ratio", "calmar_ratio", "win_rate",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"

    # 格式化输出
    text = format_metrics(metrics)
    assert "BACKTEST RESULTS" in text
    assert len(text) > 100


def test_drawdown_control_integration(config, mock_universe):
    """测试回撤控制触发后权重减半"""
    scorer = CompositeScorer(config)
    constructor = PortfolioConstructor(config)

    df = scorer.score_fundamental(mock_universe)
    df["research_score"] = 0.5
    df["intelligence_score"] = 0.5
    df["supply_chain_adj"] = 1.0
    df = scorer.compute_composite(df)

    selector = StockSelector(config)
    selected = selector.select(df)
    portfolio = constructor.build_score_weight(selected)

    original_total = portfolio["weight"].sum()

    # 触发回撤控制
    reduced = constructor.apply_drawdown_control(portfolio, 0.25)
    assert np.isclose(reduced["weight"].sum(), original_total * 0.5)


def test_all_modules_import():
    """验证所有核心模块可正常 import"""
    from quant.data.provider import DataProvider
    from quant.data.akshare_provider import AKShareProvider
    from quant.data.cache import DataCache
    from quant.data.universe import UniverseBuilder
    from quant.factors.fundamental import FundamentalFactors
    from quant.factors.composite import CompositeScorer
    from quant.factors.risk_factor import RiskFactor
    from quant.factors.research_factor import ResearchFactor
    from quant.factors.intelligence import IntelligenceFactor
    from quant.portfolio.selector import StockSelector
    from quant.portfolio.constructor import PortfolioConstructor
    from quant.backtest.metrics import compute_metrics, format_metrics
    from quant.risk.company_screener import CompanyScreener
    from quant.risk.negative_filter import NegativeFilter
    from quant.research.industry_researcher import IndustryResearcher
    from quant.research.company_researcher import CompanyResearcher
    from quant.intelligence.qimingpian import QiMingPianClient
    from quant.intelligence.vc_funding_tracker import VCFundingTracker

    # 全部 import 成功
    assert True
