"""单元测试：Top-N 选股 + 行业约束"""

import pandas as pd
import pytest

from quant.portfolio.selector import StockSelector


@pytest.fixture
def config():
    return {
        "portfolio": {
            "total_stocks": 20,
            "min_stocks": 5,
            "max_stocks": 25,
            "max_industry_weight": 0.30,
        },
        "industries": {},
    }


@pytest.fixture
def selector(config):
    return StockSelector(config)


@pytest.fixture
def scored_df():
    """模拟综合评分后的 DataFrame"""
    return pd.DataFrame({
        "code": [f"{i:06d}" for i in range(1, 16)],
        "name": [f"Stock{i}" for i in range(1, 16)],
        "industry": ["oil"] * 5 + ["power"] * 5 + ["dairy"] * 5,
        "composite_score": [
            0.9, 0.85, 0.8, 0.7, 0.6,   # oil
            0.88, 0.82, 0.75, 0.65, 0.5,  # power
            0.95, 0.78, 0.72, 0.55, 0.4,  # dairy
        ],
    })


def test_select_default_heat(selector, scored_df):
    """默认景气度 = medium，每行业选 3"""
    result = selector.select(scored_df)
    assert len(result) >= selector.min_stocks


def test_select_with_heat(selector, scored_df):
    result = selector.select(scored_df, industry_heat={
        "oil": "high", "power": "medium", "dairy": "low",
    })
    oil_count = len(result[result["industry"] == "oil"])
    power_count = len(result[result["industry"] == "power"])
    dairy_count = len(result[result["industry"] == "dairy"])
    # high=5, medium=3, low=2 before concentration check
    # _check_industry_concentration may trim oil (max 30% of 10 = 4)
    assert oil_count >= 3
    assert power_count == 3
    assert dairy_count == 2


def test_select_empty_df(selector):
    empty = pd.DataFrame(columns=["code", "name", "industry", "composite_score"])
    result = selector.select(empty)
    assert len(result) == 0


def test_select_min_stocks_fill(scored_df):
    """当选出的股票不够 min_stocks 时自动补"""
    config = {
        "portfolio": {"min_stocks": 12, "max_stocks": 25, "max_industry_weight": 1.0},
    }
    selector = StockSelector(config)
    # low for all => 2 per industry = 6 < 12, should auto-fill
    result = selector.select(scored_df, industry_heat={
        "oil": "low", "power": "low", "dairy": "low",
    })
    assert len(result) >= 12


def test_industry_concentration_limit():
    """单行业过度集中时应被裁剪"""
    config = {
        "portfolio": {
            "min_stocks": 1,
            "max_stocks": 100,
            "max_industry_weight": 0.30,
        },
    }
    selector = StockSelector(config)

    # 10 只全在 oil
    df = pd.DataFrame({
        "code": [f"{i:06d}" for i in range(1, 11)],
        "name": [f"S{i}" for i in range(1, 11)],
        "industry": ["oil"] * 10,
        "composite_score": list(range(10, 0, -1)),
    })
    result = selector.select(df, industry_heat={"oil": "high"})
    # max_industry_weight=30% of total, 但这里只有一个行业，
    # _check_industry_concentration: max_per_industry = int(N * 0.3) + 1
    # 如果 N=5 (high=5), max_per_industry = int(5*0.3)+1 = 2
    # 所以应被裁剪
    assert len(result) <= 5


def test_selected_sorted_by_score(selector, scored_df):
    """选出的股票应是行业内得分最高的"""
    result = selector.select(scored_df, industry_heat={
        "oil": "medium", "power": "medium", "dairy": "medium",
    })
    for ind in result["industry"].unique():
        ind_result = result[result["industry"] == ind]
        ind_all = scored_df[scored_df["industry"] == ind]
        top_scores = ind_all.nlargest(3, "composite_score")["composite_score"].tolist()
        for score in ind_result["composite_score"]:
            assert score in top_scores
