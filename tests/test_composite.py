"""单元测试：多层因子加权合成"""

import numpy as np
import pandas as pd
import pytest

from quant.factors.composite import CompositeScorer


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
        },
    }


@pytest.fixture
def scorer(config):
    return CompositeScorer(config)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "code": ["000001", "000002", "000003", "000004"],
        "name": ["StockA", "StockB", "StockC", "StockD"],
        "industry": ["oil", "oil", "dairy", "dairy"],
        "roe_rank": [0.9, 0.5, 0.8, 0.3],
        "revenue_growth_rank": [0.7, 0.6, 0.9, 0.4],
        "cashflow_quality_rank": [0.8, 0.4, 0.7, 0.5],
        "pe_inverse_rank": [0.6, 0.7, 0.5, 0.8],
        "gross_margin_change_rank": [0.5, 0.5, 0.6, 0.4],
    })


def test_score_fundamental(scorer, sample_df):
    result = scorer.score_fundamental(sample_df)
    assert "fundamental_score" in result.columns
    # All scores should be between 0 and 1
    assert result["fundamental_score"].between(0, 1).all()
    # StockA should score higher than StockB (better ranks)
    a_score = result[result["code"] == "000001"]["fundamental_score"].iloc[0]
    b_score = result[result["code"] == "000002"]["fundamental_score"].iloc[0]
    assert a_score > b_score


def test_score_fundamental_does_not_mutate(scorer, sample_df):
    original_cols = set(sample_df.columns)
    _ = scorer.score_fundamental(sample_df)
    assert set(sample_df.columns) == original_cols


def test_score_research_with_data(scorer):
    df = pd.DataFrame({
        "code": ["A", "B", "C"],
        "research_score_raw": [0.8, 0.5, 0.2],
    })
    result = scorer.score_research(df)
    assert "research_score" in result.columns
    assert result["research_score"].between(0, 1).all()
    # Highest raw -> highest normalized
    assert result.iloc[0]["research_score"] == 1.0
    assert result.iloc[2]["research_score"] == 0.0


def test_score_research_without_data(scorer):
    df = pd.DataFrame({"code": ["A", "B"]})
    result = scorer.score_research(df)
    assert "research_score" in result.columns
    assert (result["research_score"] == 0.5).all()


def test_supply_chain_adjustment(scorer, sample_df):
    result = scorer.apply_supply_chain_adjustment(sample_df)
    assert "supply_chain_adj" in result.columns
    # Oil has positive crack_spread_change -> adj > 1
    oil_adj = result[result["industry"] == "oil"]["supply_chain_adj"].iloc[0]
    assert oil_adj > 1.0
    # Dairy has negative milk price -> adj > 1 (cost decrease = good)
    dairy_adj = result[result["industry"] == "dairy"]["supply_chain_adj"].iloc[0]
    assert dairy_adj > 1.0


def test_compute_composite(scorer, sample_df):
    scored = scorer.score_fundamental(sample_df)
    scored["research_score"] = 0.5
    scored["intelligence_score"] = 0.5
    scored["supply_chain_adj"] = 1.0
    result = scorer.compute_composite(scored)
    assert "composite_score" in result.columns
    assert "composite_rank" in result.columns
    assert result["composite_score"].notna().all()


def test_composite_with_nan_ranks(scorer):
    df = pd.DataFrame({
        "code": ["A", "B"],
        "name": ["SA", "SB"],
        "industry": ["oil", "oil"],
        "roe_rank": [0.9, np.nan],
        "revenue_growth_rank": [np.nan, 0.6],
    })
    result = scorer.score_fundamental(df)
    # NaN ranks should be filled with 0.5
    assert result["fundamental_score"].notna().all()
