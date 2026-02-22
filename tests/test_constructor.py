"""单元测试：组合构建"""

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.constructor import PortfolioConstructor


@pytest.fixture
def config():
    return {"portfolio": {"drawdown_threshold": 0.20}}


@pytest.fixture
def constructor(config):
    return PortfolioConstructor(config)


@pytest.fixture
def selected_df():
    return pd.DataFrame({
        "code": ["000001", "000002", "000003", "000004"],
        "name": ["A", "B", "C", "D"],
        "industry": ["oil", "oil", "dairy", "dairy"],
        "composite_score": [0.9, 0.7, 0.8, 0.6],
    })


def test_equal_weight(constructor, selected_df):
    result = constructor.build_equal_weight(selected_df)
    assert "weight" in result.columns
    assert np.isclose(result["weight"].sum(), 1.0)
    assert all(np.isclose(result["weight"], 0.25))


def test_equal_weight_empty(constructor):
    empty = pd.DataFrame(columns=["code", "composite_score"])
    result = constructor.build_equal_weight(empty)
    assert len(result) == 0


def test_score_weight(constructor, selected_df):
    result = constructor.build_score_weight(selected_df)
    assert "weight" in result.columns
    assert np.isclose(result["weight"].sum(), 1.0)
    # 分数最高的权重最大
    assert result.iloc[0]["weight"] > result.iloc[1]["weight"]


def test_score_weight_no_score_col(constructor):
    df = pd.DataFrame({"code": ["A", "B"], "name": ["X", "Y"]})
    result = constructor.build_score_weight(df)
    # 退回等权
    assert np.isclose(result["weight"].sum(), 1.0)


def test_score_weight_does_not_mutate(constructor, selected_df):
    original_cols = set(selected_df.columns)
    _ = constructor.build_score_weight(selected_df)
    assert set(selected_df.columns) == original_cols


def test_compare_no_previous(constructor, selected_df):
    selected_df = selected_df.copy()
    selected_df["weight"] = 0.25
    result = constructor.compare_with_previous(selected_df, None)
    assert (result["action"] == "BUY").all()
    assert (result["prev_weight"] == 0.0).all()


def test_compare_with_previous(constructor):
    current = pd.DataFrame({
        "code": ["A", "B", "C"],
        "name": ["SA", "SB", "SC"],
        "weight": [0.4, 0.3, 0.3],
    })
    previous = pd.DataFrame({
        "code": ["A", "D"],
        "name": ["SA", "SD"],
        "weight": [0.5, 0.5],
    })
    result = constructor.compare_with_previous(current, previous)
    # A: HOLD/ADJUST, B: BUY, C: BUY, D: SELL
    assert "SELL" in result["action"].values
    assert "BUY" in result["action"].values
    d_row = result[result["code"] == "D"]
    assert d_row.iloc[0]["action"] == "SELL"
    assert d_row.iloc[0]["weight"] == 0.0


def test_drawdown_control_no_trigger(constructor, selected_df):
    selected_df = selected_df.copy()
    selected_df["weight"] = 0.25
    result = constructor.apply_drawdown_control(selected_df, 0.10)
    assert np.isclose(result["weight"].sum(), 1.0)


def test_drawdown_control_trigger(constructor, selected_df):
    selected_df = selected_df.copy()
    selected_df["weight"] = 0.25
    result = constructor.apply_drawdown_control(selected_df, 0.25)
    assert np.isclose(result["weight"].sum(), 0.5)
