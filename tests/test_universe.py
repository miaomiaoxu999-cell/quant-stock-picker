"""单元测试：股票池构建"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from quant.data.universe import UniverseBuilder


@pytest.fixture
def config():
    return {
        "industries": {
            "oil": {"concept_boards": ["油气设服"], "industry_boards": ["石油石化"]},
            "dairy": {"concept_boards": ["乳业"], "industry_boards": []},
        },
        "filters": {
            "exclude_st": True,
            "min_market_cap": 0,
            "min_daily_volume": 1_000,
        },
    }


@pytest.fixture
def mock_provider():
    provider = MagicMock()

    # get_stock_list 返回模拟数据（这是 build_universe 真正调用的方法）
    def fake_stock_list(industry):
        if industry == "oil":
            return pd.DataFrame({
                "code": ["600028", "601857", "600583"],
                "name": ["中国石化", "中国石油", "海油工程"],
                "industry": ["oil", "oil", "oil"],
            })
        elif industry == "dairy":
            return pd.DataFrame({
                "code": ["600887", "002732"],
                "name": ["伊利股份", "燕塘乳业"],
                "industry": ["dairy", "dairy"],
            })
        return pd.DataFrame(columns=["code", "name", "industry"])

    provider.get_stock_list.side_effect = fake_stock_list

    # get_realtime_quotes 返回模拟行情
    def fake_quotes(codes):
        return pd.DataFrame({
            "code": codes,
            "name": [f"Stock_{c}" for c in codes],
            "price": [10.0] * len(codes),
            "pe_ttm": [15.0] * len(codes),
            "pb": [2.0] * len(codes),
            "market_cap": [100_000_000_000] * len(codes),
            "volume": [1_000_000] * len(codes),
            "amount": [10_000_000] * len(codes),
        })

    provider.get_realtime_quotes.side_effect = fake_quotes

    return provider


def test_build_universe_basic(config, mock_provider):
    builder = UniverseBuilder(provider=mock_provider, config=config)
    result = builder.build_universe()
    assert "code" in result.columns
    assert "industry" in result.columns
    assert len(result) > 0


def test_build_universe_dedup(config, mock_provider):
    """同一只股票不应出现两次"""
    builder = UniverseBuilder(provider=mock_provider, config=config)
    result = builder.build_universe()
    assert result["code"].duplicated().sum() == 0


def test_build_universe_industry_label(config, mock_provider):
    """每只股票应有正确的行业标签"""
    builder = UniverseBuilder(provider=mock_provider, config=config)
    result = builder.build_universe()
    assert set(result["industry"].unique()).issubset({"oil", "dairy"})


def test_filter_st_stocks(config, mock_provider):
    """ST 股票应被过滤"""
    def fake_stock_list_with_st(industry):
        if industry == "oil":
            return pd.DataFrame({
                "code": ["600028", "999999"],
                "name": ["中国石化", "ST问题股"],
                "industry": ["oil", "oil"],
            })
        return pd.DataFrame(columns=["code", "name", "industry"])

    mock_provider.get_stock_list.side_effect = fake_stock_list_with_st
    builder = UniverseBuilder(provider=mock_provider, config=config)
    result = builder.build_universe()
    if not result.empty:
        st_names = result[result["name"].str.contains("ST", na=False)]
        assert len(st_names) == 0


def test_empty_industry():
    """空行业配置不应崩溃"""
    empty_provider = MagicMock()
    empty_provider.get_stock_list.return_value = pd.DataFrame(
        columns=["code", "name", "industry"]
    )
    empty_provider.get_realtime_quotes.return_value = pd.DataFrame()
    config = {"industries": {}, "filters": {}}
    builder = UniverseBuilder(provider=empty_provider, config=config)
    result = builder.build_universe()
    assert len(result) == 0
