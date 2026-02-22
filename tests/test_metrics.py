"""单元测试：回测指标"""

import numpy as np
import pandas as pd
import pytest

from quant.backtest.metrics import (
    compute_metrics,
    _compute_max_drawdown,
    _compute_alpha_beta,
    format_metrics,
)


@pytest.fixture
def sample_data():
    """模拟 1 年净值数据"""
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    np.random.seed(42)
    # 组合：年化 ~15% + 随机波动
    port_returns = np.random.normal(0.0006, 0.015, 252)
    port_values = pd.Series(
        (1 + pd.Series(port_returns)).cumprod().values, index=dates
    )
    # 基准：年化 ~8%
    bench_returns = np.random.normal(0.0003, 0.012, 252)
    bench_values = pd.Series(
        (1 + pd.Series(bench_returns)).cumprod().values, index=dates
    )
    return port_values, bench_values


def test_compute_metrics_basic(sample_data):
    port, bench = sample_data
    metrics = compute_metrics(port, bench)
    assert "error" not in metrics
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "alpha" in metrics
    assert "beta" in metrics
    assert metrics["trading_days"] == 251  # 252 data points -> 251 returns


def test_compute_metrics_insufficient_data():
    port = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1))
    bench = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1))
    metrics = compute_metrics(port, bench)
    assert metrics == {"error": "Insufficient data"}


def test_max_drawdown_values():
    # 简单序列：1 -> 2 -> 1 -> 1.5
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    values = pd.Series([1.0, 2.0, 1.0, 1.5], index=dates)
    dd, start, end = _compute_max_drawdown(values)
    assert np.isclose(dd, -0.5)  # 从 2 跌到 1 = 50% 回撤


def test_max_drawdown_no_drawdown():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    values = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
    dd, start, end = _compute_max_drawdown(values)
    assert dd == 0.0


def test_alpha_beta_few_data():
    port = pd.Series([0.01, 0.02])
    bench = pd.Series([0.005, 0.01])
    alpha, beta = _compute_alpha_beta(port, bench, 0.025)
    # 数据不足 10 期，应返回默认值
    assert alpha == 0.0
    assert beta == 1.0


def test_sharpe_positive(sample_data):
    port, bench = sample_data
    metrics = compute_metrics(port, bench)
    # 组合回报 > 无风险利率 2.5%，Sharpe 应为正
    if metrics["annual_return"] > 0.025:
        assert metrics["sharpe_ratio"] > 0


def test_format_metrics(sample_data):
    port, bench = sample_data
    metrics = compute_metrics(port, bench)
    text = format_metrics(metrics)
    assert "BACKTEST RESULTS" in text
    assert "Sharpe Ratio" in text
    assert "Max Drawdown" in text


def test_format_metrics_error():
    text = format_metrics({"error": "Insufficient data"})
    assert "Error" in text


def test_calmar_ratio(sample_data):
    port, bench = sample_data
    metrics = compute_metrics(port, bench)
    # Calmar = annual_return / |max_drawdown|
    if metrics["max_drawdown"] != 0:
        expected = metrics["annual_return"] / abs(metrics["max_drawdown"])
        assert np.isclose(metrics["calmar_ratio"], expected, rtol=0.01)
