"""PB估值分析引擎 — 替代原 fundamental.py"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant.data.akshare_provider import AKShareProvider


class ValuationEngine:
    """基于PB的估值分析（周期底部龙头策略核心）"""

    def __init__(
        self,
        provider: Optional[AKShareProvider] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider or AKShareProvider()
        self.config = config or {}
        self.valuation_cfg = self.config.get("valuation", {})
        self.undervalued_ratio = self.valuation_cfg.get("undervalued_ratio", 0.50)
        self.fair_ratio = self.valuation_cfg.get("fair_ratio", 0.67)

    def compute_all(self, universe: pd.DataFrame) -> pd.DataFrame:
        """对股票池中每只股票计算PB估值

        Args:
            universe: 含 code, name, industry, pb, pb_peak_2022, pb_peak_2017 的 DataFrame

        Returns:
            添加了估值分析列的 DataFrame:
            - pb_peak: 参考PB高点（取2022和2017的较高值）
            - pb_ratio: 当前PB / 历史高点PB
            - valuation_status: 估值状态（严重低估/低估/合理/高估）
            - potential_upside: 潜在上涨空间
            - pb_percentile: PB历史百分位
        """
        results = []
        for _, row in universe.iterrows():
            code = row["code"]
            current_pb = row.get("pb")
            pb_peak_2022 = row.get("pb_peak_2022")
            pb_peak_2017 = row.get("pb_peak_2017")

            result = self._compute_single(
                code=code,
                current_pb=current_pb,
                pb_peak_2022=pb_peak_2022,
                pb_peak_2017=pb_peak_2017,
            )
            result["code"] = code
            results.append(result)

        if not results:
            return universe

        val_df = pd.DataFrame(results)
        merged = universe.merge(val_df, on="code", how="left")

        logger.info("Valuation analysis complete:")
        for _, row in merged.iterrows():
            status = row.get("valuation_status", "N/A")
            upside = row.get("potential_upside", 0)
            logger.info(
                f"  {row['name']}({row['code']}): PB={row.get('pb', 'N/A')}, "
                f"peak={row.get('pb_peak', 'N/A')}, "
                f"ratio={row.get('pb_ratio', 'N/A')}, "
                f"status={status}, upside={upside:.0%}" if pd.notna(upside) else
                f"  {row['name']}({row['code']}): 数据不足"
            )

        return merged

    def _compute_single(
        self,
        code: str,
        current_pb: float,
        pb_peak_2022: float,
        pb_peak_2017: float,
    ) -> dict:
        """计算单只股票的PB估值"""
        result = {
            "pb_peak": np.nan,
            "pb_ratio": np.nan,
            "valuation_status": "数据不足",
            "potential_upside": np.nan,
            "pb_percentile": np.nan,
        }

        # 取历史PB高点（2022和2017两个周期高点的较高值）
        peaks = [p for p in [pb_peak_2022, pb_peak_2017] if p is not None and not np.isnan(p)]
        if not peaks:
            return result

        pb_peak = max(peaks)
        result["pb_peak"] = pb_peak

        # 当前PB有效性检查
        if current_pb is None or np.isnan(current_pb) or current_pb <= 0:
            return result

        # PB比率 = 当前PB / 历史高点PB
        pb_ratio = current_pb / pb_peak
        result["pb_ratio"] = pb_ratio

        # 估值状态判断
        result["valuation_status"] = self.get_valuation_status(current_pb, pb_peak)

        # 潜在上涨空间 = (高点PB - 当前PB) / 当前PB
        result["potential_upside"] = self.compute_potential_upside(current_pb, pb_peak)

        # PB历史百分位
        result["pb_percentile"] = self.compute_pb_percentile(code)

        return result

    def get_valuation_status(self, current_pb: float, peak_pb: float) -> str:
        """返回估值状态

        Args:
            current_pb: 当前PB
            peak_pb: 历史周期高点PB

        Returns:
            "严重低估" / "低估" / "合理" / "高估"
        """
        if current_pb <= 0 or peak_pb <= 0:
            return "数据不足"

        ratio = current_pb / peak_pb

        if ratio <= self.undervalued_ratio * 0.6:  # <= 30%
            return "严重低估"
        elif ratio <= self.undervalued_ratio:  # <= 50%
            return "低估"
        elif ratio <= self.fair_ratio:  # <= 67%
            return "合理"
        else:
            return "高估"

    @staticmethod
    def compute_potential_upside(current_pb: float, peak_pb: float) -> float:
        """计算潜在上涨空间

        假设股价与PB成正比，PB回到高点时的理论涨幅
        """
        if current_pb <= 0 or peak_pb <= 0:
            return np.nan
        return (peak_pb - current_pb) / current_pb

    def compute_pb_percentile(self, code: str, years: int = 5) -> float:
        """计算PB历史百分位

        当前PB在过去N年中的百分位排名（越低越便宜）
        """
        try:
            pb_hist = self.provider.get_pb_history(code, years=years)
            if pb_hist.empty or "pb" not in pb_hist.columns:
                return np.nan

            pb_series = pb_hist["pb"].dropna()
            if len(pb_series) < 30:  # 数据太少不计算
                return np.nan

            current_row = pb_series.iloc[-1]
            percentile = (pb_series < current_row).sum() / len(pb_series)
            return percentile

        except Exception as e:
            logger.debug(f"PB percentile failed for {code}: {e}")
            return np.nan
