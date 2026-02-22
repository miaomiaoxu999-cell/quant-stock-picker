"""固定龙头股票池构建（从配置文件直接读取6只股票）"""

from typing import Optional

import pandas as pd
from loguru import logger

from quant.data.akshare_provider import AKShareProvider
from quant.data.cache import DataCache


class UniverseBuilder:
    """构建固定的龙头股票池（从config读取）"""

    def __init__(
        self,
        provider: Optional[AKShareProvider] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider or AKShareProvider()
        self.config = config or {}
        self.filters = self.config.get("filters", {})

    def build_universe(self) -> pd.DataFrame:
        """从配置中构建固定股票池，并获取实时行情"""
        industries = self.config.get("industries", {})
        if not industries:
            logger.error("No industries configured!")
            return pd.DataFrame()

        all_stocks = []
        for ind_key, ind_config in industries.items():
            stocks = ind_config.get("stocks", [])
            for s in stocks:
                all_stocks.append({
                    "code": str(s["code"]).zfill(6),
                    "name": s["name"],
                    "industry": ind_key,
                    "target_weight": s.get("weight", 0),
                    "pb_peak_2022": s.get("pb_peak_2022"),
                    "pb_peak_2017": s.get("pb_peak_2017"),
                    "role": s.get("role", "核心持仓"),
                })

        if not all_stocks:
            logger.error("No stocks found in config!")
            return pd.DataFrame()

        universe = pd.DataFrame(all_stocks)
        logger.info(f"Fixed universe: {len(universe)} stocks from {len(industries)} industries")

        # 获取实时行情
        codes = universe["code"].tolist()
        realtime = self.provider.get_realtime_quotes(codes)
        if not realtime.empty:
            universe = universe.merge(
                realtime[["code", "price", "pe_ttm", "pb", "market_cap",
                          "float_market_cap", "volume", "amount"]],
                on="code",
                how="left",
            )

        # 基础过滤（仅ST和停牌检查，不做市值/成交额过滤，因为都是龙头）
        universe = self._apply_filters(universe)

        for _, row in universe.iterrows():
            pb_str = f"PB={row['pb']:.2f}" if pd.notna(row.get('pb')) else "PB=N/A"
            price_str = f"价格={row['price']:.2f}" if pd.notna(row.get('price')) else "价格=N/A"
            logger.info(f"  [{row['industry']}] {row['name']}({row['code']}) {price_str} {pb_str}")

        return universe

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础过滤（仅ST和停牌检查）"""
        # 去 ST
        if self.filters.get("exclude_st", True) and "name" in df.columns:
            mask = ~df["name"].str.contains("ST|\\*ST", na=False, regex=True)
            removed = len(df) - mask.sum()
            if removed > 0:
                logger.warning(f"Removed {removed} ST stocks")
            df = df[mask]

        # 去停牌
        if self.filters.get("exclude_suspended", True) and "price" in df.columns:
            before = len(df)
            df = df[(df["price"] > 0) | (df["price"].isna())]
            removed = before - len(df)
            if removed > 0:
                logger.warning(f"Removed {removed} suspended stocks")

        return df.reset_index(drop=True)
