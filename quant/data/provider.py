"""数据提供者抽象基类"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class DataProvider(ABC):
    """数据源抽象接口"""

    @abstractmethod
    def get_stock_list(self, industry: str) -> pd.DataFrame:
        """获取行业股票列表
        Returns: DataFrame with columns [code, name, industry]
        """
        pass

    @abstractmethod
    def get_daily_quotes(
        self, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取日线行情
        Returns: DataFrame with columns [date, open, high, low, close, volume, amount]
        """
        pass

    @abstractmethod
    def get_financial_indicators(
        self, code: str, date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取财务指标
        Returns: DataFrame with financial metrics
        """
        pass

    @abstractmethod
    def get_realtime_quotes(self, codes: list[str]) -> pd.DataFrame:
        """获取实时行情（含PE、市值等）
        Returns: DataFrame with realtime market data
        """
        pass
