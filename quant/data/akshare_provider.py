"""AKShare 数据提供者 - 免费 A 股数据"""

import time
from typing import Optional
from datetime import datetime, timedelta

import akshare as ak
import baostock as bs
import pandas as pd
from loguru import logger

from quant.data.provider import DataProvider
from quant.data.cache import DataCache


class AKShareProvider(DataProvider):
    """基于 AKShare 的数据获取"""

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()

    @staticmethod
    def _normalize_code(code) -> str:
        """统一股票代码为 6 位字符串（如 '002732'）"""
        return str(code).zfill(6)

    @staticmethod
    def _code_to_sina(code) -> str:
        """转换为新浪格式 sh600419 / sz002732"""
        code = str(code).zfill(6)
        prefix = "sh" if code.startswith("6") else "sz"
        return f"{prefix}{code}"

    def _to_baostock_code(self, code: str) -> str:
        """600xxx → sh.600xxx, 00xxxx/30xxxx → sz.00xxxx"""
        code = self._normalize_code(code)
        if code.startswith(('6', '9')):
            return f"sh.{code}"
        return f"sz.{code}"

    def _query_baostock(self, code: str, start_date: str, end_date: str,
                        fields: str = "date,code,close,pbMRQ,peTTM,psTTM",
                        frequency: str = "d") -> pd.DataFrame:
        """BaoStock 通用查询（自动 login/logout）"""
        bs_code = self._to_baostock_code(code)
        try:
            bs.login()
            rs = bs.query_history_k_data_plus(
                bs_code, fields,
                start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag="2",
            )
            rows = []
            while (rs.error_code == '0') and rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=rs.fields)
            # 把数值列转 float（BaoStock 返回全是字符串）
            numeric_cols = [c for c in df.columns if c not in ("date", "code")]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.warning(f"BaoStock query failed for {code}: {e}")
            return pd.DataFrame()
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    def _rate_limit(self, seconds: float = 0.5):
        """简单限速，避免被封"""
        time.sleep(seconds)

    def get_stock_list(self, industry: str) -> pd.DataFrame:
        """通过东方财富概念板块获取行业股票列表（保留兼容性）"""
        # 新策略中不再使用此方法（固定股票池），但保留接口兼容
        logger.info(f"get_stock_list called for {industry} - fixed universe mode, skipping")
        return pd.DataFrame(columns=["code", "name", "industry"])

    def get_daily_quotes(
        self, code, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取个股日线行情"""
        code = self._normalize_code(code)
        cache_key = f"daily:{code}:{start_date}:{end_date}"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            self._rate_limit(0.3)
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
            })
            df["date"] = pd.to_datetime(df["date"])
            result = df[["date", "open", "high", "low", "close", "volume", "amount"]]
            self.cache.set(cache_key, result, expire_hours=24)
            return result
        except Exception as e:
            logger.warning(f"Failed to get daily quotes for {code}: {e}")
            return pd.DataFrame()

    def get_financial_indicators(
        self, code, date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取个股财务指标（来自同花顺/新浪）"""
        code = self._normalize_code(code)
        cache_key = f"financial:{code}"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            self._rate_limit(0.5)
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
            if df is None or df.empty:
                self._rate_limit(0.5)
                df = ak.stock_financial_analysis_indicator(symbol=code)

            if df is not None and not df.empty:
                self.cache.set(cache_key, df, expire_hours=168)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to get financial indicators for {code}: {e}")
            return pd.DataFrame()

    def get_realtime_quotes(self, codes: list) -> pd.DataFrame:
        """获取全市场实时行情（含PE、PB、市值等）"""
        codes = [self._normalize_code(c) for c in codes]
        cache_key = "realtime:all"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            cached["code"] = cached["code"].apply(self._normalize_code)
            return cached[cached["code"].isin(codes)]

        try:
            self._rate_limit()
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                "代码": "code",
                "名称": "name",
                "最新价": "price",
                "市盈率-动态": "pe_ttm",
                "市净率": "pb",
                "总市值": "market_cap",
                "流通市值": "float_market_cap",
                "成交量": "volume",
                "成交额": "amount",
                "60日涨跌幅": "change_60d",
            })
            df["code"] = df["code"].apply(self._normalize_code)
            self.cache.set(cache_key, df, expire_hours=4)
            return df[df["code"].isin(codes)]
        except Exception as e:
            logger.warning(f"Failed to get realtime quotes: {e}")
            return pd.DataFrame()

    def get_cashflow_data(self, code) -> pd.DataFrame:
        """获取现金流量表数据"""
        code = self._normalize_code(code)
        cache_key = f"cashflow:{code}"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            self._rate_limit(0.5)
            sina_code = self._code_to_sina(code)
            df = ak.stock_financial_report_sina(stock=sina_code, symbol="现金流量表")
            if df is not None and not df.empty:
                self.cache.set(cache_key, df, expire_hours=168)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to get cashflow for {code}: {e}")
            return pd.DataFrame()

    def get_index_daily(
        self, index_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取指数日线（用于回测基准）"""
        cache_key = f"index:{index_code}:{start_date}:{end_date}"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            self._rate_limit()
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            if df is None or df.empty:
                self._rate_limit()
                df = ak.index_zh_a_hist(
                    symbol=index_code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                )
            if df is None or df.empty:
                return pd.DataFrame()

            col_map = {
                "date": "date", "日期": "date",
                "open": "open", "开盘": "open",
                "high": "high", "最高": "high",
                "low": "low", "最低": "low",
                "close": "close", "收盘": "close",
                "volume": "volume", "成交量": "volume",
            }
            df = df.rename(columns=col_map)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            self.cache.set(cache_key, df, expire_hours=24)
            return df
        except Exception as e:
            logger.warning(f"Failed to get index data for {index_code}: {e}")
            return pd.DataFrame()

    def get_stock_info(self, code) -> dict:
        """获取个股基本信息（上市日期、是否ST等）"""
        code = self._normalize_code(code)
        try:
            self._rate_limit(0.3)
            df = ak.stock_individual_info_em(symbol=code)
            if df is None or df.empty:
                return {}
            info = dict(zip(df["item"], df["value"]))
            return info
        except Exception as e:
            logger.warning(f"Failed to get stock info for {code}: {e}")
            return {}

    # ===== 新增接口：PB历史数据 =====

    def get_pb_history(self, code, years: int = 5) -> pd.DataFrame:
        """获取个股PB历史数据（用于估值分析）- BaoStock 数据源

        Args:
            code: 股票代码
            years: 获取近几年数据

        Returns:
            DataFrame with columns [date, close, pb]
        """
        code = self._normalize_code(code)
        cache_key = f"pb_history:{code}:{years}y"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

            self._rate_limit(0.3)
            df = self._query_baostock(
                code, start_date, end_date,
                fields="date,code,close,pbMRQ",
                frequency="d",
            )

            if df.empty:
                logger.warning(f"BaoStock returned no data for {code}")
                return pd.DataFrame()

            # 列名映射
            df = df.rename(columns={"pbMRQ": "pb"})
            # 过滤掉 pb 为空的行
            df = df.dropna(subset=["pb"])
            df = df[df["pb"] > 0]

            if df.empty:
                return pd.DataFrame()

            result = df[["date", "close", "pb"]].copy()
            self.cache.set(cache_key, result, expire_hours=24)
            logger.info(f"PB history for {code}: {len(result)} days, "
                       f"PB range [{result['pb'].min():.2f}, {result['pb'].max():.2f}]")
            return result

        except Exception as e:
            logger.warning(f"Failed to get PB history for {code}: {e}")
            return pd.DataFrame()

    # ===== 新增接口：板块成分股 =====

    def get_board_constituents(self, board_name: str, board_type: str = "industry") -> pd.DataFrame:
        """获取板块成分股列表

        Args:
            board_name: 板块名称，如 "锂电池"、"光伏设备"
            board_type: "industry" (行业板块) 或 "concept" (概念板块)

        Returns:
            DataFrame with columns [code, name]
        """
        cache_key = f"board_cons:{board_type}:{board_name}"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            self._rate_limit(0.5)
            if board_type == "concept":
                df = ak.stock_board_concept_cons_em(symbol=board_name)
            else:
                df = ak.stock_board_industry_cons_em(symbol=board_name)

            if df is None or df.empty:
                logger.warning(f"No constituents found for board '{board_name}' (type={board_type})")
                return pd.DataFrame(columns=["code", "name"])

            df = df.rename(columns={"代码": "code", "名称": "name"})
            df["code"] = df["code"].apply(self._normalize_code)
            result = df[["code", "name"]].copy()
            # 板块成分股变化不频繁，缓存 7 天
            self.cache.set(cache_key, result, expire_hours=168)
            logger.info(f"Board '{board_name}' ({board_type}): {len(result)} stocks")
            return result
        except Exception as e:
            logger.warning(f"Failed to get board constituents for '{board_name}': {e}")
            return pd.DataFrame(columns=["code", "name"])

    # ===== 新增接口：长期PB+PS历史（月频） =====

    def get_pb_history_long(self, code: str, years: int = 12) -> pd.DataFrame:
        """获取个股长期PB+PS历史数据（月频）- BaoStock 数据源

        Args:
            code: 股票代码
            years: 获取近几年数据（默认12年覆盖多轮周期）

        Returns:
            DataFrame with columns [date, close, pb, ps]
        """
        code = self._normalize_code(code)
        cache_key = f"pb_long:{code}:{years}y"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

            # 1. 从 BaoStock 日频拉取 PB/PS
            self._rate_limit(0.3)
            df = self._query_baostock(
                code, start_date, end_date,
                fields="date,code,close,pbMRQ,psTTM",
                frequency="d",
            )

            if df.empty:
                logger.warning(f"BaoStock returned no long data for {code}")
                return pd.DataFrame()

            # 列名映射
            df = df.rename(columns={"pbMRQ": "pb", "psTTM": "ps"})
            df = df.dropna(subset=["pb"])
            df = df[df["pb"] > 0]

            if df.empty:
                return pd.DataFrame()

            # 2. 日频 → 月频 resample（取月末值）
            df = df.set_index("date")
            cols_to_resample = ["close", "pb"]
            if "ps" in df.columns:
                cols_to_resample.append("ps")

            monthly = df[cols_to_resample].resample("ME").last().dropna(subset=["pb"])
            monthly = monthly.reset_index()

            # 确保列完整
            for col in ["close", "pb", "ps"]:
                if col not in monthly.columns:
                    monthly[col] = None

            result = monthly[["date", "close", "pb", "ps"]].copy()
            self.cache.set(cache_key, result, expire_hours=168)
            logger.info(f"PB long history for {code}: {len(result)} months, "
                       f"PB range [{result['pb'].min():.2f}, {result['pb'].max():.2f}]")
            return result

        except Exception as e:
            logger.warning(f"Failed to get PB long history for {code}: {e}")
            return pd.DataFrame()

    def get_commodity_price(self, commodity: str) -> pd.DataFrame:
        """获取大宗商品价格走势

        Args:
            commodity: 商品名称，支持 "lithium_carbonate"(碳酸锂),
                       "phosphate"(磷铵), "mdi"(MDI)

        Returns:
            DataFrame with columns [date, price, change_pct]
        """
        cache_key = f"commodity:{commodity}"
        cached = self.cache.get(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        try:
            self._rate_limit(0.5)

            if commodity == "lithium_carbonate":
                # 碳酸锂期货价格
                try:
                    df = ak.futures_main_sina(symbol="LC0", start_date="20230101",
                                             end_date=datetime.now().strftime("%Y%m%d"))
                    if df is not None and not df.empty:
                        df = df.rename(columns={"日期": "date", "收盘价": "price"})
                        df["date"] = pd.to_datetime(df["date"])
                        df["change_pct"] = df["price"].pct_change()
                        result = df[["date", "price", "change_pct"]].dropna()
                        self.cache.set(cache_key, result, expire_hours=24)
                        return result
                except Exception:
                    pass

                # 备用：现货价格
                try:
                    df = ak.spot_hist_sge(symbol="碳酸锂")
                    if df is not None and not df.empty:
                        self.cache.set(cache_key, df, expire_hours=24)
                        return df
                except Exception:
                    pass

            elif commodity == "phosphate":
                # 磷酸一铵/磷酸二铵现货价格
                try:
                    df = ak.futures_main_sina(symbol="SA0", start_date="20230101",
                                             end_date=datetime.now().strftime("%Y%m%d"))
                    if df is not None and not df.empty:
                        df = df.rename(columns={"日期": "date", "收盘价": "price"})
                        df["date"] = pd.to_datetime(df["date"])
                        df["change_pct"] = df["price"].pct_change()
                        result = df[["date", "price", "change_pct"]].dropna()
                        self.cache.set(cache_key, result, expire_hours=24)
                        return result
                except Exception:
                    pass

            elif commodity == "mdi":
                # MDI 没有期货，尝试从化工品现货获取
                try:
                    df = ak.spot_hist_sge(symbol="MDI")
                    if df is not None and not df.empty:
                        self.cache.set(cache_key, df, expire_hours=24)
                        return df
                except Exception:
                    pass

            logger.warning(f"No commodity data found for {commodity}")
            return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Failed to get commodity price for {commodity}: {e}")
            return pd.DataFrame()
