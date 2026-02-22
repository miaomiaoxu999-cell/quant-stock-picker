"""周期分析引擎 — 替代原 composite.py"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant.data.akshare_provider import AKShareProvider
from quant.utils.constants import INDUSTRY_MAP


class CycleAnalyzer:
    """行业周期分析（判断周期位置、检查拐点信号）"""

    def __init__(
        self,
        provider: Optional[AKShareProvider] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider or AKShareProvider()
        self.config = config or {}
        self.industries = self.config.get("industries", {})

    def analyze_all(self, universe: pd.DataFrame) -> pd.DataFrame:
        """对所有行业进行周期分析

        Args:
            universe: 含 industry 列的 DataFrame

        Returns:
            添加周期分析列的 DataFrame:
            - cycle_position: 周期位置（下行尾段/磨底/反转初期/上行）
            - cycle_score: 周期评分 0-1
            - turning_signal_count: 已确认拐点信号数
            - turning_signals_detail: 拐点信号详情
        """
        cycle_data = {}
        for ind_key in universe["industry"].unique():
            analysis = self.analyze_cycle_position(ind_key)
            cycle_data[ind_key] = analysis

        # 将周期分析结果合并到 universe
        universe = universe.copy()
        universe["cycle_position"] = universe["industry"].map(
            lambda x: cycle_data.get(x, {}).get("position", "未知")
        )
        universe["cycle_score"] = universe["industry"].map(
            lambda x: cycle_data.get(x, {}).get("score", 0.5)
        )
        universe["turning_signal_count"] = universe["industry"].map(
            lambda x: cycle_data.get(x, {}).get("signal_count", 0)
        )
        universe["cycle_probability"] = universe["industry"].map(
            lambda x: cycle_data.get(x, {}).get("probability", 0.5)
        )

        logger.info("Cycle analysis complete:")
        for ind_key, analysis in cycle_data.items():
            ind_name = self.industries.get(ind_key, {}).get("name", ind_key)
            logger.info(
                f"  [{ind_name}] position={analysis.get('position')}, "
                f"score={analysis.get('score', 0):.2f}, "
                f"signals={analysis.get('signal_count', 0)}/{analysis.get('total_signals', 0)}"
            )

        return universe

    def analyze_cycle_position(self, industry: str) -> dict:
        """判断行业周期位置

        Args:
            industry: 行业key (lithium/phosphorus/basic_chem)

        Returns:
            dict with keys: position, score, signal_count, total_signals, probability, details
        """
        ind_config = self.industries.get(industry, {})
        cycle_status = ind_config.get("cycle_status", "未知")
        cycle_probability = ind_config.get("cycle_probability", 0.5)
        turning_signals = ind_config.get("turning_signals", [])

        # 从配置中读取周期状态（手动判断为主）
        position = self._parse_cycle_status(cycle_status)

        # 周期评分：基于周期位置和反转概率
        score = self._position_to_score(position, cycle_probability)

        # 检查拐点信号（配置中的定性信号）
        signal_check = self.check_turning_signals(industry)

        return {
            "position": position,
            "score": score,
            "probability": cycle_probability,
            "signal_count": signal_check["confirmed"],
            "total_signals": signal_check["total"],
            "signals": signal_check["details"],
            "commodity_trend": self._get_commodity_trend(industry),
        }

    def _parse_cycle_status(self, status: str) -> str:
        """将配置中的周期状态文字解析为标准位置"""
        status_lower = status.lower()
        if "反转" in status or "回升" in status:
            return "反转初期"
        elif "上行" in status or "上升" in status:
            return "上行"
        elif "下行尾" in status or "即将" in status:
            return "下行尾段"
        elif "底部" in status or "磨底" in status or "低位" in status:
            return "磨底"
        elif "下行" in status or "下降" in status:
            return "下行"
        elif "中低" in status:
            return "磨底"
        else:
            return "未知"

    @staticmethod
    def _position_to_score(position: str, probability: float) -> float:
        """周期位置转评分（越接近底部越高分，因为是买入机会）"""
        base_scores = {
            "下行尾段": 0.80,
            "磨底": 0.90,
            "反转初期": 0.95,
            "上行": 0.60,
            "下行": 0.40,
            "未知": 0.50,
        }
        base = base_scores.get(position, 0.50)
        # 用反转概率加权
        return base * probability

    def check_turning_signals(self, industry: str) -> dict:
        """检查拐点确认信号

        Returns:
            dict with keys: confirmed, total, details
        """
        ind_config = self.industries.get(industry, {})
        turning_signals = ind_config.get("turning_signals", [])

        # 目前拐点信号为定性描述，需要手动确认
        # 这里返回配置中的信号列表，前端显示供用户判断
        details = []
        for signal in turning_signals:
            details.append({
                "description": signal,
                "status": "待确认",  # 需要用户手动确认
            })

        return {
            "confirmed": 0,  # 自动确认的数量
            "total": len(turning_signals),
            "details": details,
        }

    def _get_commodity_trend(self, industry: str) -> dict:
        """获取对应大宗商品价格趋势"""
        ind_info = INDUSTRY_MAP.get(industry, {})
        commodity_key = ind_info.get("commodity_key")
        if not commodity_key:
            return {}

        try:
            commodity_df = self.provider.get_commodity_price(commodity_key)
            if commodity_df.empty:
                return {"status": "数据获取失败"}

            if "price" in commodity_df.columns:
                latest_price = commodity_df["price"].iloc[-1]
                # 近30日变化
                if len(commodity_df) >= 30:
                    price_30d_ago = commodity_df["price"].iloc[-30]
                    change_30d = (latest_price - price_30d_ago) / price_30d_ago
                else:
                    change_30d = None

                return {
                    "latest_price": latest_price,
                    "change_30d": change_30d,
                    "commodity_name": ind_info.get("commodity", ""),
                }
        except Exception as e:
            logger.debug(f"Commodity trend failed for {industry}: {e}")

        return {"status": "暂无数据"}

    def compute_industry_score(self, industry: str) -> dict:
        """计算行业综合评分（周期位置 + 估值 + 供需）

        Returns:
            dict with score details
        """
        cycle = self.analyze_cycle_position(industry)
        commodity = self._get_commodity_trend(industry)

        # 综合评分 = 周期评分(60%) + 商品趋势(40%)
        cycle_score = cycle.get("score", 0.5)

        commodity_score = 0.5  # 默认中性
        change_30d = commodity.get("change_30d")
        if change_30d is not None:
            # 价格下跌 = 可能接近底部 = 更好的买入机会
            # 但也可能是需求萎缩的信号，需要结合周期位置判断
            if cycle.get("position") in ("下行尾段", "磨底"):
                # 底部区域，价格企稳或小幅回升是积极信号
                if -0.05 <= change_30d <= 0.10:
                    commodity_score = 0.70
                elif change_30d > 0.10:
                    commodity_score = 0.60  # 涨太快可能是短期反弹
                else:
                    commodity_score = 0.40  # 还在跌
            else:
                commodity_score = 0.5 + change_30d  # 简单线性映射

        total_score = cycle_score * 0.6 + commodity_score * 0.4

        return {
            "total_score": total_score,
            "cycle_score": cycle_score,
            "commodity_score": commodity_score,
            "position": cycle.get("position"),
            "probability": cycle.get("probability"),
        }
