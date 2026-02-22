"""投资建议生成（固定6只龙头，不做Top-N筛选）"""

import pandas as pd
from loguru import logger


class StockSelector:
    """根据估值状态和周期位置，生成买入/持有/卖出建议"""

    def __init__(self, config: dict):
        self.config = config
        self.buy_cfg = config.get("buy_strategy", {})
        self.sell_cfg = config.get("sell_strategy", {})
        self.stop_loss_cfg = config.get("stop_loss", {})

    def generate_advice(self, universe: pd.DataFrame) -> pd.DataFrame:
        """为每只股票生成操作建议

        Args:
            universe: 含估值和周期分析结果的 DataFrame

        Returns:
            添加 advice, advice_detail 列的 DataFrame
        """
        universe = universe.copy()
        universe["advice"] = "持有"
        universe["advice_detail"] = ""
        universe["advice_priority"] = 0  # 优先级，越高越应该操作

        for idx, row in universe.iterrows():
            advice, detail, priority = self._evaluate_single(row)
            universe.at[idx, "advice"] = advice
            universe.at[idx, "advice_detail"] = detail
            universe.at[idx, "advice_priority"] = priority

        # 按优先级排序
        universe = universe.sort_values("advice_priority", ascending=False)

        logger.info("Investment advice generated:")
        for _, row in universe.iterrows():
            logger.info(f"  {row['name']}({row['code']}): {row['advice']} - {row['advice_detail']}")

        return universe

    def _evaluate_single(self, row) -> tuple[str, str, int]:
        """评估单只股票的操作建议

        Returns:
            (advice, detail, priority)
        """
        valuation_status = row.get("valuation_status", "数据不足")
        cycle_position = row.get("cycle_position", "未知")
        target_weight = row.get("target_weight", 0)
        pb_ratio = row.get("pb_ratio")
        potential_upside = row.get("potential_upside")

        # 备选股票不建议操作
        if row.get("role") == "备选替代":
            return "观察", "备选标的，暂不操作", 0

        # 无目标仓位的不操作
        if target_weight <= 0:
            return "观察", "未分配目标仓位", 0

        # ===== 买入信号 =====
        if valuation_status in ("严重低估", "低估"):
            if cycle_position in ("下行尾段", "磨底", "反转初期"):
                priority = 90 if valuation_status == "严重低估" else 70

                upside_str = f"潜在上涨空间{potential_upside:.0%}" if pd.notna(potential_upside) else ""
                ratio_str = f"PB仅为高点的{pb_ratio:.0%}" if pd.notna(pb_ratio) else ""

                detail = f"估值{valuation_status}+周期{cycle_position}，{ratio_str}，{upside_str}"
                return "买入", detail, priority

            elif cycle_position == "下行":
                return "观察", f"估值{valuation_status}但周期仍在下行，等待确认底部", 30

        # ===== 持有/加仓信号 =====
        if valuation_status == "合理":
            if cycle_position in ("反转初期", "上行"):
                return "持有", f"估值合理+周期{cycle_position}，继续持有", 40
            else:
                return "持有", f"估值合理，等待周期拐点", 30

        # ===== 卖出信号 =====
        if valuation_status == "高估":
            sell_ratio = self.sell_cfg.get("pb_target_sell", 0.50)
            return "减仓", f"PB已达高估区域，建议减仓{sell_ratio:.0%}", 80

        # 默认
        return "持有", "维持现有仓位", 20
