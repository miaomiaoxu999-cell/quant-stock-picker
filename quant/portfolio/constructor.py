"""仓位管理与操作计划生成"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class PortfolioConstructor:
    """周期底部龙头策略的仓位管理"""

    def __init__(self, config: dict):
        self.config = config
        self.industries = config.get("industries", {})
        self.cash_reserve = config.get("cash_reserve", 0.10)
        self.buy_cfg = config.get("buy_strategy", {})
        self.sell_cfg = config.get("sell_strategy", {})
        self.stop_loss_cfg = config.get("stop_loss", {})

    def build_target_portfolio(self) -> pd.DataFrame:
        """按策略固定比例分配目标仓位

        Returns:
            DataFrame with [code, name, industry, target_weight]
        """
        rows = []
        for ind_key, ind_config in self.industries.items():
            for stock in ind_config.get("stocks", []):
                weight = stock.get("weight", 0)
                if weight > 0:
                    rows.append({
                        "code": str(stock["code"]).zfill(6),
                        "name": stock["name"],
                        "industry": ind_key,
                        "target_weight": weight,
                    })

        target = pd.DataFrame(rows)
        total_weight = target["target_weight"].sum()
        logger.info(
            f"Target portfolio: {len(target)} stocks, "
            f"total weight={total_weight:.0%}, "
            f"cash reserve={self.cash_reserve:.0%}"
        )
        return target

    def compute_buy_plan(
        self,
        code: str,
        name: str,
        target_weight: float,
        current_price: float,
        avg_cost: Optional[float] = None,
        current_holding_pct: float = 0.0,
    ) -> dict:
        """生成分批买入计划

        Args:
            code: 股票代码
            name: 股票名称
            target_weight: 目标仓位比例
            current_price: 当前价格
            avg_cost: 当前持仓均价（无持仓则为None）
            current_holding_pct: 当前实际仓位比例

        Returns:
            买入计划dict
        """
        initial_ratio = self.buy_cfg.get("initial_ratio", 0.30)
        dip_buy_threshold = self.buy_cfg.get("dip_buy_threshold", 0.08)
        max_chase_pct = self.buy_cfg.get("max_chase_pct", 0.10)

        plan = {
            "code": code,
            "name": name,
            "target_weight": target_weight,
            "current_price": current_price,
            "current_holding_pct": current_holding_pct,
            "steps": [],
        }

        if current_holding_pct >= target_weight:
            plan["status"] = "已达目标仓位"
            return plan

        remaining = target_weight - current_holding_pct

        if avg_cost is not None and current_price > avg_cost * (1 + max_chase_pct):
            plan["status"] = f"当前价格已高于成本{max_chase_pct:.0%}以上，不追涨"
            return plan

        # 首次建仓
        if current_holding_pct == 0:
            first_buy = remaining * initial_ratio
            plan["steps"].append({
                "action": "首次建仓",
                "weight": first_buy,
                "price_condition": f"当前价格 {current_price:.2f}",
            })
            remaining -= first_buy

            # 补仓计划
            step = 1
            buy_price = current_price
            while remaining > 0.001:
                buy_price = buy_price * (1 - dip_buy_threshold)
                buy_weight = min(remaining, first_buy)
                plan["steps"].append({
                    "action": f"第{step}次补仓",
                    "weight": buy_weight,
                    "price_condition": f"下跌至 {buy_price:.2f}（-{dip_buy_threshold*step:.0%}）",
                })
                remaining -= buy_weight
                step += 1
                if step > 5:
                    break
        else:
            # 已有持仓，计算补仓
            plan["steps"].append({
                "action": "补仓",
                "weight": remaining,
                "price_condition": f"当前价格合适时逐步加仓",
            })

        plan["status"] = f"计划分{len(plan['steps'])}步建仓"
        return plan

    def check_sell_signals(
        self,
        code: str,
        name: str,
        current_price: float,
        avg_cost: float,
        current_pb: float,
        peak_pb: float,
    ) -> dict:
        """检查卖出条件

        Returns:
            卖出信号dict
        """
        double_sell_ratio = self.sell_cfg.get("double_sell_ratio", 0.50)
        pb_target_sell = self.sell_cfg.get("pb_target_sell", 0.50)

        signal = {
            "code": code,
            "name": name,
            "should_sell": False,
            "sell_ratio": 0.0,
            "reason": "",
        }

        if avg_cost <= 0:
            return signal

        profit_pct = (current_price - avg_cost) / avg_cost

        # 条件1：翻倍卖50%
        if profit_pct >= 1.0:
            signal["should_sell"] = True
            signal["sell_ratio"] = double_sell_ratio
            signal["reason"] = f"已翻倍（涨幅{profit_pct:.0%}），建议卖出{double_sell_ratio:.0%}"
            return signal

        # 条件2：PB到高点卖剩余
        if current_pb > 0 and peak_pb > 0:
            pb_ratio = current_pb / peak_pb
            if pb_ratio >= 0.90:
                signal["should_sell"] = True
                signal["sell_ratio"] = pb_target_sell
                signal["reason"] = f"PB已达高点{pb_ratio:.0%}，建议卖出{pb_target_sell:.0%}"
                return signal

        return signal

    def check_stop_loss(
        self,
        code: str,
        name: str,
        current_price: float,
        avg_cost: float,
    ) -> dict:
        """检查止损条件

        Returns:
            止损信号dict
        """
        l1_drawdown = self.stop_loss_cfg.get("level_1_drawdown", 0.15)
        l1_action = self.stop_loss_cfg.get("level_1_action", 0.50)
        l2_drawdown = self.stop_loss_cfg.get("level_2_drawdown", 0.25)
        l2_action = self.stop_loss_cfg.get("level_2_action", 0.00)

        signal = {
            "code": code,
            "name": name,
            "stop_loss_triggered": False,
            "level": 0,
            "action": "",
            "drawdown": 0.0,
        }

        if avg_cost <= 0:
            return signal

        drawdown = (avg_cost - current_price) / avg_cost
        signal["drawdown"] = drawdown

        if drawdown >= l2_drawdown:
            signal["stop_loss_triggered"] = True
            signal["level"] = 2
            signal["action"] = f"回撤{drawdown:.0%}超过{l2_drawdown:.0%}，全部止损"
        elif drawdown >= l1_drawdown:
            signal["stop_loss_triggered"] = True
            signal["level"] = 1
            signal["action"] = f"回撤{drawdown:.0%}超过{l1_drawdown:.0%}，减仓{1-l1_action:.0%}"

        return signal

    def generate_actions(self, universe: pd.DataFrame, portfolio_state: dict) -> pd.DataFrame:
        """生成本期操作建议

        Args:
            universe: 含估值和周期分析结果的 DataFrame
            portfolio_state: 当前持仓状态 {code: {avg_cost, shares, weight}}

        Returns:
            操作建议 DataFrame
        """
        actions = []
        for _, row in universe.iterrows():
            code = row["code"]
            name = row["name"]
            current_price = row.get("price", 0)
            current_pb = row.get("pb", 0)
            peak_pb = row.get("pb_peak", 0)
            target_weight = row.get("target_weight", 0)

            # 获取当前持仓信息
            holding = portfolio_state.get(code, {})
            avg_cost = holding.get("avg_cost", 0)
            current_pct = holding.get("weight", 0)

            action = {
                "code": code,
                "name": name,
                "industry": row.get("industry", ""),
                "current_price": current_price,
                "avg_cost": avg_cost,
                "target_weight": target_weight,
                "current_weight": current_pct,
                "advice": row.get("advice", "持有"),
                "advice_detail": row.get("advice_detail", ""),
            }

            # 检查止损
            if avg_cost > 0:
                stop = self.check_stop_loss(code, name, current_price, avg_cost)
                if stop["stop_loss_triggered"]:
                    action["advice"] = "止损"
                    action["advice_detail"] = stop["action"]
                    action["stop_loss_level"] = stop["level"]

                # 检查卖出
                if pd.notna(current_pb) and pd.notna(peak_pb) and peak_pb > 0:
                    sell = self.check_sell_signals(
                        code, name, current_price, avg_cost, current_pb, peak_pb
                    )
                    if sell["should_sell"]:
                        action["advice"] = "卖出"
                        action["advice_detail"] = sell["reason"]

            actions.append(action)

        return pd.DataFrame(actions)

    @staticmethod
    def load_portfolio_state(path: str = "data/portfolio_state.json") -> dict:
        """加载持仓状态"""
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def save_portfolio_state(state: dict, path: str = "data/portfolio_state.json"):
        """保存持仓状态"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
