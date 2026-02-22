"""周期底部龙头策略 — 月度执行脚本

流程：
1. 获取6只股票实时行情（价格、PB、市值）
2. 获取PB历史数据
3. 计算估值状态
4. 检查周期拐点信号
5. 生成操作建议（买入/卖出/止损）
6. 保存结果
"""

import io
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

from loguru import logger

# Fix Windows terminal Chinese encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}")

from quant.data.akshare_provider import AKShareProvider
from quant.data.cache import DataCache
from quant.data.universe import UniverseBuilder
from quant.factors.valuation import ValuationEngine
from quant.factors.cycle_analyzer import CycleAnalyzer
from quant.portfolio.selector import StockSelector
from quant.portfolio.constructor import PortfolioConstructor


def load_config() -> dict:
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(universe, config):
    """保存分析结果"""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    # 保存当前分析结果
    universe.to_csv(output_dir / "latest_analysis.csv", index=False)
    universe.to_csv(output_dir / f"analysis_{date_str}.csv", index=False)

    logger.info(f"Results saved to {output_dir}")


def main():
    config = load_config()
    date_str = datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info(f"周期底部龙头投资策略 — 月度分析 ({date_str})")
    logger.info("=" * 60)

    # ========== Step 1: 构建固定股票池 + 获取实时行情 ==========
    logger.info("\nStep 1: 构建股票池，获取实时行情...")
    cache = DataCache(config.get("cache", {}).get("db_path", "data/cache.db"))
    provider = AKShareProvider(cache=cache)
    builder = UniverseBuilder(provider=provider, config=config)
    universe = builder.build_universe()

    if universe.empty:
        logger.error("股票池为空！")
        sys.exit(1)

    logger.info(f"股票池: {len(universe)} 只股票, "
                f"{universe['industry'].nunique()} 个行业")

    # ========== Step 2: PB估值分析 ==========
    logger.info("\nStep 2: PB估值分析...")
    valuation = ValuationEngine(provider=provider, config=config)
    universe = valuation.compute_all(universe)

    # ========== Step 3: 周期分析 ==========
    logger.info("\nStep 3: 行业周期分析...")
    cycle = CycleAnalyzer(provider=provider, config=config)
    universe = cycle.analyze_all(universe)

    # ========== Step 4: 生成操作建议 ==========
    logger.info("\nStep 4: 生成操作建议...")
    selector = StockSelector(config)
    universe = selector.generate_advice(universe)

    # ========== Step 5: 仓位管理 ==========
    logger.info("\nStep 5: 检查仓位和止损...")
    constructor = PortfolioConstructor(config)
    portfolio_state = constructor.load_portfolio_state()
    if portfolio_state:
        actions = constructor.generate_actions(universe, portfolio_state)
        # 用操作建议覆盖原始建议
        for _, action in actions.iterrows():
            if action.get("advice") in ("止损", "卖出"):
                mask = universe["code"] == action["code"]
                universe.loc[mask, "advice"] = action["advice"]
                universe.loc[mask, "advice_detail"] = action["advice_detail"]

    # ========== Step 6: 输出结果 ==========
    logger.info("\n" + "=" * 60)
    logger.info(f"月度分析报告 ({date_str})")
    logger.info("=" * 60)

    # 分行业输出
    from quant.utils.constants import IND_CN
    for industry in universe["industry"].unique():
        ind_cn = IND_CN.get(industry, industry)
        ind_df = universe[universe["industry"] == industry]
        ind_config = config.get("industries", {}).get(industry, {})

        logger.info(f"\n--- {ind_cn} (目标仓位 {ind_config.get('target_weight', 0):.0%}) ---")
        logger.info(f"  周期状态: {ind_config.get('cycle_status', '未知')}")

        for _, row in ind_df.iterrows():
            pb_str = f"PB={row.get('pb', 'N/A')}"
            if isinstance(row.get('pb'), (int, float)):
                pb_str = f"PB={row['pb']:.2f}"

            peak_str = ""
            if isinstance(row.get('pb_peak'), (int, float)):
                peak_str = f" (高点={row['pb_peak']:.1f})"

            status = row.get('valuation_status', 'N/A')
            advice = row.get('advice', 'N/A')
            detail = row.get('advice_detail', '')
            upside = row.get('potential_upside')
            upside_str = f", 上涨空间={upside:.0%}" if isinstance(upside, (int, float)) else ""

            logger.info(
                f"  {row['name']}({row['code']}): {pb_str}{peak_str}, "
                f"估值={status}{upside_str}"
            )
            logger.info(f"    操作建议: [{advice}] {detail}")

    # 汇总
    logger.info(f"\n--- 操作汇总 ---")
    for advice in ["买入", "减仓", "卖出", "止损", "持有", "观察"]:
        subset = universe[universe["advice"] == advice]
        if not subset.empty:
            names = ", ".join(subset["name"].tolist())
            logger.info(f"  {advice}: {names}")

    # 保存
    save_results(universe, config)

    logger.info(f"\n完成! 使用 'streamlit run app.py' 查看交互式看板。")


if __name__ == "__main__":
    main()
