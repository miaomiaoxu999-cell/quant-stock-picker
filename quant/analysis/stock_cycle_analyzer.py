"""个股周期关联分析引擎

从已完成的周期分析导入板块 → 拉取板块 top N 个股 → 获取长期 PB/股价
→ 分析个股与周期的关联性 → 排名出最具上行空间的股票
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from quant.data.akshare_provider import AKShareProvider
from quant.data.cache import DataCache

# ==================== 评分权重预设 ====================

WEIGHT_PRESETS = {
    "default": {"upside": 0.40, "alignment": 0.30, "valuation": 0.15, "momentum": 0.15},
    "cycle_bottom": {"upside": 0.35, "alignment": 0.35, "valuation": 0.20, "momentum": 0.10},
    "cycle_top": {"upside": 0.20, "alignment": 0.25, "valuation": 0.30, "momentum": 0.25},
}


def _select_weights(cycle_position: str) -> dict:
    """根据行业周期位置自动选择评分权重方案"""
    bottom_keywords = ["磨底", "下行尾段", "筑底", "底部"]
    top_keywords = ["上行末期", "过热", "顶部"]
    if any(k in cycle_position for k in bottom_keywords):
        return WEIGHT_PRESETS["cycle_bottom"]
    if any(k in cycle_position for k in top_keywords):
        return WEIGHT_PRESETS["cycle_top"]
    return WEIGHT_PRESETS["default"]


# ==================== 板块名映射 ====================
# cycle_analysis.json 中的板块名 → 东方财富行业/概念板块名列表（取并集去重）

SECTOR_BOARD_MAP = {
    "锂电池/锂盐": ["锂电池", "能源金属"],
    "光伏": ["光伏设备"],
    "磷化工": ["磷化工"],
    "基础化工": ["化学制品"],
    "石油开采": ["油气开采Ⅱ", "油气开采Ⅲ", "石油石化"],
    "黄金": ["贵金属", "黄金"],
}

# ==================== 自动板块匹配 ====================

def _fetch_all_board_names() -> dict[str, str]:
    """获取东方财富所有行业+概念板块名，结果缓存 24h

    Returns:
        {"板块名": "industry"|"concept", ...}
    """
    import akshare as ak

    cache = DataCache()
    cache_key = "board_names:all"
    cached = cache.get(cache_key)
    if cached is not None and not cached.empty:
        return dict(zip(cached["name"], cached["type"]))

    board_map: dict[str, str] = {}

    # 行业板块
    try:
        industry_df = ak.stock_board_industry_name_em()
        if industry_df is not None and not industry_df.empty:
            col = "板块名称" if "板块名称" in industry_df.columns else industry_df.columns[0]
            for name in industry_df[col]:
                board_map[str(name)] = "industry"
            logger.info(f"Loaded {len(industry_df)} industry boards")
    except Exception as e:
        logger.warning(f"Failed to fetch industry boards: {e}")

    # 概念板块
    try:
        concept_df = ak.stock_board_concept_name_em()
        if concept_df is not None and not concept_df.empty:
            col = "板块名称" if "板块名称" in concept_df.columns else concept_df.columns[0]
            for name in concept_df[col]:
                if str(name) not in board_map:
                    board_map[str(name)] = "concept"
            logger.info(f"Loaded {len(concept_df)} concept boards")
    except Exception as e:
        logger.warning(f"Failed to fetch concept boards: {e}")

    # 缓存为 DataFrame
    if board_map:
        df = pd.DataFrame(
            [{"name": k, "type": v} for k, v in board_map.items()]
        )
        cache.set(cache_key, df, expire_hours=24.0)
        logger.info(f"Cached {len(board_map)} board names")

    return board_map


def resolve_sector_boards(sector: str) -> list[str]:
    """自动匹配板块名 → 东方财富板块名列表

    优先级:
    1. SECTOR_BOARD_MAP 手动映射 → 直接返回
    2. 自动匹配（精确 → 子串 → 分词 → difflib 模糊）
    3. 匹配不到 → 返回空列表

    行业板块优先，最多返回 3 个。
    """
    # 1. 手动映射优先
    if sector in SECTOR_BOARD_MAP:
        return SECTOR_BOARD_MAP[sector]

    # 2. 自动匹配
    all_boards = _fetch_all_board_names()
    if not all_boards:
        logger.warning("No board names available for matching")
        return []

    industry_boards = [n for n, t in all_boards.items() if t == "industry"]
    concept_boards = [n for n, t in all_boards.items() if t == "concept"]
    all_names = list(all_boards.keys())

    matched_industry: list[str] = []
    matched_concept: list[str] = []

    def _add(name: str):
        btype = all_boards.get(name, "concept")
        if btype == "industry":
            if name not in matched_industry:
                matched_industry.append(name)
        else:
            if name not in matched_concept:
                matched_concept.append(name)

    # 按 "/" 拆分关键词（如 "锂电池/锂盐" → ["锂电池", "锂盐"]）
    keywords = [kw.strip() for kw in sector.split("/") if kw.strip()]

    for kw in keywords:
        # a. 精确匹配
        if kw in all_boards:
            _add(kw)
            continue

        # b. 子串匹配（关键词 >= 2 字）
        if len(kw) >= 2:
            for name in all_names:
                if kw in name:
                    _add(name)

        # c. difflib 模糊匹配
        close = difflib.get_close_matches(kw, all_names, n=3, cutoff=0.5)
        for name in close:
            _add(name)

    # 合并：行业优先，概念补充，最多 3 个
    result = matched_industry[:3]
    remaining = 3 - len(result)
    if remaining > 0:
        result.extend(matched_concept[:remaining])

    if result:
        logger.info(f"resolve_sector_boards('{sector}') → {result}")
    else:
        logger.warning(f"resolve_sector_boards('{sector}') → no match")

    return result


# ==================== 路径常量 ====================

_ROOT = Path(__file__).parent.parent.parent
CYCLE_DATA_PATH = _ROOT / "data" / "cycle_analysis.json"
PROFILES_DIR = _ROOT / "data" / "stock_profiles"
WATCHLIST_PATH = _ROOT / "data" / "watchlist.json"


def load_cycle_analysis() -> dict:
    """加载周期分析结果"""
    if CYCLE_DATA_PATH.exists():
        try:
            with open(CYCLE_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_analyzed_sectors() -> list[str]:
    """返回已完成周期分析的板块列表"""
    return list(load_cycle_analysis().keys())


# ==================== 核心分析类 ====================

class StockCycleAnalyzer:
    """个股与周期关联分析器"""

    def __init__(self, provider: Optional[AKShareProvider] = None):
        self.provider = provider or AKShareProvider(cache=DataCache())

    # ---------- 2a. 获取板块 top N 个股 ----------

    def get_sector_top_stocks(self, sector: str, top_n: int = 10) -> list[dict]:
        """获取板块内按市值排序的 top N 个股

        Args:
            sector: 周期分析中的板块名，如 "锂电池/锂盐"
            top_n: 返回数量

        Returns:
            [{code, name, market_cap, price, pb, pe_ttm}]

        Raises:
            ValueError: 板块名在 SECTOR_BOARD_MAP 中无映射
        """
        board_names = resolve_sector_boards(sector)
        if not board_names:
            raise ValueError(
                f"板块 '{sector}' 无法匹配到任何东方财富板块，可在 SECTOR_BOARD_MAP 中手动配置"
            )
        all_stocks = pd.DataFrame()

        for board_name in board_names:
            cons = self.provider.get_board_constituents(board_name)
            if not cons.empty:
                all_stocks = pd.concat([all_stocks, cons], ignore_index=True)

        if all_stocks.empty:
            logger.warning(f"No constituents found for sector '{sector}'")
            return []

        # 去重（多个板块可能有交集）
        all_stocks = all_stocks.drop_duplicates(subset=["code"])
        codes = all_stocks["code"].tolist()

        # 获取实时行情（含市值、PB、PE）
        quotes = self.provider.get_realtime_quotes(codes)
        if quotes.empty:
            logger.warning(f"No realtime quotes for sector '{sector}'")
            return []

        # 统一 code 列类型为 str（避免 object vs int64 merge 报错）
        quotes["code"] = quotes["code"].astype(str).str.zfill(6)
        all_stocks["code"] = all_stocks["code"].astype(str).str.zfill(6)

        # merge 名称
        merged = quotes.merge(
            all_stocks[["code", "name"]], on="code", how="inner", suffixes=("_q", "")
        )
        # 优先用 quotes 的 name
        if "name_q" in merged.columns:
            merged["name"] = merged["name_q"].fillna(merged["name"])
            merged.drop(columns=["name_q"], inplace=True)

        # 按市值降序
        merged["market_cap"] = pd.to_numeric(merged.get("market_cap", pd.Series(dtype=float)), errors="coerce")
        merged = merged.sort_values("market_cap", ascending=False).head(top_n)

        results = []
        for _, row in merged.iterrows():
            results.append({
                "code": row["code"],
                "name": row.get("name", ""),
                "market_cap": row.get("market_cap", 0),
                "price": row.get("price", 0),
                "pb": row.get("pb", None),
                "pe_ttm": row.get("pe_ttm", None),
            })

        return results

    # ---------- 2b. 相关性计算 ----------

    def compute_correlation(
        self, stock_pb_monthly: pd.DataFrame, factor: dict
    ) -> dict:
        """计算个股PB与单个因子的相关性（两种方法）

        Args:
            stock_pb_monthly: DataFrame [date, pb]（月频）
            factor: cycle_analysis.json 中的单个 factor 对象

        Returns:
            {pearson, best_lag_months, data_points, alignment, peak_match_rate, trough_match_rate, confidence}
        """
        result = {
            "pearson": None,
            "best_lag_months": 0,
            "data_points": 0,
            "alignment": "weak",
            "peak_match_rate": 0,
            "trough_match_rate": 0,
            "confidence": "low",
        }

        cycle_data = factor.get("cycle_data", [])
        if not cycle_data or stock_pb_monthly.empty:
            return result

        # === 方法A：插值 Pearson 相关 ===
        pearson_result = self._compute_pearson_correlation(stock_pb_monthly, cycle_data)
        result.update(pearson_result)

        # === 方法B：周期位置对比 ===
        position_result = self._compute_position_alignment(stock_pb_monthly, cycle_data)
        result.update(position_result)

        return result

    def _compute_pearson_correlation(
        self, stock_pb: pd.DataFrame, cycle_data: list
    ) -> dict:
        """方法A：将因子 peak/trough 插值为月度序列，与个股 PB 计算 Pearson 相关"""
        # 提取所有 peak/trough 点
        points = []
        for cycle in cycle_data:
            peak = cycle.get("peak", {})
            trough = cycle.get("trough", {})
            if peak.get("date") and peak.get("value") is not None:
                points.append((self._parse_quarter_date(peak["date"]), float(peak["value"])))
            if trough.get("date") and trough.get("value") is not None:
                points.append((self._parse_quarter_date(trough["date"]), float(trough["value"])))

        if len(points) < 2:
            return {"pearson": None, "best_lag_months": 0, "data_points": 0, "confidence": "low"}

        # 因子原始点太少 → 插值后 Pearson 无统计意义
        if len(points) < 4:
            return {"pearson": None, "best_lag_months": 0, "data_points": 0, "confidence": "low"}

        # 排序并创建时间序列
        points.sort(key=lambda x: x[0])
        factor_dates = [p[0] for p in points]
        factor_values = [p[1] for p in points]

        # 创建月度插值序列
        date_range = pd.date_range(
            start=min(factor_dates), end=max(factor_dates), freq="ME"
        )
        factor_series = pd.DataFrame({"date": date_range})
        # 线性插值
        factor_ts = pd.DataFrame({"date": factor_dates, "factor_value": factor_values})
        factor_ts["date"] = factor_ts["date"] + pd.offsets.MonthEnd(0)
        factor_ts = factor_ts.drop_duplicates(subset=["date"])

        merged = factor_series.merge(factor_ts, on="date", how="left")
        merged["factor_value"] = merged["factor_value"].interpolate(method="linear")
        merged = merged.dropna()

        # 与个股 PB 对齐
        stock_pb = stock_pb.copy()
        stock_pb["date"] = pd.to_datetime(stock_pb["date"]) + pd.offsets.MonthEnd(0)
        stock_pb = stock_pb.drop_duplicates(subset=["date"])

        aligned = merged.merge(stock_pb[["date", "pb"]], on="date", how="inner")
        aligned = aligned.dropna(subset=["factor_value", "pb"])

        data_points = len(aligned)
        if data_points < 6:
            return {"pearson": None, "best_lag_months": 0, "data_points": data_points}

        # 计算 0~3 月滞后的 Pearson 相关
        best_r = 0
        best_lag = 0
        for lag in range(4):
            if lag > 0:
                lagged_pb = aligned["pb"].shift(-lag)
            else:
                lagged_pb = aligned["pb"]

            valid = pd.DataFrame({
                "factor": aligned["factor_value"],
                "pb": lagged_pb
            }).dropna()

            if len(valid) < 6:
                continue

            if not np.isfinite(valid["factor"]).all() or not np.isfinite(valid["pb"]).all():
                continue

            r, _ = stats.pearsonr(valid["factor"], valid["pb"])
            if abs(r) > abs(best_r):
                best_r = r
                best_lag = lag

        # 需要因子原始点 >= 6 且对齐数据点 >= 24 才算 high confidence
        confidence = "high" if data_points >= 24 and len(points) >= 6 else "low"

        return {
            "pearson": round(best_r, 3),
            "best_lag_months": best_lag,
            "data_points": data_points,
            "confidence": confidence,
        }

    def _compute_position_alignment(
        self, stock_pb: pd.DataFrame, cycle_data: list
    ) -> dict:
        """方法B：在因子 peak/trough 时期（±2月），检查个股 PB 是高还是低"""
        stock_pb = stock_pb.copy()
        stock_pb["date"] = pd.to_datetime(stock_pb["date"])

        peak_matches = 0
        peak_total = 0
        trough_matches = 0
        trough_total = 0

        for cycle in cycle_data:
            peak = cycle.get("peak", {})
            trough = cycle.get("trough", {})

            if peak.get("date"):
                peak_date = self._parse_quarter_date(peak["date"])
                window_start = peak_date - timedelta(days=60)
                window_end = peak_date + timedelta(days=60)
                window_pb = stock_pb[
                    (stock_pb["date"] >= window_start) & (stock_pb["date"] <= window_end)
                ]["pb"].dropna()

                if len(window_pb) > 0:
                    peak_total += 1
                    # 检查窗口内 PB 是否处于整体历史的上半段
                    pb_median = stock_pb["pb"].dropna().median()
                    if window_pb.mean() > pb_median:
                        peak_matches += 1

            if trough.get("date"):
                trough_date = self._parse_quarter_date(trough["date"])
                window_start = trough_date - timedelta(days=60)
                window_end = trough_date + timedelta(days=60)
                window_pb = stock_pb[
                    (stock_pb["date"] >= window_start) & (stock_pb["date"] <= window_end)
                ]["pb"].dropna()

                if len(window_pb) > 0:
                    trough_total += 1
                    pb_median = stock_pb["pb"].dropna().median()
                    if window_pb.mean() < pb_median:
                        trough_matches += 1

        peak_rate = peak_matches / peak_total if peak_total > 0 else 0
        trough_rate = trough_matches / trough_total if trough_total > 0 else 0
        avg_rate = (peak_rate + trough_rate) / 2 if (peak_total + trough_total) > 0 else 0

        if avg_rate > 0.6:
            alignment = "positive"
        elif avg_rate < 0.3:
            alignment = "negative"
        else:
            alignment = "weak"

        return {
            "alignment": alignment,
            "peak_match_rate": round(peak_rate, 2),
            "trough_match_rate": round(trough_rate, 2),
        }

    # ---------- 2c. 估值位置计算 ----------

    def compute_valuation_position(
        self, pb_history: pd.DataFrame, cycle_data: list
    ) -> dict:
        """计算个股当前 PB 在周期中的位置

        Returns:
            {current_pb, cycle_peak_pb, cycle_trough_pb, historical_percentile,
             upside_to_peak, confidence}
        """
        result = {
            "current_pb": None,
            "cycle_peak_pb": None,
            "cycle_trough_pb": None,
            "historical_percentile": None,
            "upside_to_peak": None,
            "confidence": "low",
        }

        if pb_history.empty or "pb" not in pb_history.columns:
            return result

        pb_values = pb_history["pb"].dropna()
        if len(pb_values) == 0:
            return result

        current_pb = float(pb_values.iloc[-1])
        result["current_pb"] = round(current_pb, 2)

        # PB 为负 → 标注异常，不参与排名评分
        if current_pb <= 0:
            result["confidence"] = "pb_anomaly"
            return result

        # 从 cycle_data 提取 peak/trough 区间
        peak_pbs = []
        trough_pbs = []
        pb_history_copy = pb_history.copy()
        pb_history_copy["date"] = pd.to_datetime(pb_history_copy["date"])

        for factor_cycles in (cycle_data if isinstance(cycle_data, list) else [cycle_data]):
            cycles = factor_cycles if isinstance(factor_cycles, list) else factor_cycles.get("cycle_data", [])
            for cycle in cycles:
                peak = cycle.get("peak", {})
                trough = cycle.get("trough", {})

                if peak.get("date"):
                    peak_date = self._parse_quarter_date(peak["date"])
                    window = pb_history_copy[
                        (pb_history_copy["date"] >= peak_date - timedelta(days=90)) &
                        (pb_history_copy["date"] <= peak_date + timedelta(days=90))
                    ]["pb"].dropna()
                    if len(window) > 0:
                        peak_pbs.append(float(window.max()))

                if trough.get("date"):
                    trough_date = self._parse_quarter_date(trough["date"])
                    window = pb_history_copy[
                        (pb_history_copy["date"] >= trough_date - timedelta(days=90)) &
                        (pb_history_copy["date"] <= trough_date + timedelta(days=90))
                    ]["pb"].dropna()
                    if len(window) > 0:
                        trough_pbs.append(float(window.min()))

        if peak_pbs and trough_pbs:
            cycle_peak_pb = np.mean(peak_pbs)
            cycle_trough_pb = np.mean(trough_pbs)
            result["cycle_peak_pb"] = round(cycle_peak_pb, 2)
            result["cycle_trough_pb"] = round(cycle_trough_pb, 2)

            pb_range = cycle_peak_pb - cycle_trough_pb
            if pb_range > 0 and current_pb > 0:
                percentile = (current_pb - cycle_trough_pb) / pb_range
                percentile = max(0, min(1, percentile))
                result["historical_percentile"] = round(percentile, 3)
                result["upside_to_peak"] = round((cycle_peak_pb / current_pb - 1) * 100, 1)
                result["confidence"] = "high" if len(peak_pbs) >= 2 else "medium"
            else:
                result["confidence"] = "low"
        else:
            # 没有周期数据时用整体历史分位
            percentile = float((pb_values < current_pb).sum() / len(pb_values))
            result["historical_percentile"] = round(percentile, 3)
            hist_max = float(pb_values.max())
            hist_min = float(pb_values.min())
            result["cycle_peak_pb"] = round(hist_max, 2)
            result["cycle_trough_pb"] = round(hist_min, 2)
            if current_pb > 0:
                result["upside_to_peak"] = round((hist_max / current_pb - 1) * 100, 1)
            result["confidence"] = "low"

        return result

    # ---------- 2d. 排名算法 ----------

    def rank_stocks(
        self, stocks_analysis: list[dict], cycle_position: str, weights: dict | None = None
    ) -> list[dict]:
        """四维评分排名（行业自适应权重）

        Args:
            stocks_analysis: 每只股票的分析结果列表
            cycle_position: 行业周期位置（来自 cycle_analysis.json overall.cycle_position）
            weights: 可选自定义权重 {"upside", "alignment", "valuation", "momentum"}

        Returns:
            排序后的 stocks_analysis（加入 scores 和 total_score 字段）
        """
        w = weights or _select_weights(cycle_position)

        for stock in stocks_analysis:
            valuation = stock.get("valuation", {})
            confidence = valuation.get("confidence", "low")

            # PB 异常不参与排名
            if confidence == "pb_anomaly":
                stock["scores"] = {"upside": 0, "alignment": 0, "valuation": 0, "momentum": 0}
                stock["total_score"] = 0
                stock["pb_anomaly"] = True
                continue

            stock["pb_anomaly"] = False

            # 1. 上行空间 — sigmoid 归一化，区分度更好
            upside = valuation.get("upside_to_peak")
            if upside is not None and upside > 0:
                # sigmoid: 100% → ~50, 200% → ~67, 300% → ~75, 500% → ~83
                upside_score = 100 * (1 - 1 / (1 + upside / 100))
            else:
                upside_score = 0

            # 2. 周期吻合度 — 考虑相关方向 + 置信度过滤
            correlations = stock.get("correlation", {})
            if correlations:
                alignment_scores = []
                for factor_name, corr in correlations.items():
                    pearson = corr.get("pearson")
                    factor_confidence = corr.get("confidence", "low")
                    if pearson is not None and factor_confidence != "low":
                        base = abs(pearson) * 100
                        align = corr.get("alignment", "weak")
                        if align == "positive":
                            base = min(100, base * 1.2)  # 方向一致 +20%
                        elif align == "negative":
                            base = base * 0.5  # 方向相反 -50%
                        alignment_scores.append(base)
                # 所有因子都是 low confidence → 默认 30（保守）
                alignment_score = np.mean(alignment_scores) if alignment_scores else 30
            else:
                alignment_score = 30

            # 3. 估值位置 — 越低越好
            percentile = valuation.get("historical_percentile")
            if percentile is not None:
                valuation_score = 100 - percentile * 100
            else:
                valuation_score = 50

            # 4. 动量惩罚
            momentum_score = self._compute_momentum_score(
                stock.get("recent_6m_change", 0), cycle_position
            )

            scores = {
                "upside": round(upside_score, 1),
                "alignment": round(alignment_score, 1),
                "valuation": round(valuation_score, 1),
                "momentum": round(momentum_score, 1),
            }
            total = (
                upside_score * w["upside"]
                + alignment_score * w["alignment"]
                + valuation_score * w["valuation"]
                + momentum_score * w["momentum"]
            )
            stock["scores"] = scores
            stock["total_score"] = round(total, 1)

        # 排序：PB 异常排最后，其他按总分降序
        stocks_analysis.sort(
            key=lambda s: (not s.get("pb_anomaly", False), s.get("total_score", 0)),
            reverse=True,
        )

        # 编排名（PB异常不编排名）
        rank = 1
        for stock in stocks_analysis:
            if stock.get("pb_anomaly"):
                stock["rank"] = "-"
            else:
                stock["rank"] = rank
                rank += 1

        return stocks_analysis

    def _compute_momentum_score(self, change_6m: float, cycle_position: str) -> float:
        """动量惩罚评分

        底部周期（磨底/筑底/底部等） → 不惩罚（满分）
        反转初期/复苏 → 温和惩罚
        其他 → 正常惩罚
        """
        no_penalty_positions = ["磨底", "下行尾段", "筑底", "底部", "观察期"]
        if any(pos in cycle_position for pos in no_penalty_positions):
            return 100

        mild_penalty_positions = ["反转初期", "复苏"]
        change_pct = change_6m * 100 if abs(change_6m) < 5 else change_6m

        if any(pos in cycle_position for pos in mild_penalty_positions):
            if change_pct > 60:
                return 50
            elif change_pct > 40:
                return 70
            else:
                return 100

        # 其他情况（上行/过热等）：正常惩罚
        if change_pct > 60:
            return 0
        elif change_pct > 40:
            return 30
        elif change_pct > 20:
            return 60
        else:
            return 100

    # ---------- 完整分析流程 ----------

    def analyze_sector(
        self, sector: str, top_n: int = 10, progress_callback=None
    ) -> list[dict]:
        """执行完整的板块个股分析

        Args:
            sector: 板块名
            top_n: 分析个股数
            progress_callback: 可选的进度回调 fn(phase, message)

        Returns:
            排名后的完整分析结果列表
        """
        cycle_data = load_cycle_analysis()
        sector_cycle = cycle_data.get(sector, {})
        if not sector_cycle:
            raise ValueError(f"板块 '{sector}' 尚未完成周期分析")

        cycle_position = sector_cycle.get("overall", {}).get("cycle_position", "未知")
        factors = sector_cycle.get("factors", [])

        def _progress(phase, msg):
            if progress_callback:
                progress_callback(phase, msg)

        # 阶段1：获取成分股
        _progress(1, f"获取 {sector} 板块成分股...")
        stocks = self.get_sector_top_stocks(sector, top_n)
        if not stocks:
            return []
        _progress(1, f"按市值取前 {len(stocks)} 只")

        # 阶段2：获取历史数据
        _progress(2, "获取历史PB/PS数据...")
        stocks_analysis = []
        for i, stock in enumerate(stocks):
            _progress(2, f"[{i+1}/{len(stocks)}] {stock['name']}({stock['code']})")
            pb_data = self.provider.get_pb_history_long(stock["code"])
            stock["pb_data"] = pb_data
            stock["pb_months"] = len(pb_data)
            _progress(2, f"  → {len(pb_data)} 个月数据")

            # 获取近 6 个月涨幅
            recent_change = self._get_recent_change(stock["code"], months=6)
            stock["recent_6m_change"] = recent_change

            stocks_analysis.append(stock)

        # 阶段3：计算相关性与排名
        _progress(3, "计算相关性...")
        for stock in stocks_analysis:
            pb_data = stock.get("pb_data", pd.DataFrame())

            # 每个因子算相关性
            correlation = {}
            for factor in factors:
                factor_name = factor.get("name", "unknown")
                if not pb_data.empty:
                    corr = self.compute_correlation(pb_data, factor)
                else:
                    corr = {"pearson": None, "confidence": "low", "alignment": "weak",
                            "peak_match_rate": 0, "trough_match_rate": 0}
                correlation[factor_name] = corr

            stock["correlation"] = correlation

            # 估值位置（用所有因子的 cycle_data 综合判断）
            all_factor_cycles = [f.get("cycle_data", []) for f in factors]
            valuation = self.compute_valuation_position(pb_data, all_factor_cycles)
            stock["valuation"] = valuation

        # 排名
        _progress(3, "生成排名...")
        self.rank_stocks(stocks_analysis, cycle_position)

        # 清理 pb_data（不序列化大 DataFrame）
        for stock in stocks_analysis:
            if "pb_data" in stock:
                del stock["pb_data"]

        return stocks_analysis

    def _get_recent_change(self, code: str, months: int = 6) -> float:
        """获取近 N 个月涨幅"""
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=months * 30)).strftime("%Y-%m-%d")
            daily = self.provider.get_daily_quotes(code, start, end)
            if daily.empty or len(daily) < 2:
                return 0
            first_close = float(daily.iloc[0]["close"])
            last_close = float(daily.iloc[-1]["close"])
            if first_close > 0:
                return round((last_close / first_close - 1), 4)
        except Exception:
            pass
        return 0

    # ---------- 持久化 ----------

    def save_analysis(self, sector: str, stocks_analysis: list[dict], cycle_position: str, top_n: int):
        """保存分析结果到 JSON"""
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = sector.replace("/", "_").replace("\\", "_")
        out_path = PROFILES_DIR / f"{safe_name}.json"

        data = {
            "sector": sector,
            "analyzed_at": datetime.now().isoformat(),
            "cycle_position": cycle_position,
            "top_n": top_n,
            "stocks": stocks_analysis,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Saved analysis to {out_path}")
        return out_path

    @staticmethod
    def load_analysis(sector: str) -> Optional[dict]:
        """加载已保存的分析结果"""
        safe_name = sector.replace("/", "_").replace("\\", "_")
        path = PROFILES_DIR / f"{safe_name}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    @staticmethod
    def load_all_analyses() -> list[dict]:
        """加载所有已分析板块的结果"""
        results = []
        if PROFILES_DIR.exists():
            for f in PROFILES_DIR.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        results.append(json.load(fp))
                except Exception:
                    continue
        return results

    # ---------- 自选列表 ----------

    @staticmethod
    def load_watchlist() -> list[dict]:
        if WATCHLIST_PATH.exists():
            try:
                with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    @staticmethod
    def save_watchlist(watchlist: list[dict]):
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)

    @staticmethod
    def add_to_watchlist(code: str, name: str, sector: str, note: str = ""):
        watchlist = StockCycleAnalyzer.load_watchlist()
        # 检查是否已存在
        if any(w["code"] == code for w in watchlist):
            return False
        watchlist.append({
            "code": code,
            "name": name,
            "sector": sector,
            "added_at": datetime.now().isoformat(),
            "note": note,
        })
        StockCycleAnalyzer.save_watchlist(watchlist)
        return True

    @staticmethod
    def remove_from_watchlist(code: str):
        watchlist = StockCycleAnalyzer.load_watchlist()
        watchlist = [w for w in watchlist if w["code"] != code]
        StockCycleAnalyzer.save_watchlist(watchlist)

    # ---------- 工具方法 ----------

    @staticmethod
    def _parse_quarter_date(quarter_str: str) -> datetime:
        """解析 '2022-Q1' 格式为 datetime（取季度中间月）"""
        try:
            parts = quarter_str.split("-")
            year = int(parts[0])
            if len(parts) > 1 and parts[1].startswith("Q"):
                q = int(parts[1][1])
                month = q * 3 - 1  # Q1→2, Q2→5, Q3→8, Q4→11
                return datetime(year, month, 15)
            elif len(parts) > 1:
                month = int(parts[1])
                return datetime(year, month, 15)
            return datetime(year, 6, 15)
        except Exception:
            return datetime(2020, 1, 1)
