"""Stocks router - stock profile CRUD, analysis, watchlist, chart data."""

from __future__ import annotations

import json
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import (
    load_all_stock_profiles,
    load_stock_profile,
    load_watchlist,
    save_watchlist,
    get_llm_config,
    STOCK_PROFILES_DIR,
)
from api.schemas.stocks import (
    StocksSummaryResponse,
    StockSummaryItem,
    SectorsListResponse,
    AnalyzeRequest,
    RedoRequest,
    SectorResultsResponse,
    ChartDataResponse,
    ExtremesResponse,
    WatchlistItem,
    WatchlistAddRequest,
    LegacyProfilesResponse,
)
from api.services.task_manager import task_manager, TaskStatus
from quant.utils.constants import STOCK_PROFILES

router = APIRouter(tags=["stocks"])

# Shared progress dicts keyed by sector
_stocks_progress: dict[str, dict] = {}


@router.get("/stocks/summary", response_model=StocksSummaryResponse)
def get_cross_sector_summary():
    """Cross-sector top 3 from all stock_profiles/*.json."""
    all_analyses = load_all_stock_profiles()
    rows = []
    for analysis in all_analyses:
        sector = analysis.get("sector", "")
        cycle_pos = analysis.get("cycle_position", "-")
        stocks = analysis.get("stocks", [])
        valid = [s for s in stocks if not s.get("pb_anomaly", False)]
        valid.sort(key=lambda s: s.get("total_score", 0), reverse=True)
        for stock in valid[:3]:
            val = stock.get("valuation", {})
            rows.append(StockSummaryItem(
                sector=sector,
                rank=stock.get("rank", "-"),
                name=stock.get("name", ""),
                code=stock.get("code", ""),
                price=stock.get("price"),
                pb=val.get("current_pb") or stock.get("pb"),
                cycle_position=cycle_pos,
                upside_pct=val.get("upside_to_peak"),
                total_score=round(stock.get("total_score", 0), 1),
            ))
    rows.sort(key=lambda r: r.total_score, reverse=True)
    return {"stocks": rows}


@router.get("/stocks/sectors", response_model=SectorsListResponse)
def get_analyzed_sectors():
    """List sectors that have stock analysis results."""
    all_analyses = load_all_stock_profiles()
    sectors = [a.get("sector", "") for a in all_analyses if a.get("sector")]
    return {"sectors": sectors}


@router.get("/stocks/watchlist", response_model=list[WatchlistItem])
def get_watchlist():
    """Load watchlist."""
    return load_watchlist()


@router.post("/stocks/watchlist", response_model=WatchlistItem)
def add_to_watchlist(body: WatchlistAddRequest):
    """Add stock to watchlist."""
    watchlist = load_watchlist()
    if any(w["code"] == body.code for w in watchlist):
        raise HTTPException(status_code=409, detail=f"Stock '{body.code}' already in watchlist")
    item = {
        "code": body.code,
        "name": body.name,
        "sector": body.sector,
        "note": body.note,
        "added_at": datetime.now().isoformat(),
    }
    watchlist.append(item)
    save_watchlist(watchlist)
    return item


@router.delete("/stocks/watchlist/{code}")
def remove_from_watchlist(code: str):
    """Remove stock from watchlist."""
    watchlist = load_watchlist()
    new_list = [w for w in watchlist if w["code"] != code]
    if len(new_list) == len(watchlist):
        raise HTTPException(status_code=404, detail=f"Stock '{code}' not in watchlist")
    save_watchlist(new_list)
    return {"status": "ok", "removed": code}


@router.get("/stocks/legacy", response_model=LegacyProfilesResponse)
def get_legacy_profiles():
    """Return STOCK_PROFILES constant."""
    return {"profiles": STOCK_PROFILES}


@router.post("/stocks/{sector}/analyze")
def analyze_sector_stocks(sector: str, body: AnalyzeRequest):
    """SSE endpoint: run background stock ranking analysis."""
    task_id = f"stocks_analysis_{sector}"

    progress = {"status": "running", "sector": sector, "phase": "init", "message": ""}
    _stocks_progress[sector] = progress

    def worker(_cancel_event=None):
        try:
            from quant.analysis.stock_cycle_analyzer import StockCycleAnalyzer

            def progress_cb(phase, msg):
                progress["phase"] = str(phase)
                progress["message"] = msg

            analyzer = StockCycleAnalyzer()
            stocks = analyzer.analyze_sector(sector, top_n=body.top_n, progress_callback=progress_cb)

            if _cancel_event and _cancel_event.is_set():
                progress["status"] = "cancelled"
                return

            from quant.analysis.stock_cycle_analyzer import load_cycle_analysis
            cycle_data = load_cycle_analysis()
            cycle_position = cycle_data.get(sector, {}).get("overall", {}).get("cycle_position", "")
            analyzer.save_analysis(sector, stocks, cycle_position, body.top_n)

            progress["status"] = "completed"
            progress["result"] = {"sector": sector, "stock_count": len(stocks)}
        except Exception as e:
            progress["status"] = "failed"
            progress["error"] = str(e)

    task_manager.submit(worker, task_id=task_id)

    def event_stream():
        while True:
            status = progress.get("status", "running")
            if status == "running":
                yield f"data: {json.dumps({'type': 'progress', 'phase': progress.get('phase', ''), 'message': progress.get('message', '')}, ensure_ascii=False)}\n\n"
                time.sleep(2)
            elif status == "completed":
                yield f"data: {json.dumps({'type': 'done', 'content': 'Analysis completed'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'analysis_saved', 'result': progress.get('result', {})}, ensure_ascii=False)}\n\n"
                _stocks_progress.pop(sector, None)
                break
            elif status == "cancelled":
                yield f"data: {json.dumps({'type': 'error', 'message': 'Cancelled'}, ensure_ascii=False)}\n\n"
                _stocks_progress.pop(sector, None)
                break
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': progress.get('error', 'Unknown error')}, ensure_ascii=False)}\n\n"
                _stocks_progress.pop(sector, None)
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/stocks/{sector}/redo")
def redo_sector_stocks(sector: str, body: RedoRequest):
    """SSE endpoint: redo analysis with weight/exclusion overrides."""
    task_id = f"stocks_redo_{sector}"

    progress = {"status": "running", "sector": sector, "phase": "init", "message": ""}
    _stocks_progress[sector] = progress

    audit_params = None
    if body.weights or body.excluded_codes:
        audit_params = {
            "weights": body.weights,
            "excluded_codes": body.excluded_codes,
        }

    def worker(_cancel_event=None):
        try:
            from quant.analysis.stock_cycle_analyzer import (
                StockCycleAnalyzer, load_cycle_analysis,
            )
            from quant.data.akshare_provider import AKShareProvider
            from quant.data.cache import DataCache

            cache = DataCache()

            # Clear old PB caches
            old = StockCycleAnalyzer.load_analysis(sector)
            if old and old.get("stocks"):
                for s in old["stocks"]:
                    code = s.get("code", "")
                    if code:
                        cache.clear_pattern(f"pb_%{code}%")

            # Delete old JSON
            safe_name = sector.replace("/", "_").replace("\\", "_")
            old_path = STOCK_PROFILES_DIR / f"{safe_name}.json"
            if old_path.exists():
                old_path.unlink()

            # Re-analyze
            cycle_data = load_cycle_analysis()
            sector_info = cycle_data.get(sector, {})
            cycle_position = sector_info.get("overall", {}).get("cycle_position", "")

            analyzer = StockCycleAnalyzer()
            stocks = analyzer.get_sector_top_stocks(sector, body.top_n)
            if not stocks:
                raise ValueError("No sector constituent stocks found")

            # Apply exclusions
            if audit_params and audit_params.get("excluded_codes"):
                excluded = set(audit_params["excluded_codes"])
                stocks = [s for s in stocks if s["code"] not in excluded]
                if not stocks:
                    stocks = analyzer.get_sector_top_stocks(sector, body.top_n)

            if _cancel_event and _cancel_event.is_set():
                progress["status"] = "cancelled"
                return

            factors = sector_info.get("factors", [])
            for stock in stocks:
                if _cancel_event and _cancel_event.is_set():
                    progress["status"] = "cancelled"
                    return
                pb_data = analyzer.provider.get_pb_history_long(stock["code"])
                stock["pb_data"] = pb_data
                stock["pb_months"] = len(pb_data)
                stock["recent_6m_change"] = analyzer._get_recent_change(stock["code"], months=6)

            for stock in stocks:
                import pandas as pd
                pb_data = stock.get("pb_data", pd.DataFrame())
                correlation = {}
                for factor in factors:
                    factor_name = factor.get("name", "unknown")
                    if not pb_data.empty:
                        corr = analyzer.compute_correlation(pb_data, factor)
                    else:
                        corr = {"pearson": None, "confidence": "low", "alignment": "weak",
                                "peak_match_rate": 0, "trough_match_rate": 0}
                    correlation[factor_name] = corr
                stock["correlation"] = correlation

                all_factor_cycles = [f.get("cycle_data", []) for f in factors]
                valuation = analyzer.compute_valuation_position(pb_data, all_factor_cycles)
                stock["valuation"] = valuation

            custom_weights = audit_params.get("weights") if audit_params else None
            analyzer.rank_stocks(stocks, cycle_position, weights=custom_weights)

            for stock in stocks:
                stock.pop("pb_data", None)

            analyzer.save_analysis(sector, stocks, cycle_position, body.top_n)
            progress["status"] = "completed"
            progress["result"] = {"sector": sector, "stock_count": len(stocks)}

        except Exception as e:
            progress["status"] = "failed"
            progress["error"] = str(e)

    task_manager.submit(worker, task_id=task_id)

    def event_stream():
        while True:
            status = progress.get("status", "running")
            if status == "running":
                yield f"data: {json.dumps({'type': 'progress', 'phase': progress.get('phase', ''), 'message': progress.get('message', '')}, ensure_ascii=False)}\n\n"
                time.sleep(2)
            elif status == "completed":
                yield f"data: {json.dumps({'type': 'done', 'content': 'Redo completed'}, ensure_ascii=False)}\n\n"
                _stocks_progress.pop(sector, None)
                break
            elif status == "cancelled":
                yield f"data: {json.dumps({'type': 'error', 'message': 'Cancelled'}, ensure_ascii=False)}\n\n"
                _stocks_progress.pop(sector, None)
                break
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': progress.get('error', 'Unknown error')}, ensure_ascii=False)}\n\n"
                _stocks_progress.pop(sector, None)
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/stocks/{sector}/results", response_model=SectorResultsResponse)
def get_sector_results(sector: str):
    """Load analysis results from stock_profiles/{sector}.json."""
    data = load_stock_profile(sector)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No results for sector '{sector}'")
    return data


@router.get("/stocks/{code}/chart-data", response_model=ChartDataResponse)
def get_chart_data(code: str):
    """PB/price history via AKShareProvider."""
    from quant.data.akshare_provider import AKShareProvider
    from quant.data.cache import DataCache

    provider = AKShareProvider(cache=DataCache())
    pb_data = provider.get_pb_history_long(code, years=15)

    if pb_data.empty:
        return {"code": code, "dates": [], "pb": [], "close": []}

    dates = pb_data["date"].astype(str).tolist()
    pb_vals = [None if v != v else float(v) for v in pb_data["pb"].tolist()]
    close_vals = []
    if "close" in pb_data.columns:
        close_vals = [None if v != v else float(v) for v in pb_data["close"].tolist()]

    return {"code": code, "dates": dates, "pb": pb_vals, "close": close_vals}


@router.get("/stocks/{code}/extremes", response_model=ExtremesResponse)
def get_extremes(code: str):
    """Historical high/low from PB history."""
    import pandas as pd
    from quant.data.akshare_provider import AKShareProvider
    from quant.data.cache import DataCache

    provider = AKShareProvider(cache=DataCache())
    pb_data = provider.get_pb_history_long(code, years=15)

    result = {
        "code": code,
        "high_price": "-", "high_pb": "-", "high_date": "-",
        "low_price": "-", "low_pb": "-", "low_date": "-",
    }

    if pb_data.empty or "close" not in pb_data.columns:
        return result

    df = pb_data.dropna(subset=["close", "pb"]).copy()
    df = df[df["close"] > 0]
    if df.empty:
        return result

    df["date"] = pd.to_datetime(df["date"])

    idx_max = df["close"].idxmax()
    row_max = df.loc[idx_max]
    result["high_price"] = round(float(row_max["close"]), 2)
    result["high_pb"] = round(float(row_max["pb"]), 2)
    result["high_date"] = row_max["date"].strftime("%Y-%m")

    idx_min = df["close"].idxmin()
    row_min = df.loc[idx_min]
    result["low_price"] = round(float(row_min["close"]), 2)
    result["low_pb"] = round(float(row_min["pb"]), 2)
    result["low_date"] = row_min["date"].strftime("%Y-%m")

    return result
