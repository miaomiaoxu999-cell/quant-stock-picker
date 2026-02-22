"""Portfolio router - holdings CRUD, risk checks, targets, realtime prices."""

from __future__ import annotations

import csv
import io
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File

from api.deps import (
    load_portfolio_state,
    save_portfolio_state,
    load_config_yaml,
)
from api.schemas.portfolio import (
    PortfolioStateResponse,
    PortfolioUpdateRequest,
    PortfolioImportRequest,
    RiskCheckItem,
    RiskCheckResponse,
    TargetItem,
    TargetsResponse,
    RealtimePriceItem,
    RealtimePricesResponse,
)

router = APIRouter(tags=["portfolio"])


@router.get("/portfolio/state", response_model=PortfolioStateResponse)
def get_portfolio_state():
    """Load portfolio_state.json."""
    state = load_portfolio_state()
    holdings = state.get("holdings", state)
    meta = state.get("meta", {})
    # Legacy format: top-level keys are stock codes (no "holdings" wrapper)
    if "holdings" not in state and any(
        isinstance(v, dict) and "avg_cost" in v for v in state.values()
    ):
        holdings = state
        meta = {}
    return {"holdings": holdings, "meta": meta}


@router.put("/portfolio/state", response_model=PortfolioStateResponse)
def update_portfolio_state(body: PortfolioUpdateRequest):
    """Update holdings in portfolio_state.json."""
    state = load_portfolio_state()
    state["holdings"] = body.holdings
    state["meta"] = state.get("meta", {})
    state["meta"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_portfolio_state(state)
    return {"holdings": state["holdings"], "meta": state["meta"]}


@router.post("/portfolio/import", response_model=PortfolioStateResponse)
def import_portfolio(body: PortfolioImportRequest):
    """Import holdings from parsed CSV items (code, name, avg_cost, shares)."""
    state = load_portfolio_state()
    holdings = state.get("holdings", {})

    for item in body.items:
        code = item.code.zfill(6)
        holdings[code] = {
            "name": item.name,
            "avg_cost": item.avg_cost,
            "shares": item.shares,
        }

    state["holdings"] = holdings
    state["meta"] = state.get("meta", {})
    state["meta"]["imported_at"] = datetime.now().isoformat(timespec="seconds")
    state["meta"]["import_count"] = len(body.items)
    save_portfolio_state(state)
    return {"holdings": state["holdings"], "meta": state["meta"]}


@router.get("/portfolio/risk", response_model=RiskCheckResponse)
def check_portfolio_risk():
    """Stop-loss + sell signal check using PortfolioConstructor + realtime quotes."""
    from quant.portfolio.constructor import PortfolioConstructor
    from quant.data.akshare_provider import AKShareProvider
    from quant.data.cache import DataCache

    config = load_config_yaml()
    if not config:
        raise HTTPException(status_code=400, detail="No config.yaml found")

    state = load_portfolio_state()
    holdings = state.get("holdings", state)
    if not holdings:
        return {"items": []}

    codes = list(holdings.keys())

    provider = AKShareProvider(cache=DataCache())
    quotes = provider.get_realtime_quotes(codes)

    constructor = PortfolioConstructor(config)
    items: list[RiskCheckItem] = []

    for code, holding in holdings.items():
        avg_cost = holding.get("avg_cost", 0)
        name = holding.get("name", "")

        quote_row = quotes[quotes["code"] == code] if not quotes.empty else None
        current_price = 0.0
        current_pb = 0.0
        if quote_row is not None and not quote_row.empty:
            row = quote_row.iloc[0]
            current_price = float(row.get("price", 0) or 0)
            current_pb = float(row.get("pb", 0) or 0)

        if avg_cost <= 0 or current_price <= 0:
            items.append(RiskCheckItem(code=code, name=name))
            continue

        # Stop-loss check
        stop = constructor.check_stop_loss(code, name, current_price, avg_cost)

        # Sell signal check - need peak PB from config
        peak_pb = 0.0
        for ind_config in config.get("industries", {}).values():
            for stock in ind_config.get("stocks", []):
                if str(stock.get("code", "")).zfill(6) == code:
                    peak_pb = stock.get("pb_peak_2022") or 0
                    break

        sell = constructor.check_sell_signals(
            code, name, current_price, avg_cost, current_pb, peak_pb,
        )

        items.append(RiskCheckItem(
            code=code,
            name=name,
            stop_loss_triggered=stop.get("stop_loss_triggered", False),
            level=stop.get("level", 0),
            action=stop.get("action", ""),
            drawdown=stop.get("drawdown", 0.0),
            should_sell=sell.get("should_sell", False),
            sell_ratio=sell.get("sell_ratio", 0.0),
            sell_reason=sell.get("reason", ""),
        ))

    return {"items": items}


@router.get("/portfolio/targets", response_model=TargetsResponse)
def get_portfolio_targets():
    """Target vs actual weights from config.yaml vs portfolio_state."""
    config = load_config_yaml()
    if not config:
        raise HTTPException(status_code=400, detail="No config.yaml found")

    state = load_portfolio_state()
    holdings = state.get("holdings", state)

    targets: list[TargetItem] = []
    for ind_key, ind_config in config.get("industries", {}).items():
        for stock in ind_config.get("stocks", []):
            code = str(stock.get("code", "")).zfill(6)
            target_w = stock.get("weight", 0)
            current_w = holdings.get(code, {}).get("weight", 0)
            targets.append(TargetItem(
                code=code,
                name=stock.get("name", ""),
                industry=ind_key,
                target_weight=target_w,
                current_weight=current_w,
                diff=round(target_w - current_w, 4),
            ))

    return {
        "targets": targets,
        "cash_reserve": config.get("cash_reserve", 0.10),
    }


@router.get("/portfolio/realtime-prices", response_model=RealtimePricesResponse)
def get_realtime_prices():
    """Get latest prices via AKShareProvider.get_realtime_quotes() for holdings."""
    from quant.data.akshare_provider import AKShareProvider
    from quant.data.cache import DataCache

    state = load_portfolio_state()
    holdings = state.get("holdings", state)
    if not holdings:
        return {"prices": []}

    codes = list(holdings.keys())
    provider = AKShareProvider(cache=DataCache())
    quotes = provider.get_realtime_quotes(codes)

    prices: list[RealtimePriceItem] = []
    for code in codes:
        name = holdings.get(code, {}).get("name", "")
        if quotes.empty:
            prices.append(RealtimePriceItem(code=code, name=name))
            continue

        row = quotes[quotes["code"] == code]
        if row.empty:
            prices.append(RealtimePriceItem(code=code, name=name))
            continue

        r = row.iloc[0]
        prices.append(RealtimePriceItem(
            code=code,
            name=r.get("name", name) or name,
            price=_safe_float(r.get("price")),
            pb=_safe_float(r.get("pb")),
            pe_ttm=_safe_float(r.get("pe_ttm")),
            market_cap=_safe_float(r.get("market_cap")),
            change_60d=_safe_float(r.get("change_60d")),
        ))

    return {"prices": prices}


def _safe_float(val) -> float | None:
    """Convert to float, return None if NaN or invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None
