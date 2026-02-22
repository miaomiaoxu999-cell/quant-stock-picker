"""Pydantic v2 schemas for portfolio endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PortfolioStateResponse(BaseModel):
    holdings: dict[str, Any] = {}
    meta: dict[str, Any] = {}


class PortfolioUpdateRequest(BaseModel):
    holdings: dict[str, Any]


class PortfolioImportItem(BaseModel):
    code: str
    name: str
    avg_cost: float
    shares: int


class PortfolioImportRequest(BaseModel):
    items: list[PortfolioImportItem]


class RiskCheckItem(BaseModel):
    code: str
    name: str
    stop_loss_triggered: bool = False
    level: int = 0
    action: str = ""
    drawdown: float = 0.0
    should_sell: bool = False
    sell_ratio: float = 0.0
    sell_reason: str = ""


class RiskCheckResponse(BaseModel):
    items: list[RiskCheckItem]


class TargetItem(BaseModel):
    code: str
    name: str
    industry: str
    target_weight: float
    current_weight: float = 0.0
    diff: float = 0.0


class TargetsResponse(BaseModel):
    targets: list[TargetItem]
    cash_reserve: float = 0.0


class RealtimePriceItem(BaseModel):
    code: str
    name: str = ""
    price: float | None = None
    pb: float | None = None
    pe_ttm: float | None = None
    market_cap: float | None = None
    change_60d: float | None = None


class RealtimePricesResponse(BaseModel):
    prices: list[RealtimePriceItem]
