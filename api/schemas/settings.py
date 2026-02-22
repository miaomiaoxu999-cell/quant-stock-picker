"""Pydantic v2 schemas for settings endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LLMSettingsResponse(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    tavily_api_key: str = ""
    jina_api_key: str = ""
    apify_api_key: str = ""
    brave_api_key: str = ""
    audit_api_key: str = ""
    audit_base_url: str = ""
    audit_model: str = ""


class LLMSettingsUpdate(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    tavily_api_key: str | None = None
    jina_api_key: str | None = None
    apify_api_key: str | None = None
    brave_api_key: str | None = None
    audit_api_key: str | None = None
    audit_base_url: str | None = None
    audit_model: str | None = None


class StrategyConfigResponse(BaseModel):
    model_config = {"extra": "allow"}

    industries: dict[str, Any] = {}
    cash_reserve: float = 0.10
    buy_strategy: dict[str, Any] = {}
    sell_strategy: dict[str, Any] = {}
    stop_loss: dict[str, Any] = {}
    valuation: dict[str, Any] = {}


class StrategyConfigUpdate(BaseModel):
    model_config = {"extra": "allow"}

    cash_reserve: float | None = None
    buy_strategy: dict[str, Any] | None = None
    sell_strategy: dict[str, Any] | None = None
    stop_loss: dict[str, Any] | None = None
    valuation: dict[str, Any] | None = None
