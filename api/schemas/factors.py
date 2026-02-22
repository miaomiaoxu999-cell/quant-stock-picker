"""Pydantic v2 schemas for factors endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Factor(BaseModel):
    name: str
    weight: int
    description: str
    data_source: str


class SectorFactorsResponse(BaseModel):
    sector: str
    factors: list[Factor] = []
    conversation: list[dict[str, str]] = []
    updated_at: str | None = None


class AllSectorsResponse(BaseModel):
    sectors: dict[str, Any]


class PresetSectorsResponse(BaseModel):
    sectors: list[str]


class FactorWeightUpdate(BaseModel):
    weights: list[int]


class FactorChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []
