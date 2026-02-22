"""Pydantic v2 schemas for audit endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AuditSectorResponse(BaseModel):
    sector: str
    results: dict[str, Any] = {}


class AuditRunRequest(BaseModel):
    pass  # audit_type comes from path


class AuditChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class AuditFeedbackRequest(BaseModel):
    feedback: str


class AuditSectorsResponse(BaseModel):
    sectors: list[str]
