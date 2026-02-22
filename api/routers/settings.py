"""Settings router - LLM settings & strategy config endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.deps import load_llm_settings, save_llm_settings, load_config_yaml, save_config_yaml
from api.schemas.settings import (
    LLMSettingsResponse,
    LLMSettingsUpdate,
    StrategyConfigResponse,
    StrategyConfigUpdate,
)

router = APIRouter(tags=["settings"])


@router.get("/settings/llm", response_model=LLMSettingsResponse)
def get_llm_settings():
    """Get current LLM settings."""
    return load_llm_settings()


@router.put("/settings/llm", response_model=LLMSettingsResponse)
def update_llm_settings(body: LLMSettingsUpdate):
    """Update LLM settings."""
    current = load_llm_settings()
    updates = body.model_dump(exclude_unset=True)
    # Strip whitespace from string values
    for k, v in updates.items():
        if isinstance(v, str):
            updates[k] = v.strip()
    current.update(updates)
    save_llm_settings(current)
    return current


@router.get("/settings/strategy", response_model=StrategyConfigResponse)
def get_strategy_config():
    """Get current strategy config (config.yaml)."""
    config = load_config_yaml()
    if not config:
        raise HTTPException(status_code=404, detail="config.yaml not found")
    return config


@router.put("/settings/strategy", response_model=StrategyConfigResponse)
def update_strategy_config(body: StrategyConfigUpdate):
    """Update strategy config (config.yaml)."""
    config = load_config_yaml()
    if not config:
        raise HTTPException(status_code=404, detail="config.yaml not found")
    updates = body.model_dump(exclude_unset=True)
    # Deep merge top-level keys
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    save_config_yaml(config)
    return config
