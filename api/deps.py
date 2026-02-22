"""Shared dependencies - file paths, locks, config loaders."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import yaml

from quant.llm.client import LLMConfig

# ==================== Paths ====================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"

LLM_SETTINGS_PATH = DATA_DIR / "llm_settings.json"
SECTOR_FACTORS_PATH = DATA_DIR / "sector_factors.json"
CONFIG_YAML_PATH = CONFIG_DIR / "config.yaml"
CYCLE_ANALYSIS_PATH = DATA_DIR / "cycle_analysis.json"
AUDIT_RESULTS_PATH = DATA_DIR / "audit_results.json"
ADVISOR_SESSION_PATH = DATA_DIR / "advisor_session.json"
PORTFOLIO_STATE_PATH = DATA_DIR / "portfolio_state.json"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
STOCK_PROFILES_DIR = DATA_DIR / "stock_profiles"
RESEARCH_DIR = DATA_DIR / "research"

# ==================== Locks ====================

_llm_settings_lock = threading.Lock()
_sector_factors_lock = threading.Lock()
_config_yaml_lock = threading.Lock()
_cycle_analysis_lock = threading.Lock()
_audit_results_lock = threading.Lock()
_advisor_session_lock = threading.Lock()
_portfolio_state_lock = threading.Lock()
_watchlist_lock = threading.Lock()

# ==================== LLM Settings ====================

_LLM_DEFAULTS = {
    "api_key": "",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "Pro/zai-org/GLM-5",
    "tavily_api_key": "",
    "jina_api_key": "",
    "apify_api_key": "",
    "audit_api_key": "",
    "audit_base_url": "https://api.siliconflow.cn/v1",
    "audit_model": "deepseek-ai/DeepSeek-V3",
}


def load_llm_settings() -> dict:
    with _llm_settings_lock:
        if LLM_SETTINGS_PATH.exists():
            try:
                with open(LLM_SETTINGS_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for k, v in _LLM_DEFAULTS.items():
                    if k not in saved:
                        saved[k] = v
                return saved
            except Exception:
                pass
        return dict(_LLM_DEFAULTS)


def save_llm_settings(settings: dict) -> None:
    with _llm_settings_lock:
        LLM_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LLM_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)


def get_llm_config() -> LLMConfig | None:
    s = load_llm_settings()
    if not s.get("api_key"):
        return None
    return LLMConfig(
        api_key=s["api_key"],
        base_url=s.get("base_url", _LLM_DEFAULTS["base_url"]),
        model=s.get("model", _LLM_DEFAULTS["model"]),
    )


def get_audit_llm_config() -> LLMConfig | None:
    """Return LLMConfig for audit model. Falls back to main LLM if audit key is empty."""
    s = load_llm_settings()
    api_key = s.get("audit_api_key") or s.get("api_key")
    if not api_key:
        return None
    return LLMConfig(
        api_key=api_key,
        base_url=s.get("audit_base_url") or s.get("base_url", _LLM_DEFAULTS["audit_base_url"]),
        model=s.get("audit_model") or s.get("model", _LLM_DEFAULTS["audit_model"]),
    )


# ==================== Sector Factors ====================

def load_sector_factors() -> dict:
    with _sector_factors_lock:
        if SECTOR_FACTORS_PATH.exists():
            try:
                with open(SECTOR_FACTORS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


def save_sector_factors(data: dict) -> None:
    with _sector_factors_lock:
        SECTOR_FACTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SECTOR_FACTORS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ==================== Config YAML ====================

def load_config_yaml() -> dict:
    with _config_yaml_lock:
        if CONFIG_YAML_PATH.exists():
            try:
                with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}


def save_config_yaml(config: dict) -> None:
    with _config_yaml_lock:
        CONFIG_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ==================== Cycle Analysis ====================

def load_cycle_analysis() -> dict:
    with _cycle_analysis_lock:
        if CYCLE_ANALYSIS_PATH.exists():
            try:
                with open(CYCLE_ANALYSIS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


def save_cycle_analysis(data: dict) -> None:
    with _cycle_analysis_lock:
        CYCLE_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CYCLE_ANALYSIS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ==================== Audit Results ====================

def load_audit_results() -> dict:
    with _audit_results_lock:
        if AUDIT_RESULTS_PATH.exists():
            try:
                with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


def save_audit_result(sector: str, audit_type: str, data: dict) -> None:
    with _audit_results_lock:
        all_results: dict = {}
        if AUDIT_RESULTS_PATH.exists():
            try:
                with open(AUDIT_RESULTS_PATH, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
            except Exception:
                pass
        if sector not in all_results:
            all_results[sector] = {}
        all_results[sector][audit_type] = data
        AUDIT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)


# ==================== Advisor Session ====================

def load_advisor_session() -> dict | None:
    with _advisor_session_lock:
        if ADVISOR_SESSION_PATH.exists():
            try:
                with open(ADVISOR_SESSION_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None


def save_advisor_session(session: dict) -> None:
    with _advisor_session_lock:
        ADVISOR_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ADVISOR_SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)


# ==================== Portfolio State ====================

def load_portfolio_state() -> dict:
    with _portfolio_state_lock:
        if PORTFOLIO_STATE_PATH.exists():
            try:
                with open(PORTFOLIO_STATE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


def save_portfolio_state(state: dict) -> None:
    with _portfolio_state_lock:
        PORTFOLIO_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PORTFOLIO_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


# ==================== Watchlist ====================

def load_watchlist() -> list:
    with _watchlist_lock:
        if WATCHLIST_PATH.exists():
            try:
                with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []


def save_watchlist(watchlist: list) -> None:
    with _watchlist_lock:
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
            json.dump(watchlist, f, ensure_ascii=False, indent=2)


# ==================== Stock Profiles ====================

def load_stock_profile(sector: str) -> dict | None:
    safe_name = sector.replace("/", "_").replace("\\", "_")
    path = STOCK_PROFILES_DIR / f"{safe_name}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_all_stock_profiles() -> list[dict]:
    results = []
    if STOCK_PROFILES_DIR.exists():
        for fp in STOCK_PROFILES_DIR.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception:
                continue
    return results
