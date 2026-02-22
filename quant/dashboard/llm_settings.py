"""LLM 设置页面 — API Key / URL / 模型管理 + 搜索引擎 Key"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from quant.llm.client import LLMConfig

SETTINGS_PATH = Path(__file__).parent.parent.parent / "data" / "llm_settings.json"

_DEFAULTS = {
    "api_key": "",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "Pro/zai-org/GLM-5",
    "tavily_api_key": "",
    "jina_api_key": "",
    "apify_api_key": "",
    "brave_api_key": "",
    # 审计模型（独立配置，留空则复用主 LLM）
    "audit_api_key": "",
    "audit_base_url": "https://api.siliconflow.cn/v1",
    "audit_model": "deepseek-ai/DeepSeek-V3",
}


def load_llm_settings() -> dict:
    """加载 LLM 设置，不存在则返回默认值"""
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # 补充新增的默认 key（兼容旧配置文件）
            for k, v in _DEFAULTS.items():
                if k not in saved:
                    saved[k] = v
            return saved
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_llm_settings(settings: dict) -> None:
    """保存 LLM 设置到 JSON"""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def get_llm_config() -> LLMConfig | None:
    """读取设置并转为 LLMConfig，api_key 为空则返回 None"""
    s = load_llm_settings()
    if not s.get("api_key"):
        return None
    return LLMConfig(
        api_key=s["api_key"],
        base_url=s.get("base_url", _DEFAULTS["base_url"]),
        model=s.get("model", _DEFAULTS["model"]),
    )


def get_audit_llm_config() -> LLMConfig | None:
    """获取审计模型配置。审计 key 为空时 fallback 到主 LLM。"""
    s = load_llm_settings()
    audit_key = s.get("audit_api_key", "")
    if not audit_key:
        return get_llm_config()
    return LLMConfig(
        api_key=audit_key,
        base_url=s.get("audit_base_url", _DEFAULTS["audit_base_url"]),
        model=s.get("audit_model", _DEFAULTS["audit_model"]),
        max_tokens=4096,
    )


def get_tavily_key() -> str:
    """获取 Tavily API Key"""
    return load_llm_settings().get("tavily_api_key", "")


def get_jina_key() -> str:
    """获取 Jina Reader API Key"""
    return load_llm_settings().get("jina_api_key", "")


def get_apify_key() -> str:
    """获取 Apify API Key"""
    return load_llm_settings().get("apify_api_key", "")


def get_brave_key() -> str:
    """获取 Brave Search API Key"""
    return load_llm_settings().get("brave_api_key", "")


def render_settings_page() -> None:
    """渲染 LLM 设置页面"""
    st.header("LLM 设置")
    st.markdown("配置大模型 API 连接参数，用于 AI 因子生成和周期分析。")
    st.markdown("---")

    settings = load_llm_settings()

    # ---- LLM 配置 ----
    st.subheader("大模型 API")

    api_key = st.text_input(
        "API Key",
        value=settings.get("api_key", ""),
        type="password",
        help="硅基流动或其他 OpenAI 兼容服务的 API Key",
    )
    base_url = st.text_input(
        "API URL",
        value=settings.get("base_url", _DEFAULTS["base_url"]),
        help="API 基础 URL，如 https://api.siliconflow.cn/v1",
    )
    model = st.text_input(
        "模型名称",
        value=settings.get("model", _DEFAULTS["model"]),
        help="模型 ID，如 Pro/zai-org/GLM-5",
    )

    # ---- 搜索引擎 & 数据获取 ----
    st.markdown("---")
    st.subheader("搜索引擎 & 数据获取")
    st.caption("用于「周期分析」页面的多级数据获取。留空则跳过对应数据源。")

    tavily_key = st.text_input(
        "Tavily API Key",
        value=settings.get("tavily_api_key", ""),
        type="password",
        help="Tavily AI 搜索 API Key，用于搜索行业数据",
    )
    jina_key = st.text_input(
        "Jina Reader API Key",
        value=settings.get("jina_api_key", ""),
        type="password",
        help="Jina Reader API Key，用于抓取网页内容",
    )
    apify_key = st.text_input(
        "Apify API Key",
        value=settings.get("apify_api_key", ""),
        type="password",
        help="Apify API Key，用于网页爬虫（最后手段）",
    )

    # ---- 审计模型（可选） ----
    st.markdown("---")
    st.subheader("审计模型（可选）")
    st.caption("独立审计 agent 使用的模型。留空 API Key 则复用上方的主 LLM 配置。建议使用 DeepSeek-V3。")

    audit_key = st.text_input(
        "审计 API Key",
        value=settings.get("audit_api_key", ""),
        type="password",
        help="可选。留空则复用主 LLM 的 API Key",
    )
    audit_base_url = st.text_input(
        "审计 API URL",
        value=settings.get("audit_base_url", _DEFAULTS["audit_base_url"]),
    )
    audit_model = st.text_input(
        "审计模型名称",
        value=settings.get("audit_model", _DEFAULTS["audit_model"]),
        help="建议: deepseek-ai/DeepSeek-V3",
    )

    if st.button("保存设置", type="primary"):
        new_settings = {
            "api_key": api_key.strip(),
            "base_url": base_url.strip(),
            "model": model.strip(),
            "tavily_api_key": tavily_key.strip(),
            "jina_api_key": jina_key.strip(),
            "apify_api_key": apify_key.strip(),
            "audit_api_key": audit_key.strip(),
            "audit_base_url": audit_base_url.strip(),
            "audit_model": audit_model.strip(),
        }
        save_llm_settings(new_settings)
        st.success("设置已保存")
