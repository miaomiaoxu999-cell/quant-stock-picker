"""硅基流动 / OpenAI 兼容 LLM 客户端"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Generator

import requests


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Pro/zai-org/GLM-5"
    temperature: float = 0.7
    max_tokens: int = 2048


class LLMError(Exception):
    """LLM 调用异常，包含中文错误信息"""
    pass


class SiliconFlowClient:
    """硅基流动 LLM 客户端，兼容 OpenAI Chat Completions API"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._url = f"{config.base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[dict], stream: bool = False) -> dict:
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }

    def _handle_error(self, e: Exception) -> LLMError:
        """将各种异常转换为带中文提示的 LLMError"""
        if isinstance(e, requests.exceptions.ConnectionError):
            return LLMError("无法连接到 API 服务器，请检查网络或 API URL 设置")
        if isinstance(e, requests.exceptions.Timeout):
            return LLMError("API 请求超时，请稍后重试")
        if isinstance(e, requests.exceptions.HTTPError):
            status = e.response.status_code if e.response is not None else 0
            if status == 401:
                return LLMError("API Key 无效，请在设置页面检查 API Key")
            if status == 429:
                return LLMError("请求频率过高，请稍后重试")
            if status == 404:
                return LLMError("模型不存在，请检查模型名称设置")
            body = ""
            try:
                body = e.response.text[:200]
            except Exception:
                pass
            return LLMError(f"API 返回错误 ({status}): {body}")
        return LLMError(f"LLM 调用失败: {e}")

    def chat(self, messages: list[dict], timeout: int = 180) -> str:
        """阻塞式调用，返回完整回复文本"""
        try:
            resp = requests.post(
                self._url,
                headers=self._headers,
                json=self._build_payload(messages, stream=False),
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise self._handle_error(e) from e
        except (KeyError, IndexError) as e:
            raise LLMError(f"API 返回数据格式异常: {e}") from e

    def chat_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """流式调用，逐 token yield，适配 st.write_stream"""
        try:
            resp = requests.post(
                self._url,
                headers=self._headers,
                json=self._build_payload(messages, stream=True),
                timeout=(10, 300),
                stream=True,
            )
            resp.raise_for_status()
            resp.encoding = "utf-8"
        except requests.exceptions.RequestException as e:
            raise self._handle_error(e) from e

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
