"""Tavily REST 搜索封装"""

from __future__ import annotations

import time

import requests
from loguru import logger


class TavilyClient:
    """Tavily AI 搜索 API 客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5, retries: int = 3) -> dict:
        """搜索并返回结果，网络错误时指数退避重试

        Returns:
            {answer: str, results: [{title, url, content, score}]}
        """
        for attempt in range(retries):
            try:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                        "include_answer": True,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError:
                logger.warning(f"Tavily search HTTP error: {query}")
                return {"answer": "", "results": []}
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Tavily search retry {attempt+1}/{retries-1} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.warning(f"Tavily search failed after {retries} attempts: {e}")
                    return {"answer": "", "results": []}
