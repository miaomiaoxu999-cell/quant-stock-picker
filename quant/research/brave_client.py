"""Brave Search REST 搜索封装"""

from __future__ import annotations

import time

import requests
from loguru import logger


class BraveClient:
    """Brave Search API 客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, max_results: int = 5, retries: int = 3) -> dict:
        """搜索并返回结果，网络错误时指数退避重试

        Returns:
            {answer: str, results: [{title, url, content, score, published_date}]}
        """
        for attempt in range(retries):
            try:
                resp = requests.get(
                    self.base_url,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": self.api_key,
                    },
                    params={
                        "q": query,
                        "count": max_results,
                        "search_lang": "zh-hans",
                        "country": "cn",
                        "freshness": "pw",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data)
            except requests.exceptions.HTTPError as e:
                logger.warning(f"Brave search HTTP error: {query}, status: {e.response.status_code}")
                return {"answer": "", "results": []}
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Brave search retry {attempt+1}/{retries-1} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.warning(f"Brave search failed after {retries} attempts: {e}")
                    return {"answer": "", "results": []}

    def _parse_response(self, data: dict) -> dict:
        """解析 Brave Search API 响应"""
        results = []
        web_results = data.get("web", {}).get("results", [])

        for r in web_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("description", ""),
                "score": r.get("family_friendly", False) and 1.0 or 0.5,
                "published_date": r.get("page_age", ""),
            })

        return {
            "answer": "",
            "results": results,
        }
