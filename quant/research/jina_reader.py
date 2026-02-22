"""Jina Reader 网页抓取封装"""

from __future__ import annotations

import time

import requests
from loguru import logger


class JinaReader:
    """Jina Reader API — 将网页转为 Markdown 正文"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def read_url(self, url: str, max_chars: int = 5000, retries: int = 3) -> str:
        """抓取网页并返回 Markdown 正文，网络错误时指数退避重试"""
        for attempt in range(retries):
            try:
                resp = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "text/markdown",
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                return resp.text[:max_chars]
            except requests.exceptions.HTTPError:
                logger.warning(f"Jina read HTTP error for {url}")
                return ""
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Jina read retry {attempt+1}/{retries-1} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.warning(f"Jina read failed after {retries} attempts for {url}: {e}")
                    return ""
