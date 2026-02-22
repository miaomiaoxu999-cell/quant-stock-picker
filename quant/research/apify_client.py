"""Apify 网页抓取封装"""

from __future__ import annotations

import time

import requests
from loguru import logger


class ApifyClient:
    """Apify Web Scraper — 抓取网页内容的最后手段"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.apify.com/v2"

    def scrape_url(self, url: str, max_chars: int = 5000) -> str:
        """用 Apify Web Scraper actor 抓取网页内容"""
        try:
            # 启动 actor run
            run_resp = requests.post(
                f"{self.base_url}/acts/apify~web-scraper/runs",
                params={"token": self.api_key},
                json={
                    "startUrls": [{"url": url}],
                    "maxPagesPerCrawl": 1,
                    "pageFunction": """async function pageFunction(context) {
                        const $ = context.jQuery;
                        const text = $('body').text().replace(/\\s+/g, ' ').trim();
                        return { url: context.request.url, text: text.substring(0, 8000) };
                    }""",
                },
                timeout=30,
            )
            run_resp.raise_for_status()
            run_data = run_resp.json()
            run_id = run_data["data"]["id"]

            # 轮询等待完成（最多 60 秒）
            for _ in range(12):
                time.sleep(5)
                status_resp = requests.get(
                    f"{self.base_url}/actor-runs/{run_id}",
                    params={"token": self.api_key},
                    timeout=10,
                )
                status_resp.raise_for_status()
                status = status_resp.json()["data"]["status"]
                if status == "SUCCEEDED":
                    break
                if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    logger.warning(f"Apify run {run_id} ended with status: {status}")
                    return ""
            else:
                logger.warning(f"Apify run {run_id} timed out waiting")
                return ""

            # 获取结果
            dataset_id = run_data["data"]["defaultDatasetId"]
            items_resp = requests.get(
                f"{self.base_url}/datasets/{dataset_id}/items",
                params={"token": self.api_key, "format": "json"},
                timeout=10,
            )
            items_resp.raise_for_status()
            items = items_resp.json()
            if items:
                text = items[0].get("text", "")
                return text[:max_chars]
            return ""

        except requests.exceptions.RequestException as e:
            logger.warning(f"Apify scrape failed for {url}: {e}")
            return ""
