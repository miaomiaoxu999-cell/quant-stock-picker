"""多级回退数据获取编排器 — 绝不造假数据"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

from quant.llm.client import SiliconFlowClient, LLMConfig, LLMError
from quant.llm.prompts import build_data_source_guidance_prompt
from quant.research.tavily_client import TavilyClient
from quant.research.jina_reader import JinaReader
from quant.research.apify_client import ApifyClient
from quant.research.brave_client import BraveClient


@dataclass
class FactorData:
    """单个因子的数据获取结果"""
    found: bool
    data_points: list = field(default_factory=list)   # AKShare 数值数据
    raw_text: str = ""          # 网页抓取的文本
    search_snippets: list = field(default_factory=list)  # Tavily 摘要
    news: list = field(default_factory=list)  # 新闻条目 [{title, url, snippet}]
    source: str = ""            # 数据来源标注
    reason: str = ""            # 找不到时的原因说明
    # ---- 中间过程数据 ----
    guidance: dict = field(default_factory=dict)           # LLM 数据源指导结果
    search_results_raw: list = field(default_factory=list)  # Tavily 原始搜索结果
    fetch_log: list = field(default_factory=list)           # 抓取日志 [{url, method, success, chars}]


def _extract_json_from_text(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON"""
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _contains_useful_data(content: str, factor_name: str) -> bool:
    """判断抓取内容是否包含有用数据"""
    if len(content) < 100:
        return False
    # 检查是否包含数字（价格、百分比等）
    numbers = re.findall(r"\d+\.?\d*", content)
    if len(numbers) < 3:
        return False
    # 检查关键词
    keywords = ["价格", "price", "走势", "趋势", "历史", "数据",
                 "万元", "元/吨", "%", "同比", "环比", "产量", "库存"]
    match_count = sum(1 for kw in keywords if kw in content)
    return match_count >= 2


class DataFetcher:
    """多级回退数据获取：LLM指导 -> AKShare -> Tavily/Brave -> Jina -> Apify"""

    def __init__(
        self,
        llm_client: SiliconFlowClient,
        tavily: Optional[TavilyClient] = None,
        jina: Optional[JinaReader] = None,
        apify: Optional[ApifyClient] = None,
        brave: Optional[BraveClient] = None,
        akshare_provider=None,
    ):
        self.llm = llm_client
        self.tavily = tavily
        self.jina = jina
        self.apify = apify
        self.brave = brave
        self.akshare = akshare_provider
        # 归档数据
        self._archive: dict[str, Any] = {}

    def fetch_factor_data(
        self,
        sector: str,
        factor: dict,
        progress_callback: Optional[Callable] = None,
    ) -> FactorData:
        """获取单个因子的真实历史数据（多级回退）

        progress_callback 签名: (step: str, status: str, detail: Any) -> None
            step:   "guidance" | "akshare" | "tavily" | "jina" | "apify"
            status: "start" | "done" | "fail" | "skip"
            detail: 对应数据
        """
        name = factor["name"]
        description = factor.get("description", "")
        data_source = factor.get("data_source", "")
        fetch_log: list[dict] = []

        def cb(step: str, status: str, detail: Any = None):
            if progress_callback:
                progress_callback(step, status, detail)

        # ---- Step 1: 问 LLM 去哪找 ----
        cb("guidance", "start", {"factor": name})
        guidance = self._ask_llm_for_source(sector, name, description, data_source)
        self._archive[f"guidance_{name}"] = guidance
        cb("guidance", "done", guidance)

        # ---- Step 2: AKShare 直接拉 ----
        akshare_api = guidance.get("akshare_api")
        if akshare_api and self.akshare:
            cb("akshare", "start", {"api": akshare_api})
            data = self._try_akshare(akshare_api, factor)
            if data:
                cb("akshare", "done", {"records": len(data)})
                return FactorData(
                    found=True,
                    data_points=data,
                    source=f"AKShare ({akshare_api})",
                    guidance=guidance,
                    fetch_log=[{"url": f"akshare://{akshare_api}", "method": "akshare", "success": True, "chars": 0}],
                )
            else:
                cb("akshare", "fail", {"api": akshare_api})
        elif akshare_api:
            cb("akshare", "skip", {"reason": "AKShare provider not configured"})

        # ---- Step 3: Tavily + Brave 并行搜索 ----
        search_results = []
        news_items = []
        queries = guidance.get("search_queries", [])
        if not queries:
            queries = [f"{sector} {name} 历史数据 走势"]

        # Tavily 搜索
        if self.tavily:
            cb("tavily", "start", {"queries": queries[:3]})
            for query in queries[:3]:
                try:
                    results = self.tavily.search(query, max_results=3)
                except Exception as e:
                    logger.debug(f"Tavily search failed for '{query}': {e}")
                    fetch_log.append({"url": query, "method": "tavily", "success": False, "chars": 0})
                    continue
                self._archive[f"tavily_{name}_{query[:20]}"] = results
                result_list = results.get("results", [])
                fetch_log.append({"url": query, "method": "tavily", "success": len(result_list) > 0, "chars": len(result_list)})
                for r in result_list:
                    r["_source"] = "tavily"
                    search_results.append(r)

        # Brave 搜索（并行补充）
        if self.brave:
            cb("brave", "start", {"queries": queries[:3]})
            for query in queries[:3]:
                try:
                    results = self.brave.search(query, max_results=3)
                except Exception as e:
                    logger.debug(f"Brave search failed for '{query}': {e}")
                    fetch_log.append({"url": query, "method": "brave", "success": False, "chars": 0})
                    continue
                self._archive[f"brave_{name}_{query[:20]}"] = results
                result_list = results.get("results", [])
                fetch_log.append({"url": query, "method": "brave", "success": len(result_list) > 0, "chars": len(result_list)})
                for r in result_list:
                    r["_source"] = "brave"
                    search_results.append(r)

        # 去重 URL，按发布时间排序（最新的优先）
        seen_urls = set()
        unique_results = []
        for r in search_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # 按发布时间排序（有 published_date 的排前面）
        def get_date_key(r):
            date_str = r.get("published_date", "")
            return date_str if date_str else ""
        unique_results.sort(key=get_date_key, reverse=True)
        search_results = unique_results

        if self.tavily or self.brave:
            if search_results:
                sources = [r.get("_source", "") for r in search_results[:5]]
                cb("search", "done", {"count": len(search_results), "sources": list(set(sources))})
            else:
                cb("search", "fail", {"reason": "No results from any query"})

        # 构建 news_items
        for r in search_results[:10]:
            news_items.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:200],
            })

        # ---- Step 4: Jina Reader 抓取 top URL ----
        detailed_text = ""
        top_urls = [r["url"] for r in search_results[:5] if r.get("url")]
        suggested_urls = guidance.get("suggested_urls", [])

        if self.jina:
            for url in top_urls[:3]:
                cb("jina", "start", {"url": url})
                try:
                    content = self.jina.read_url(url)
                    if content and _contains_useful_data(content, name):
                        detailed_text = content
                        self._archive[f"jina_{name}"] = content
                        fetch_log.append({"url": url, "method": "jina", "success": True, "chars": len(content)})
                        cb("jina", "done", {"url": url, "chars": len(content)})
                        break
                    else:
                        fetch_log.append({"url": url, "method": "jina", "success": False, "chars": len(content) if content else 0})
                        cb("jina", "fail", {"url": url})
                except Exception as e:
                    logger.debug(f"Jina failed for {url}: {e}")
                    fetch_log.append({"url": url, "method": "jina", "success": False, "chars": 0})
                    cb("jina", "fail", {"url": url})
                    continue

        # ---- Step 5: 尝试 suggested_urls（Jina 或直接 requests） ----
        if not detailed_text and suggested_urls and self.jina:
            for url in suggested_urls[:2]:
                cb("jina", "start", {"url": url})
                try:
                    content = self.jina.read_url(url)
                    if content and _contains_useful_data(content, name):
                        detailed_text = content
                        self._archive[f"jina_suggested_{name}"] = content
                        fetch_log.append({"url": url, "method": "jina", "success": True, "chars": len(content)})
                        cb("jina", "done", {"url": url, "chars": len(content)})
                        break
                    else:
                        fetch_log.append({"url": url, "method": "jina", "success": False, "chars": len(content) if content else 0})
                        cb("jina", "fail", {"url": url})
                except Exception as e:
                    logger.debug(f"Jina suggested URL failed for {url}: {e}")
                    fetch_log.append({"url": url, "method": "jina", "success": False, "chars": 0})
                    cb("jina", "fail", {"url": url})
                    continue

        # ---- Step 6: Apify 最后手段 ----
        if not detailed_text and self.apify:
            for url in (top_urls[:1] + suggested_urls[:1]):
                if not url:
                    continue
                cb("apify", "start", {"url": url})
                try:
                    content = self.apify.scrape_url(url)
                    if content and _contains_useful_data(content, name):
                        detailed_text = content
                        self._archive[f"apify_{name}"] = content
                        fetch_log.append({"url": url, "method": "apify", "success": True, "chars": len(content)})
                        cb("apify", "done", {"url": url, "chars": len(content)})
                        break
                    else:
                        fetch_log.append({"url": url, "method": "apify", "success": False, "chars": len(content) if content else 0})
                        cb("apify", "fail", {"url": url})
                except Exception as e:
                    logger.debug(f"Apify failed for {url}: {e}")
                    fetch_log.append({"url": url, "method": "apify", "success": False, "chars": 0})
                    cb("apify", "fail", {"url": url})
                    continue

        # ---- 汇总 ----
        if detailed_text or search_results:
            sources = []
            if detailed_text:
                sources.append("Jina/Apify")
            if search_results:
                sources.append("Tavily")
            return FactorData(
                found=True,
                raw_text=detailed_text,
                search_snippets=[r.get("content", "")[:300] for r in search_results[:5]],
                news=_dedupe_news(news_items),
                source=" + ".join(sources),
                guidance=guidance,
                search_results_raw=search_results,
                fetch_log=fetch_log,
            )
        else:
            return FactorData(
                found=False,
                reason=f"所有数据源均未找到「{name}」的历史数据",
                guidance=guidance,
                search_results_raw=search_results,
                fetch_log=fetch_log,
            )

    def _ask_llm_for_source(
        self, sector: str, factor_name: str, description: str, data_source: str,
    ) -> dict:
        """问 LLM 去哪找数据"""
        messages = build_data_source_guidance_prompt(
            sector, factor_name, description, data_source,
        )
        try:
            reply = self.llm.chat(messages)
            parsed = _extract_json_from_text(reply)
            if parsed:
                return parsed
        except LLMError as e:
            logger.warning(f"LLM guidance failed: {e}")
        # 返回默认搜索词
        return {
            "suggested_urls": [],
            "akshare_api": None,
            "search_queries": [
                f"{sector} {factor_name} 历史数据",
                f"{factor_name} price history trend",
                f"{sector} {factor_name} 走势 周期",
            ],
        }

    # 只允许调用这些 AKShare 函数（安全白名单，防止 LLM 注入）
    SAFE_AKSHARE_FUNCTIONS = {
        "futures_main_sina", "futures_zh_spot", "futures_zh_daily_sina",
        "spot_hist_sge", "spot_golden_benchmark",
        "stock_zh_a_hist",
        "index_zh_a_hist", "index_zh_a_hist_min_em",
        "macro_china_pmi", "macro_china_cpi",
        "bond_zh_hs_cov_spot", "commodity_futures_daily",
    }

    def _try_akshare(self, api_call: str, factor: dict) -> list | None:
        """尝试调用 AKShare 对应接口（白名单限制）"""
        try:
            import akshare as ak

            # 解析 LLM 返回的 api 调用字符串，如 "futures_main_sina(symbol='LC0')"
            match = re.match(r"(\w+)\((.*)\)", api_call.strip())
            if not match:
                return None

            func_name = match.group(1)
            args_str = match.group(2)

            # 安全检查：只允许白名单内的函数
            if func_name not in self.SAFE_AKSHARE_FUNCTIONS:
                logger.warning(f"AKShare function not in allowlist: {func_name}")
                return None

            func = getattr(ak, func_name, None)
            if func is None:
                logger.warning(f"AKShare has no function: {func_name}")
                return None

            # 解析参数（只允许简单字符串值）
            kwargs = {}
            if args_str.strip():
                for part in args_str.split(","):
                    part = part.strip()
                    if "=" in part:
                        k, v = part.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        kwargs[k] = v

            df = func(**kwargs)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return None

            if isinstance(df, pd.DataFrame):
                records = df.tail(100).to_dict("records")
                return records
            return None

        except Exception as e:
            logger.warning(f"AKShare call failed ({api_call}): {e}")
            return None

    def save_archive(self, archive_dir: Path) -> None:
        """保存原始数据到归档目录"""
        archive_dir.mkdir(parents=True, exist_ok=True)
        for key, data in self._archive.items():
            try:
                if isinstance(data, str):
                    (archive_dir / f"{key}.md").write_text(data, encoding="utf-8")
                elif isinstance(data, (dict, list)):
                    (archive_dir / f"{key}.json").write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception as e:
                logger.warning(f"Failed to save archive {key}: {e}")


def _dedupe_news(items: list[dict]) -> list[dict]:
    """新闻去重"""
    seen = set()
    result = []
    for item in items:
        url = item.get("url", "")
        if url and url not in seen:
            seen.add(url)
            result.append(item)
    return result[:8]
