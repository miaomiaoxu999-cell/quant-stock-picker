"""Cycle analysis router - multi-source data fetch + LLM analysis (SSE)."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import (
    load_llm_settings,
    load_cycle_analysis,
    save_cycle_analysis,
    load_sector_factors,
    get_llm_config,
    RESEARCH_DIR,
)
from api.schemas.cycle import (
    AllCyclesResponse,
    CycleSectorResponse,
    CycleChatRequest,
    CycleProgressResponse,
)
from api.services.llm_streaming import stream_sse
from api.services.task_manager import task_manager, TaskStatus
from quant.llm.client import SiliconFlowClient, LLMConfig
from quant.llm.prompts import build_cycle_analysis_prompt, build_cycle_chat_messages
from quant.research.data_fetcher import DataFetcher, _dedupe_news as dedupe_news

router = APIRouter(tags=["cycle"])

# Shared progress dicts keyed by sector (safe for thread access)
_cycle_progress: dict[str, dict] = {}


def _safe_dirname(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


# ==================== Endpoints ====================


@router.get("/cycle", response_model=AllCyclesResponse)
def list_cycles():
    """List all analyzed cycles."""
    return {"sectors": load_cycle_analysis()}


@router.get("/cycle/{sector}", response_model=CycleSectorResponse)
def get_cycle(sector: str):
    """Get cycle data for a single sector."""
    data = load_cycle_analysis()
    sector_data = data.get(sector)
    if sector_data is None:
        raise HTTPException(status_code=404, detail=f"Sector '{sector}' not found")
    return {
        "sector": sector,
        "overall": sector_data.get("overall", {}),
        "factors": sector_data.get("factors", []),
        "news": sector_data.get("news", []),
        "conversation": sector_data.get("conversation", []),
        "analyzed_at": sector_data.get("analyzed_at"),
        "archive_path": sector_data.get("archive_path"),
    }


@router.post("/cycle/{sector}/analyze")
def analyze_cycle(sector: str):
    """SSE endpoint: run full cycle analysis (data fetch + LLM) in background."""
    # Validate all required API keys
    settings = load_llm_settings()
    missing = []
    if not settings.get("api_key"):
        missing.append("LLM API Key")
    if not settings.get("tavily_api_key"):
        missing.append("Tavily API Key")
    if not settings.get("jina_api_key"):
        missing.append("Jina API Key")
    if not settings.get("apify_api_key"):
        missing.append("Apify API Key")
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing API keys: {', '.join(missing)}")

    # Load factors for sector
    all_factors = load_sector_factors()
    sector_factors = all_factors.get(sector, {}).get("factors", [])
    if not sector_factors:
        raise HTTPException(status_code=400, detail=f"Sector '{sector}' has no factors configured")

    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM API Key not configured")

    # Initialize shared progress dict
    progress = {
        "sector": sector,
        "status": "running",
        "current_step": "init",
        "current_factor": "",
        "factor_index": 0,
        "factor_total": 0,
        "log": [],
        "factor_details": [],
        "result": None,
        "error": None,
        "llm_tail": "",
    }
    _cycle_progress[sector] = progress

    task_id = f"cycle_analysis_{sector}"

    def worker(_cancel_event=None):
        _run_cycle_analysis(sector, sector_factors, llm_config, settings, progress, _cancel_event)

    task_manager.submit(worker, task_id=task_id)

    def event_stream():
        while True:
            status = progress.get("status", "running")
            if status == "running":
                yield f"data: {json.dumps({'type': 'progress', 'step': progress.get('current_step', ''), 'factor_index': progress.get('factor_index', 0), 'factor_total': progress.get('factor_total', 0), 'current_factor': progress.get('current_factor', ''), 'log': progress.get('log', [])[-5:]}, ensure_ascii=False)}\n\n"
                time.sleep(2)
            elif status == "completed":
                result = progress.get("result", {})
                yield f"data: {json.dumps({'type': 'done', 'content': 'Analysis completed'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'analysis_saved', 'result': result}, ensure_ascii=False)}\n\n"
                _cycle_progress.pop(sector, None)
                break
            elif status == "cancelled":
                yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis cancelled'}, ensure_ascii=False)}\n\n"
                _cycle_progress.pop(sector, None)
                break
            else:  # failed
                yield f"data: {json.dumps({'type': 'error', 'message': progress.get('error', 'Unknown error')}, ensure_ascii=False)}\n\n"
                _cycle_progress.pop(sector, None)
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.delete("/cycle/{sector}/analyze")
def cancel_cycle_analysis(sector: str):
    """Cancel running cycle analysis."""
    task_id = f"cycle_analysis_{sector}"
    cancelled = task_manager.cancel(task_id)
    if cancelled:
        progress = _cycle_progress.get(sector)
        if progress:
            progress["status"] = "cancelled"
        return {"status": "cancelled"}
    return {"status": "not_running"}


@router.post("/cycle/{sector}/chat")
def chat_cycle(sector: str, body: CycleChatRequest):
    """SSE endpoint: chat about sector cycle analysis."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM API Key not configured")

    all_data = load_cycle_analysis()
    sector_data = all_data.get(sector, {})
    if not sector_data:
        raise HTTPException(status_code=404, detail=f"Sector '{sector}' has no cycle analysis")

    llm_messages = build_cycle_chat_messages(sector, sector_data, body.history, body.message)

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_messages):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                # Check if reply contains updated cycle JSON
                parsed = _extract_cycle_json(full_text)
                if parsed and "overall" in parsed:
                    # Preserve metadata
                    parsed["conversation"] = sector_data.get("conversation", [])
                    parsed["analyzed_at"] = sector_data.get("analyzed_at", "")
                    parsed["archive_path"] = sector_data.get("archive_path", "")
                    parsed["news"] = sector_data.get("news", [])
                    all_data[sector] = parsed
                    save_cycle_analysis(all_data)
                    yield f"data: {json.dumps({'type': 'cycle_updated', 'overall': parsed.get('overall', {})}, ensure_ascii=False)}\n\n"

                # Save conversation
                conversation = sector_data.get("conversation", [])
                conversation.append({"role": "user", "content": body.message})
                conversation.append({"role": "assistant", "content": full_text})
                if len(conversation) > 40:
                    conversation = conversation[-40:]
                sector_data["conversation"] = conversation
                all_data[sector] = sector_data
                save_cycle_analysis(all_data)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/cycle/{sector}/progress", response_model=CycleProgressResponse)
def get_cycle_progress(sector: str):
    """Poll progress for a running cycle analysis."""
    progress = _cycle_progress.get(sector)
    if progress is None:
        task = task_manager.get(f"cycle_analysis_{sector}")
        if task is None:
            return {"status": "not_found"}
        return {"status": task.status.value}
    return {
        "status": progress.get("status", "unknown"),
        "current_step": progress.get("current_step", ""),
        "current_factor": progress.get("current_factor", ""),
        "factor_index": progress.get("factor_index", 0),
        "factor_total": progress.get("factor_total", 0),
        "log": progress.get("log", []),
        "error": progress.get("error"),
    }


# ==================== Background Worker ====================


def _run_cycle_analysis(
    sector: str,
    factors: list[dict],
    llm_config: LLMConfig,
    settings: dict,
    progress: dict,
    cancel_event=None,
) -> None:
    """Background worker for cycle analysis."""
    from quant.research.tavily_client import TavilyClient
    from quant.research.jina_reader import JinaReader
    from quant.research.apify_client import ApifyClient
    from quant.research.brave_client import BraveClient

    try:
        tavily_key = settings.get("tavily_api_key", "")
        jina_key = settings.get("jina_api_key", "")
        apify_key = settings.get("apify_api_key", "")
        brave_key = settings.get("brave_api_key", "")

        top_factors = sorted(factors, key=lambda f: f.get("weight", 0), reverse=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = RESEARCH_DIR / _safe_dirname(sector) / timestamp

        fetcher = DataFetcher(
            llm_client=SiliconFlowClient(llm_config),
            tavily=TavilyClient(tavily_key) if tavily_key else None,
            jina=JinaReader(jina_key) if jina_key else None,
            apify=ApifyClient(apify_key) if apify_key else None,
            brave=BraveClient(brave_key) if brave_key else None,
        )

        factor_data_map = {}
        all_news = []

        for i, f in enumerate(top_factors):
            if cancel_event and cancel_event.is_set():
                progress["status"] = "cancelled"
                return

            progress["current_factor"] = f["name"]
            progress["factor_index"] = i + 1
            progress["factor_total"] = len(top_factors)
            progress["current_step"] = "fetch"
            progress["log"].append(f"({i+1}/{len(top_factors)}) Fetching '{f['name']}' data...")

            def on_step(step, status, detail, _name=f["name"]):
                if step == "guidance" and status == "done":
                    progress["log"].append(f"  AI data source guidance complete")
                elif step == "search" and status == "done":
                    n = detail.get("count", 0) if isinstance(detail, dict) else 0
                    sources = detail.get("sources", []) if isinstance(detail, dict) else []
                    progress["log"].append(f"  Search complete: {n} results from {', '.join(sources)}")
                elif step == "tavily" and status == "done":
                    n = len(detail) if isinstance(detail, list) else 0
                    progress["log"].append(f"  Tavily search: {n} results")
                elif step == "brave" and status == "done":
                    progress["log"].append(f"  Brave search complete")
                elif step == "jina" and status == "done":
                    chars = detail.get("chars", 0) if isinstance(detail, dict) else 0
                    progress["log"].append(f"  Web scrape: {chars} chars")
                elif step == "akshare" and status == "done":
                    n = detail.get("records", 0)
                    progress["log"].append(f"  AKShare: {n} records")

            result = fetcher.fetch_factor_data(sector, f, progress_callback=on_step)
            factor_data_map[f["name"]] = result
            all_news.extend(result.news)

            progress["factor_details"].append({
                "name": f["name"],
                "weight": f.get("weight", 0),
                "status": "completed" if result.found else "failed",
                "source": result.source,
            })

            if result.found:
                progress["log"].append(f"'{f['name']}' data fetched — {result.source}")
            else:
                progress["log"].append(f"'{f['name']}' no data found")

        if cancel_event and cancel_event.is_set():
            progress["status"] = "cancelled"
            return

        fetcher.save_archive(archive_dir)

        has_data = any(r.found for r in factor_data_map.values())
        if not has_data:
            progress["status"] = "failed"
            progress["error"] = "No data found for any factor"
            return

        # LLM analysis
        progress["current_step"] = "llm"
        progress["log"].append("AI analyzing cycles...")
        messages = build_cycle_analysis_prompt(sector, top_factors, factor_data_map)

        client = SiliconFlowClient(LLMConfig(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            model=llm_config.model,
            max_tokens=4096,
        ))

        full_reply = client.chat(messages)

        if cancel_event and cancel_event.is_set():
            progress["status"] = "cancelled"
            return

        # Save raw LLM response
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            (archive_dir / "llm_response.txt").write_text(full_reply, encoding="utf-8")
        except Exception:
            pass

        parsed = _extract_cycle_json(full_reply)
        if parsed and "overall" in parsed:
            parsed["news"] = dedupe_news(all_news)
            parsed["conversation"] = []
            parsed["analyzed_at"] = datetime.now().isoformat(timespec="seconds")
            parsed["archive_path"] = str(archive_dir)

            all_data = load_cycle_analysis()
            all_data[sector] = parsed
            save_cycle_analysis(all_data)

            progress["result"] = parsed
            progress["status"] = "completed"
            progress["log"].append("Analysis complete, results saved.")
        else:
            progress["status"] = "failed"
            progress["error"] = "AI did not return valid analysis"
            progress["llm_tail"] = full_reply[-500:] if full_reply else ""
            progress["log"].append("AI did not return valid results")

    except Exception as e:
        progress["status"] = "failed"
        progress["error"] = str(e)
        progress["log"].append(f"Analysis error: {e}")


def _extract_cycle_json(text: str) -> dict | None:
    """Extract cycle analysis JSON from LLM response."""
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\"overall\"[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None
