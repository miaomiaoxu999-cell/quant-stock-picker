"""Audit router - AI red-team audit of analysis results (SSE)."""

from __future__ import annotations

import json
import re
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import (
    load_audit_results,
    save_audit_result,
    load_cycle_analysis,
    load_sector_factors,
    load_stock_profile,
    get_audit_llm_config,
    get_llm_config,
    RESEARCH_DIR,
    STOCK_PROFILES_DIR,
)
from api.schemas.audit import (
    AuditSectorResponse,
    AuditChatRequest,
    AuditFeedbackRequest,
    AuditSectorsResponse,
)
from api.services.llm_streaming import stream_sse
from quant.llm.prompts import build_audit_prompt, build_audit_chat_messages
from api.services.factor_utils import extract_json_from_text

router = APIRouter(tags=["audit"])

_VALID_AUDIT_TYPES = {"factors", "cycle", "stock", "full"}


def _load_sector_complete_data(sector: str) -> dict:
    """Load all analysis data for a sector (factors + cycle + stocks + archive)."""
    data = {"sector": sector, "found": False}

    # Factors
    all_factors = load_sector_factors()
    if sector in all_factors:
        data["factors_config"] = all_factors[sector]
        data["found"] = True

    # Cycle
    all_cycles = load_cycle_analysis()
    if sector in all_cycles:
        data["cycle_analysis"] = all_cycles[sector]
        data["found"] = True

    # Stock profiles
    profile = load_stock_profile(sector)
    if profile:
        data["stock_profiles"] = profile
        data["found"] = True

    # Archive
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", sector)
    sector_research_dir = RESEARCH_DIR / safe_name
    if sector_research_dir.exists():
        subdirs = sorted(
            [d for d in sector_research_dir.iterdir() if d.is_dir()], reverse=True,
        )
        if subdirs:
            latest = subdirs[0]
            data["archive_path"] = str(latest)
            data["archive_file_count"] = len(list(latest.glob("*")))

    return data


def _parse_audit_json(text: str) -> dict | None:
    """Extract audit report JSON from LLM response."""
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if "risk_level" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


@router.get("/audit/sectors", response_model=AuditSectorsResponse)
def get_auditable_sectors():
    """List sectors with enough data for auditing."""
    sectors = set()
    cycle_data = load_cycle_analysis()
    sectors.update(cycle_data.keys())
    all_factors = load_sector_factors()
    sectors.update(all_factors.keys())
    return {"sectors": sorted(sectors)}


@router.get("/audit/{sector}", response_model=AuditSectorResponse)
def get_audit_results(sector: str):
    """All audit results for a sector."""
    all_results = load_audit_results()
    sector_results = all_results.get(sector, {})
    return {"sector": sector, "results": sector_results}


@router.post("/audit/{sector}/{audit_type}/run")
def run_audit(sector: str, audit_type: str):
    """SSE endpoint: run audit analysis."""
    if audit_type not in _VALID_AUDIT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid audit type: '{audit_type}'")

    llm_config = get_audit_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="Audit LLM not configured")

    sector_data = _load_sector_complete_data(sector)
    if not sector_data.get("found"):
        raise HTTPException(status_code=400, detail=f"No data for sector '{sector}'")

    messages = build_audit_prompt(sector, sector_data, audit_type)

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, messages):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                parsed = _parse_audit_json(full_text)
                if parsed:
                    audit_result = {
                        "report": parsed,
                        "raw_response": full_text,
                        "conversation": [],
                        "audited_at": datetime.now().isoformat(timespec="seconds"),
                        "sector_data_snapshot": {
                            "cycle_position": sector_data.get("cycle_analysis", {}).get("overall", {}).get("cycle_position"),
                            "factors_count": len(sector_data.get("factors_config", {}).get("factors", [])),
                            "stocks_count": len(sector_data.get("stock_profiles", {}).get("stocks", [])),
                        },
                    }
                    save_audit_result(sector, audit_type, audit_result)
                    yield f"data: {json.dumps({'type': 'audit_saved', 'report': parsed}, ensure_ascii=False)}\n\n"
                else:
                    audit_result = {
                        "report": None,
                        "raw_response": full_text,
                        "conversation": [],
                        "audited_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    save_audit_result(sector, audit_type, audit_result)
                    yield f"data: {json.dumps({'type': 'parse_error', 'message': 'Could not parse audit JSON'}, ensure_ascii=False)}\n\n"

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/audit/{sector}/{audit_type}/chat")
def chat_audit(sector: str, audit_type: str, body: AuditChatRequest):
    """SSE endpoint: chat about audit results."""
    if audit_type not in _VALID_AUDIT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid audit type: '{audit_type}'")

    llm_config = get_audit_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="Audit LLM not configured")

    all_results = load_audit_results()
    audit_data = all_results.get(sector, {}).get(audit_type, {})
    if not audit_data:
        raise HTTPException(status_code=404, detail=f"No audit results for '{sector}' / '{audit_type}'")

    llm_messages = build_audit_chat_messages(sector, audit_data, body.history, body.message)

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_messages):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                # Save conversation
                conversation = audit_data.get("conversation", [])
                conversation.append({"role": "user", "content": body.message})
                conversation.append({"role": "assistant", "content": full_text})
                if len(conversation) > 40:
                    conversation = conversation[-40:]
                audit_data["conversation"] = conversation
                save_audit_result(sector, audit_type, audit_data)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/audit/{sector}/{audit_type}/feedback")
def submit_audit_feedback(sector: str, audit_type: str, body: AuditFeedbackRequest):
    """SSE endpoint: inject audit feedback and auto-trigger reanalysis of affected module."""
    if audit_type not in _VALID_AUDIT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid audit type: '{audit_type}'")

    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM not configured")

    from quant.llm.client import SiliconFlowClient
    from quant.llm.prompts import build_factor_chat_messages, build_cycle_chat_messages
    from api.deps import load_sector_factors, save_sector_factors, save_cycle_analysis
    from api.services.factor_utils import validate_factors

    def event_stream():
        feedback_msg = body.feedback

        if audit_type in ("factors", "full"):
            # Inject feedback into factors
            all_factors = load_sector_factors()
            sector_data = all_factors.get(sector, {})
            if sector_data:
                factors_json = json.dumps(sector_data.get("factors", []), ensure_ascii=False, indent=2)
                history = sector_data.get("conversation", [])
                msgs = build_factor_chat_messages(sector, factors_json, history, feedback_msg)

                full_text = ""
                for event in stream_sse(llm_config, msgs):
                    full_text += event.get("content", "")
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                    if event.get("type") == "done":
                        # Save conversation
                        history.append({"role": "user", "content": feedback_msg})
                        history.append({"role": "assistant", "content": full_text})
                        if len(history) > 40:
                            history = history[-40:]
                        sector_data["conversation"] = history

                        # Try extract updated factors
                        parsed = extract_json_from_text(full_text)
                        if parsed:
                            valid = validate_factors(parsed)
                            if valid:
                                sector_data["factors"] = valid
                                sector_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
                                yield f"data: {json.dumps({'type': 'factors_updated', 'factors': valid}, ensure_ascii=False)}\n\n"

                        all_factors[sector] = sector_data
                        save_sector_factors(all_factors)

                    if event.get("type") == "error":
                        break

        elif audit_type == "cycle":
            all_cycles = load_cycle_analysis()
            sector_data = all_cycles.get(sector, {})
            if sector_data:
                history = sector_data.get("conversation", [])
                msgs = build_cycle_chat_messages(sector, sector_data, history, feedback_msg)

                full_text = ""
                for event in stream_sse(llm_config, msgs):
                    full_text += event.get("content", "")
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                    if event.get("type") == "done":
                        history.append({"role": "user", "content": feedback_msg})
                        history.append({"role": "assistant", "content": full_text})
                        if len(history) > 40:
                            history = history[-40:]
                        sector_data["conversation"] = history

                        # Try extract updated cycle JSON
                        m = re.search(r"```json\s*\n?(.*?)\n?\s*```", full_text, re.DOTALL)
                        candidate = None
                        if m:
                            try:
                                candidate = json.loads(m.group(1))
                            except json.JSONDecodeError:
                                pass

                        if candidate and "overall" in candidate:
                            for key in ("news", "archive_path", "analyzed_at", "factors", "conversation"):
                                if key in sector_data and key not in candidate:
                                    candidate[key] = sector_data[key]
                            sector_data.update(candidate)
                            yield f"data: {json.dumps({'type': 'cycle_updated', 'overall': sector_data.get('overall', {})}, ensure_ascii=False)}\n\n"

                        all_cycles[sector] = sector_data
                        save_cycle_analysis(all_cycles)

                    if event.get("type") == "error":
                        break

        elif audit_type == "stock":
            # Stock audit feedback triggers reanalysis
            yield f"data: {json.dumps({'type': 'chunk', 'content': 'Stock reanalysis will be triggered.'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'content': 'Use POST /stocks/{sector}/redo to trigger reanalysis with audit parameters.'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
