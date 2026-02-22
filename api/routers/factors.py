"""Factors router - sector factors CRUD + AI generation/chat (SSE)."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import (
    load_sector_factors,
    save_sector_factors,
    get_llm_config,
)
from api.schemas.factors import (
    SectorFactorsResponse,
    AllSectorsResponse,
    FactorWeightUpdate,
    FactorChatRequest,
    PresetSectorsResponse,
)
from api.services.llm_streaming import stream_sse
from quant.llm.prompts import build_factor_generation_prompt, build_factor_chat_messages
from api.services.factor_utils import (
    extract_json_from_text,
    validate_factors,
    PRESET_SECTORS,
)

router = APIRouter(tags=["factors"])


@router.get("/factors/presets", response_model=PresetSectorsResponse)
def get_preset_sectors():
    """Return the list of preset sector names."""
    return {"sectors": PRESET_SECTORS}


@router.get("/factors", response_model=AllSectorsResponse)
def get_all_factors():
    """Get all sector factors data."""
    data = load_sector_factors()
    return {"sectors": data}


@router.get("/factors/{sector}", response_model=SectorFactorsResponse)
def get_sector_factors(sector: str):
    """Get factors for a specific sector."""
    all_data = load_sector_factors()
    sector_data = all_data.get(sector, {})
    return {
        "sector": sector,
        "factors": sector_data.get("factors", []),
        "conversation": sector_data.get("conversation", []),
        "updated_at": sector_data.get("updated_at"),
    }


@router.put("/factors/{sector}/weights")
def update_factor_weights(sector: str, body: FactorWeightUpdate):
    """Update factor weights for a sector."""
    all_data = load_sector_factors()
    sector_data = all_data.get(sector)
    if not sector_data or not sector_data.get("factors"):
        raise HTTPException(status_code=404, detail=f"Sector '{sector}' has no factors")

    if len(body.weights) != len(sector_data["factors"]):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(sector_data['factors'])} weights, got {len(body.weights)}",
        )

    total = sum(body.weights)
    if not (95 <= total <= 105):
        raise HTTPException(status_code=400, detail=f"Weights must sum to ~100, got {total}")

    for i, w in enumerate(body.weights):
        sector_data["factors"][i]["weight"] = w

    sector_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    all_data[sector] = sector_data
    save_sector_factors(all_data)
    return {"status": "ok", "factors": sector_data["factors"]}


@router.post("/factors/{sector}/generate")
def generate_factors(sector: str):
    """SSE endpoint: AI generates factors for a sector."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM API Key not configured")

    messages = build_factor_generation_prompt(sector)

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, messages):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                # Parse and save factors
                parsed = extract_json_from_text(full_text)
                if parsed:
                    valid = validate_factors(parsed)
                    if valid:
                        all_data = load_sector_factors()
                        sector_data = all_data.get(sector, {})
                        sector_data["factors"] = valid
                        sector_data["conversation"] = []
                        sector_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
                        all_data[sector] = sector_data
                        save_sector_factors(all_data)
                        yield f"data: {json.dumps({'type': 'factors_saved', 'factors': valid}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'parse_error', 'message': 'Factor validation failed'}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'parse_error', 'message': 'Could not extract JSON from response'}, ensure_ascii=False)}\n\n"

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/factors/{sector}/chat")
def chat_factors(sector: str, body: FactorChatRequest):
    """SSE endpoint: chat about sector factors."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM API Key not configured")

    all_data = load_sector_factors()
    sector_data = all_data.get(sector, {})
    current_factors = sector_data.get("factors", [])
    factors_json = json.dumps({"factors": current_factors}, ensure_ascii=False, indent=2)

    llm_messages = build_factor_chat_messages(
        sector, factors_json, body.history, body.message,
    )

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_messages):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                # Check if reply contains updated factors
                parsed = extract_json_from_text(full_text)
                if parsed:
                    valid = validate_factors(parsed)
                    if valid:
                        sector_data["factors"] = valid
                        # Append to conversation
                        conversation = sector_data.get("conversation", [])
                        conversation.append({"role": "user", "content": body.message})
                        conversation.append({"role": "assistant", "content": full_text})
                        if len(conversation) > 40:
                            conversation = conversation[-40:]
                        sector_data["conversation"] = conversation
                        sector_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
                        all_data[sector] = sector_data
                        save_sector_factors(all_data)
                        yield f"data: {json.dumps({'type': 'factors_updated', 'factors': valid}, ensure_ascii=False)}\n\n"
                else:
                    # No factor update, just save conversation
                    conversation = sector_data.get("conversation", [])
                    conversation.append({"role": "user", "content": body.message})
                    conversation.append({"role": "assistant", "content": full_text})
                    if len(conversation) > 40:
                        conversation = conversation[-40:]
                    sector_data["conversation"] = conversation
                    sector_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    all_data[sector] = sector_data
                    save_sector_factors(all_data)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
