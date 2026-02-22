"""Advisor router - 5-step LLM-driven investment wizard."""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response

from api.deps import (
    load_advisor_session,
    save_advisor_session,
    load_portfolio_state,
    save_portfolio_state,
    load_cycle_analysis,
    load_sector_factors,
    load_all_stock_profiles,
    get_llm_config,
    ADVISOR_SESSION_PATH,
)
from api.schemas.advisor import (
    AdvisorSessionResponse,
    Step1SubmitRequest,
    Step2ChatRequest,
    Step2ConfirmRequest,
    Step3ChatRequest,
    Step3ConfirmRequest,
    Step5SaveRequest,
    AvailableStocksResponse,
)
from api.services.llm_streaming import stream_sse
from quant.llm.advisor_prompts import (
    build_diagnosis_messages,
    build_allocation_messages,
    build_prediction_messages,
    build_risk_plan_messages,
)
from quant.utils.constants import STOCK_PROFILES

router = APIRouter(tags=["advisor"])


# ==================== Helper: load all available stocks ====================

def _load_all_available_stocks() -> list[dict]:
    """Merge analyzed stocks from stock_profiles + STOCK_PROFILES constant."""
    stocks = []
    seen_codes: set[str] = set()

    # From STOCK_PROFILES constant
    for code, profile in STOCK_PROFILES.items():
        if code in seen_codes:
            continue
        seen_codes.add(code)
        stocks.append({
            "code": code,
            "name": profile.get("name", ""),
            "source": "core",
        })

    # From stock_profiles dir
    for analysis in load_all_stock_profiles():
        for s in analysis.get("stocks", []):
            code = str(s.get("code", "")).zfill(6)
            if code in seen_codes:
                continue
            seen_codes.add(code)
            val = s.get("valuation", {})
            stocks.append({
                "code": code,
                "name": s.get("name", ""),
                "price": s.get("price"),
                "pb": s.get("pb"),
                "pe_ttm": s.get("pe_ttm"),
                "valuation_status": _derive_valuation_status(val),
                "potential_upside": val.get("upside_to_peak"),
                "cycle_position": s.get("cycle_position", ""),
                "total_score": s.get("total_score"),
                "source": "profile",
            })

    return stocks


def _derive_valuation_status(valuation: dict) -> str:
    current_pb = valuation.get("current_pb", 0)
    peak_pb = valuation.get("cycle_peak_pb", 0)
    if not current_pb or not peak_pb or current_pb <= 0 or peak_pb <= 0:
        return ""
    ratio = current_pb / peak_pb
    if ratio <= 0.30:
        return "严重低估"
    elif ratio <= 0.50:
        return "低估"
    elif ratio <= 0.67:
        return "合理"
    return "高估"


def _get_session() -> dict:
    """Load or create session."""
    session = load_advisor_session()
    if session is None:
        session = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "current_step": 1,
            "inputs": {},
            "confirmed_stocks": [],
            "allocation": [],
            "cash_reserve": {},
            "prediction": {},
            "risk_plan": {},
            "conversations": {"diagnosis": [], "allocation": []},
        }
    return session


def _save(session: dict) -> None:
    session["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_advisor_session(session)


# ==================== Endpoints ====================


@router.get("/advisor/session", response_model=AdvisorSessionResponse)
def get_session():
    """Load advisor session."""
    return {"session": load_advisor_session()}


@router.delete("/advisor/session")
def delete_session():
    """Delete advisor session (reset)."""
    if ADVISOR_SESSION_PATH.exists():
        ADVISOR_SESSION_PATH.unlink()
    return {"status": "ok"}


@router.get("/advisor/available-stocks", response_model=AvailableStocksResponse)
def get_available_stocks():
    """Merge analyzed stocks from stock_profiles + STOCK_PROFILES constant."""
    return {"stocks": _load_all_available_stocks()}


@router.post("/advisor/step1/submit")
def step1_submit(body: Step1SubmitRequest):
    """Save step 1 inputs."""
    session = _get_session()
    session["current_step"] = 2
    session["inputs"] = {
        "total_capital": body.total_capital,
        "bullish_sectors": body.bullish_sectors,
        "favored_stock_codes": body.favored_stock_codes,
    }
    # Build favored stocks detail
    all_stocks = _load_all_available_stocks()
    favored = [s for s in all_stocks if s["code"] in body.favored_stock_codes]
    session["inputs"]["favored_stocks"] = favored
    session["conversations"] = {"diagnosis": [], "allocation": []}
    _save(session)
    return {"status": "ok", "current_step": 2}


@router.post("/advisor/step2/diagnose")
def step2_diagnose():
    """SSE endpoint: AI diagnosis of selected stocks."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM not configured")

    session = _get_session()
    inputs = session.get("inputs", {})
    total_capital = inputs.get("total_capital", 500000)
    bullish = inputs.get("bullish_sectors", [])
    favored = inputs.get("favored_stocks", [])
    portfolio = load_portfolio_state()
    all_stocks = _load_all_available_stocks()
    cycle_data = load_cycle_analysis()

    llm_msgs = build_diagnosis_messages(
        total_capital, bullish, favored, portfolio, all_stocks, cycle_data,
    )

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_msgs):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                conv = session.get("conversations", {})
                diag_conv = conv.get("diagnosis", [])
                diag_conv.append({"role": "assistant", "content": full_text})
                conv["diagnosis"] = diag_conv
                session["conversations"] = conv

                # Try extract recommended stocks
                parsed = _extract_json(full_text)
                if parsed and "recommended_stocks" in parsed:
                    session["diagnosis_json"] = parsed

                _save(session)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/advisor/step2/chat")
def step2_chat(body: Step2ChatRequest):
    """SSE endpoint: chat follow-up for diagnosis."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM not configured")

    session = _get_session()
    inputs = session.get("inputs", {})
    total_capital = inputs.get("total_capital", 500000)
    bullish = inputs.get("bullish_sectors", [])
    favored = inputs.get("favored_stocks", [])
    portfolio = load_portfolio_state()
    all_stocks = _load_all_available_stocks()
    cycle_data = load_cycle_analysis()

    llm_msgs = build_diagnosis_messages(
        total_capital, bullish, favored, portfolio, all_stocks, cycle_data,
    )
    llm_msgs.extend(body.history[-20:])
    llm_msgs.append({"role": "user", "content": body.message})

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_msgs):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                conv = session.get("conversations", {})
                diag_conv = conv.get("diagnosis", [])
                diag_conv.append({"role": "user", "content": body.message})
                diag_conv.append({"role": "assistant", "content": full_text})
                if len(diag_conv) > 40:
                    diag_conv = diag_conv[-40:]
                conv["diagnosis"] = diag_conv
                session["conversations"] = conv

                parsed = _extract_json(full_text)
                if parsed and "recommended_stocks" in parsed:
                    session["diagnosis_json"] = parsed

                _save(session)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/advisor/step2/confirm")
def step2_confirm(body: Step2ConfirmRequest):
    """Confirm stocks and advance to step 3."""
    session = _get_session()
    session["confirmed_stocks"] = body.confirmed_codes
    # Build confirmed stocks detail
    all_stocks = _load_all_available_stocks()
    session["confirmed_stocks_detail"] = [
        s for s in all_stocks if s["code"] in body.confirmed_codes
    ]
    session["current_step"] = 3
    _save(session)
    return {"status": "ok", "current_step": 3, "confirmed_count": len(body.confirmed_codes)}


@router.post("/advisor/step3/allocate")
def step3_allocate():
    """SSE endpoint: AI allocation recommendation."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM not configured")

    session = _get_session()
    inputs = session.get("inputs", {})
    total_capital = inputs.get("total_capital", 500000)
    confirmed = session.get("confirmed_stocks_detail", [])
    portfolio = load_portfolio_state()

    llm_msgs = build_allocation_messages(total_capital, confirmed, portfolio)

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_msgs):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                conv = session.get("conversations", {})
                alloc_conv = conv.get("allocation", [])
                alloc_conv.append({"role": "assistant", "content": full_text})
                conv["allocation"] = alloc_conv
                session["conversations"] = conv

                parsed = _extract_json(full_text)
                if parsed and "allocations" in parsed:
                    session["allocation"] = parsed.get("allocations", [])
                    session["cash_reserve"] = parsed.get("cash_reserve", {})
                    session["allocation_json"] = parsed

                _save(session)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/advisor/step3/chat")
def step3_chat(body: Step3ChatRequest):
    """SSE endpoint: chat follow-up for allocation."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM not configured")

    session = _get_session()
    inputs = session.get("inputs", {})
    total_capital = inputs.get("total_capital", 500000)
    confirmed = session.get("confirmed_stocks_detail", [])
    portfolio = load_portfolio_state()

    llm_msgs = build_allocation_messages(
        total_capital, confirmed, portfolio, body.history[-20:],
    )

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_msgs):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                conv = session.get("conversations", {})
                alloc_conv = conv.get("allocation", [])
                alloc_conv.append({"role": "user", "content": body.message})
                alloc_conv.append({"role": "assistant", "content": full_text})
                if len(alloc_conv) > 40:
                    alloc_conv = alloc_conv[-40:]
                conv["allocation"] = alloc_conv
                session["conversations"] = conv

                parsed = _extract_json(full_text)
                if parsed and "allocations" in parsed:
                    session["allocation"] = parsed.get("allocations", [])
                    session["cash_reserve"] = parsed.get("cash_reserve", {})
                    session["allocation_json"] = parsed

                _save(session)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/advisor/step3/confirm")
def step3_confirm(body: Step3ConfirmRequest):
    """Confirm allocation and advance to step 4."""
    session = _get_session()
    session["allocation"] = body.allocations
    session["cash_reserve"] = body.cash_reserve
    session["current_step"] = 4
    _save(session)
    return {"status": "ok", "current_step": 4}


@router.post("/advisor/step4/predict")
def step4_predict():
    """SSE endpoint: AI return prediction."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM not configured")

    session = _get_session()
    inputs = session.get("inputs", {})
    total_capital = inputs.get("total_capital", 500000)
    allocation = session.get("allocation", [])
    cycle_data = load_cycle_analysis()

    llm_msgs = build_prediction_messages(total_capital, allocation, cycle_data)

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_msgs):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                parsed = _extract_json(full_text)
                if parsed and "predictions" in parsed:
                    session["prediction"] = parsed

                session["current_step"] = 5
                _save(session)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/advisor/step5/risk")
def step5_risk():
    """SSE endpoint: AI risk monitoring plan."""
    llm_config = get_llm_config()
    if llm_config is None:
        raise HTTPException(status_code=400, detail="LLM not configured")

    session = _get_session()
    allocation = session.get("allocation", [])
    cycle_data = load_cycle_analysis()

    llm_msgs = build_risk_plan_messages(allocation, cycle_data)

    def event_stream():
        full_text = ""
        for event in stream_sse(llm_config, llm_msgs):
            full_text += event.get("content", "")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if event.get("type") == "done":
                parsed = _extract_json(full_text)
                if parsed:
                    session["risk_plan"] = parsed
                _save(session)

            if event.get("type") == "error":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/advisor/step5/save")
def step5_save(body: Step5SaveRequest):
    """Save plan and optionally sync portfolio."""
    session = _get_session()

    if body.sync_portfolio:
        _sync_portfolio(session)

    _save(session)
    return {"status": "ok", "synced_portfolio": body.sync_portfolio}


@router.get("/advisor/step5/download")
def step5_download():
    """Return session JSON as file download."""
    session = load_advisor_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No advisor session found")

    content = json.dumps(session, ensure_ascii=False, indent=2)
    filename = f"advisor_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ==================== Helpers ====================


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    import re
    m = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Balanced braces fallback
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _sync_portfolio(session: dict) -> None:
    """Sync final allocation to portfolio_state.json."""
    allocation = session.get("allocation", [])
    if not allocation:
        return

    portfolio = load_portfolio_state()
    for a in allocation:
        code = a.get("code", "")
        if not code:
            continue
        action = a.get("action", "")
        if action in ("buy", "add"):
            existing = portfolio.get(code, {})
            old_shares = existing.get("shares", 0)
            old_cost = existing.get("avg_cost", 0)
            new_shares = a.get("shares", 0)
            try:
                new_price = float(str(a.get("price_range", "0")).split("-")[0])
            except (ValueError, TypeError):
                new_price = 0
            total_shares = old_shares + new_shares
            if total_shares > 0 and new_price > 0:
                avg_cost = (old_cost * old_shares + new_price * new_shares) / total_shares
            else:
                avg_cost = old_cost or new_price
            portfolio[code] = {
                "avg_cost": avg_cost,
                "shares": total_shares,
                "weight": a.get("ratio", 0),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        elif action == "sell":
            portfolio.pop(code, None)
        elif action == "reduce":
            if code in portfolio:
                portfolio[code]["shares"] = max(
                    0, portfolio[code].get("shares", 0) - abs(a.get("shares", 0))
                )
                portfolio[code]["weight"] = a.get("ratio", 0)
                portfolio[code]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        elif action == "hold":
            if code in portfolio:
                portfolio[code]["weight"] = a.get("ratio", 0)
                portfolio[code]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    save_portfolio_state(portfolio)
