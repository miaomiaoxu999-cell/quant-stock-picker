"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import settings, factors, cycle, stocks, audit, advisor, portfolio

app = FastAPI(
    title="Quant Stock Picker API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings.router, prefix="/api")
app.include_router(factors.router, prefix="/api")
app.include_router(cycle.router, prefix="/api")
app.include_router(stocks.router, prefix="/api")
app.include_router(audit.router, prefix="/api")
app.include_router(advisor.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
