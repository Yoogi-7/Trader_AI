"""
FastAPI server exposing /scan endpoint for TRADER_AI MVP.
- POST /scan accepts symbols, risk_profile, equity, run_ingest, and tfs
- Returns structured list of signals from the shared pipeline
All user-facing texts and comments are in English.
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.pipeline.scan import scan_symbols

app = FastAPI(title="TRADER_AI API", version="0.1.0")

class ScanRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default_factory=lambda: ["BTCUSDT"])
    risk_profile: str = Field(default="medium", pattern="^(low|medium|high)$")
    equity: float = 5000.0
    run_ingest: bool = True
    tfs: Optional[List[str]] = Field(default_factory=lambda: ["10m", "15m", "30m"])

class ScanResponse(BaseModel):
    signals: List[dict]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    signals = scan_symbols(
        symbols=req.symbols,
        equity=req.equity,
        risk_profile=req.risk_profile,
        signal_tfs=req.tfs,
        run_ingest=req.run_ingest,
    )
    return {"signals": signals}
