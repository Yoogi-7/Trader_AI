"""
FastAPI server for TRADER_AI MVP.
Endpoints:
- GET  /health
- POST /scan         -> run pipeline and (optionally) persist signals
- GET  /signals/last -> fetch recent persisted signals
- GET  /stats        -> basic aggregates from persisted signals
- GET  /export       -> download CSV/Parquet of persisted signals
All user-facing texts and comments are in English.
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.pipeline.scan import scan_symbols, SUPPORTED_TFS
from app.storage.db import insert_signals, fetch_last_signals, basic_stats, SignalDTO
from app.storage.export import export_signals

app = FastAPI(title="TRADER_AI API", version="0.3.0")

SUPPORTED_DEFAULT_SYMBOLS = ["BTCUSDT"]
DEFAULT_TFS = ["10m", "15m", "30m"]


class ScanRequest(BaseModel):
    symbols: Optional[List[str]] = Field(
        default_factory=lambda: SUPPORTED_DEFAULT_SYMBOLS, examples=[SUPPORTED_DEFAULT_SYMBOLS]
    )
    risk_profile: str = Field(default="medium", pattern="^(low|medium|high)$")
    equity: float = 5000.0
    run_ingest: bool = True
    tfs: Optional[List[str]] = Field(default_factory=lambda: DEFAULT_TFS)
    persist: bool = False  # if True, store signals in SQLite

    @field_validator("symbols", mode="before")
    @classmethod
    def normalize_symbols(cls, v):
        # Swagger sometimes injects "string" as a placeholder — normalize it away.
        if v is None:
            return SUPPORTED_DEFAULT_SYMBOLS
        if isinstance(v, str):
            vs = v.strip()
            if not vs or vs.lower() == "string":
                return SUPPORTED_DEFAULT_SYMBOLS
            return [vs]
        out = []
        for s in v:
            if isinstance(s, str):
                ss = s.strip()
                if ss and ss.lower() != "string":
                    out.append(ss)
        return out or SUPPORTED_DEFAULT_SYMBOLS

    @field_validator("tfs", mode="before")
    @classmethod
    def normalize_tfs(cls, v):
        # Clean placeholders and keep only supported TFs
        if v is None:
            return DEFAULT_TFS
        if isinstance(v, str):
            vs = v.strip().lower()
            if not vs or vs == "string":
                return DEFAULT_TFS
            v_list = [vs]
        else:
            v_list = [(str(x)).strip().lower() for x in v if str(x).strip()]

        cleaned = [tf for tf in v_list if tf in SUPPORTED_TFS]
        return cleaned or DEFAULT_TFS


class ScanResponse(BaseModel):
    signals: List[dict]
    persisted: int = 0


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    if not req.symbols:
        raise HTTPException(status_code=400, detail="No symbols provided.")
    if not req.tfs:
        raise HTTPException(status_code=400, detail="No valid timeframes provided.")

    signals = scan_symbols(
        symbols=req.symbols,
        equity=req.equity,
        risk_profile=req.risk_profile,
        signal_tfs=req.tfs,
        run_ingest=req.run_ingest,
    )
    persisted = 0
    if req.persist and signals:
        dto = [
            SignalDTO(
                ts=s["ts"],
                symbol=s["symbol"],
                tf=s["tf"],
                side=s["side"],
                htf1=s["htf1"],
                htf2=s["htf2"],
                entry=float(s["entry"]),
                sl=float(s["sl"]),
                tp1=float(s["tp1"]),
                tp2=float(s["tp2"]),
                rr=float(s["rr"]),
                p_hit=float(s["p_hit"]),
                notional=float(s["notional"]),
                fee=float(s["fee"]),
                slip=float(s["slip"]),
                net_tp=float(s["net_tp"]),
                ev=float(s["ev"]),
                ok=bool(s["ok"]),
            )
            for s in signals
        ]
        persisted = insert_signals(dto)
    return {"signals": signals, "persisted": persisted}


@app.get("/signals/last")
def last_signals(limit: int = Query(50, ge=1, le=500)):
    rows = fetch_last_signals(limit=limit)
    return {"rows": rows, "count": len(rows)}


@app.get("/stats")
def stats():
    return basic_stats()


@app.get("/export")
def export(format: str = Query("csv", pattern="^(csv|parquet)$"), limit: int | None = Query(None, ge=1)):
    path = export_signals(format=format, limit=limit)
    media = "text/csv" if format == "csv" else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)
