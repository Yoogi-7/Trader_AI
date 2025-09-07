"""FastAPI route to backfill two years for a list of symbols & timeframes."""
from __future__ import annotations

from typing import List, Dict, Any
from fastapi import APIRouter, Query
from app.scripts.backfill_binance_bulk import main as backfill_main

router = APIRouter(tags=["ingest"])

@router.post("/ingest/binance/bulk")
def ingest_binance_bulk(
    symbols: str = Query("BTCUSDT,ETHUSDT,SOLUSDT"),
    tfs: str = Query("1m,10m,15m,30m,1h,2h,4h"),
    days: int = Query(730, ge=1, le=3650),
    limit: int = Query(1000, ge=100, le=1500),
    sleep_ms: int = Query(150, ge=0, le=1000),
) -> Dict[str, Any]:
    args = [
        "--symbols", symbols,
        "--tfs", tfs,
        "--days", str(days),
        "--limit", str(limit),
        "--sleep_ms", str(sleep_ms),
    ]
    code = backfill_main(args)
    return {"ok": code == 0, "code": code, "params": {"symbols": symbols, "tfs": tfs, "days": days}}
