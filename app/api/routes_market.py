"""API for coverage reporting and resume backfill (GET+POST) — robust and defensive."""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.storage.market import get_coverage, resume_backfill

router = APIRouter(tags=["market"])

@router.get("/market/coverage")
def market_coverage(
    symbol: Optional[str] = Query(None),
    timeframe: Optional[str] = Query(None),
) -> List[Dict[str, Any]] | Dict[str, Any]:
    try:
        return get_coverage(symbol=symbol, timeframe=timeframe)
    except Exception as e:
        # Never 500 to the dashboard; return a helpful error payload.
        return JSONResponse({"error": f"coverage failed: {e}"}, status_code=200)

def _resume_impl(symbols: str, tfs: str, days: int, limit: int, sleep_ms: int) -> Dict[str, Any]:
    syms = [s.strip() for s in symbols.split(',') if s.strip()]
    frames = [s.strip() for s in tfs.split(',') if s.strip()]
    return resume_backfill(symbols=syms, tfs=frames, days=days, limit=limit, sleep_ms=sleep_ms)

@router.post("/ingest/binance/resume")
def ingest_binance_resume_post(
    symbols: str = Query("BTCUSDT,ETHUSDT"),
    tfs: str = Query("1m,10m,15m,30m,1h,2h,4h"),
    days: int = Query(730, ge=1, le=3650),
    limit: int = Query(1000, ge=100, le=1500),
    sleep_ms: int = Query(150, ge=0, le=1000),
) -> Dict[str, Any]:
    return _resume_impl(symbols, tfs, days, limit, sleep_ms)

@router.get("/ingest/binance/resume")
def ingest_binance_resume_get(
    symbols: str = Query("BTCUSDT,ETHUSDT"),
    tfs: str = Query("1m,10m,15m,30m,1h,2h,4h"),
    days: int = Query(730, ge=1, le=3650),
    limit: int = Query(1000, ge=100, le=1500),
    sleep_ms: int = Query(150, ge=0, le=1000),
) -> Dict[str, Any]:
    return _resume_impl(symbols, tfs, days, limit, sleep_ms)
