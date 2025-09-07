"""FastAPI router to ingest Binance OHLCV and persist in SQLite."""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query
from app.data.exchange import backfill_ohlcv_1m, resample, normalize_symbol
from app.storage.market import upsert_ohlcv

router = APIRouter(tags=["ingest"])

@router.post("/ingest/binance")
def ingest_binance(
    symbol: str,
    days: int = Query(90, ge=1, le=365*3),
    timeframe: str = Query("1m", description="Target TF to save; '1m' saves raw, others resampled."),
    persist: bool = True,
) -> Dict[str, Any]:
    """Backfill 1m from Binance, optionally resample, and save to DB (ohlcv)."""
    df_1m = backfill_ohlcv_1m(symbol, days=days)
    if timeframe != "1m":
        df = resample(df_1m, timeframe)
    else:
        df = df_1m
    if persist and not df.empty:
        sym = normalize_symbol(symbol)
        rows = [
            dict(ts=int(r.ts), symbol=sym, timeframe=timeframe,
                 open=float(r.open), high=float(r.high), low=float(r.low),
                 close=float(r.close), volume=float(r.volume))
            for r in df.itertuples(index=False)
        ]
        n = upsert_ohlcv(rows)
    else:
        n = 0
    return {"symbol": normalize_symbol(symbol), "timeframe": timeframe, "rows": int(n), "empty": bool(df.empty)}
