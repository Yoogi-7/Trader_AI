"""FastAPI router for scanning and persisting signals using Binance data."""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Query
from app.pipeline.scan import scan_symbols

router = APIRouter(tags=["scan"])

@router.post("/scan")
def scan_endpoint(
    symbols: Optional[str] = Query("BTCUSDT,ETHUSDT", description="Comma-separated list"),
    tfs: Optional[str] = Query("10m,15m,30m,1h,2h,4h", description="Comma-separated TFs"),
    days: int = Query(120, ge=1, le=365*3),
    persist: bool = Query(True, description="Persist signals to DB"),
    save_ohlcv: bool = Query(True, description="Persist OHLCV to DB"),
    fee_bps: float = Query(6.0, ge=0.0, le=100.0),
    slippage_bps: float = Query(2.0, ge=0.0, le=100.0),
    k_sl_atr: float = Query(1.5, ge=0.1, le=10.0),
    k_tp_atr: float = Query(2.5, ge=0.1, le=20.0),
) -> Dict[str, Any]:
    syms: List[str] = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    frames: List[str] = [s.strip() for s in (tfs or "").split(",") if s.strip()]
    rows = scan_symbols(
        symbols=syms,
        tfs=frames,
        days=days,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        k_sl_atr=k_sl_atr,
        k_tp_atr=k_tp_atr,
        persist_signals=persist,
        persist_ohlcv=save_ohlcv,
    )
    return {"count": len(rows), "signals": rows[-50:]}  # return last 50 for brevity
