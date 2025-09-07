"""FastAPI routes for testing and dispatching alerts."""
from __future__ import annotations

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query
from app.alerts.notify import notify_signal
from app.storage.db import fetch_last_signals

router = APIRouter(tags=["alerts"])


@router.get("/alerts/test")
def alerts_test() -> Dict[str, Any]:
    """Send a sample alert to all configured channels."""
    sample = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "direction": "long",
        "entry": 60000.0,
        "sl": 59000.0,
        "tp": 61500.0,
        "p_hit": 0.72,
        "rr": 1.5,
    }
    return notify_signal(sample)


@router.post("/alerts/dispatch")
def alerts_dispatch(
    min_p_hit: float = Query(0.7, ge=0.0, le=1.0),
    min_rr: float = Query(0.0, ge=0.0),
    limit: int = Query(50, ge=1, le=500),
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch recent signals and send alerts for those above thresholds."""
    rows: List[Dict[str, Any]] = fetch_last_signals(limit=limit, symbol=symbol, timeframe=timeframe)
    sent = 0
    results: List[Dict[str, Any]] = []
    for r in rows:
        p_hit = (r.get("p_hit") or 0.0)
        rr = (r.get("rr") or 0.0)
        if p_hit >= min_p_hit and rr >= min_rr:
            results.append({"signal": r, "result": notify_signal(r)})
            sent += 1
    return {"requested": len(rows), "sent": sent, "details": results}
