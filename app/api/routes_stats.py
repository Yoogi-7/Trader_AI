"""FastAPI router for rolling stats endpoints."""
from fastapi import APIRouter, Query
from typing import Optional

from app.storage.db import rolling_stats

router = APIRouter(tags=["stats"])


@router.get("/stats/rolling")
def get_rolling_stats(
    days: int = Query(30, ge=1, le=365),
    symbol: Optional[str] = Query(None, description="Filter by symbol, e.g. BTCUSDT"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe, e.g. 1h"),
):
    """Return per-day aggregates for charts.

    Response format:
    [
      {"day": "2025-09-01", "cnt": 12, "avg_p_hit": 0.61, "avg_rr": 1.8, "avg_ev": 0.12},
      ...
    ]
    """
    return rolling_stats(days=days, symbol=symbol, timeframe=timeframe)
