"""Symbol utilities shared across the app."""
from __future__ import annotations

from typing import Iterable, List

from app.data.exchange import normalize_symbol

DEFAULT_SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "TONUSDT", "LINKUSDT", "AVAXUSDT",
]

def normalize_many(symbols: Iterable[str]) -> List[str]:
    return [normalize_symbol(s) for s in symbols]
