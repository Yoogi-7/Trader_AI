"""
Timeframe utilities and mapping for resampling.
"""
from typing import Tuple

TF_TO_PANDAS = {"10m": "10T", "15m": "15T", "30m": "30T", "1h": "1H", "2h": "2H", "4h": "4H"}
TF_ORDER = ["10m", "15m", "30m", "1h", "2h", "4h"]

def next_two_higher_tfs(tf: str) -> Tuple[str, str]:
    i = TF_ORDER.index(tf)
    h1 = TF_ORDER[min(i + 1, len(TF_ORDER) - 1)]
    h2 = TF_ORDER[min(i + 2, len(TF_ORDER) - 1)]
    return h1, h2
