"""
Basic level proposal using swing range + Fibonacci and ATR buffer.
This is intentionally simple for the MVP; we'll refine later.
"""
from typing import Tuple

def propose_levels(side: str,
                   swing_low: float,
                   swing_high: float,
                   atr: float,
                   k_atr_sl: float = 0.7) -> Tuple[float, float, float, float]:
    if side == "LONG":
        entry = swing_low + 0.618 * (swing_high - swing_low)
        sl = swing_low - k_atr_sl * atr
        tp1 = swing_high
        tp2 = swing_high + 0.272 * (swing_high - swing_low)
    else:
        entry = swing_high - 0.618 * (swing_high - swing_low)
        sl = swing_high + k_atr_sl * atr
        tp1 = swing_low
        tp2 = swing_low - 0.272 * (swing_high - swing_low)
    return float(entry), float(sl), float(tp1), float(tp2)
