"""
Lightweight setup detectors:
1) Breakout of 50-bar channel (configurable lookback),
2) Fallback momentum: recent contraction (low true range) and MA(20) cross.
These are intentionally simple to get the pipeline producing candidates.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

def _atr_proxy(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return (df["high"] - df["low"]).rolling(n).mean()

def detect_breakout_retest(
    df: pd.DataFrame,
    lookback: int = 50
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if len(df) < lookback + 2:
        return out

    hi = df["high"].rolling(lookback).max()
    lo = df["low"].rolling(lookback).min()
    atr = _atr_proxy(df, 14)

    for i in range(lookback + 1, len(df)):
        ts = df.index[i - 1]
        close_prev = df["close"].iloc[i - 1]
        hi_prev = hi.iloc[i - 2]
        lo_prev = lo.iloc[i - 2]
        atr_v = float(atr.iloc[i - 1])

        if pd.isna(hi_prev) or pd.isna(lo_prev):
            continue

        if close_prev > hi_prev:
            out.append({"i": ts, "side": "LONG", "swing_low": float(lo_prev), "swing_high": float(hi_prev), "atr": atr_v})
        elif close_prev < lo_prev:
            out.append({"i": ts, "side": "SHORT", "swing_low": float(lo_prev), "swing_high": float(hi_prev), "atr": atr_v})
    return out

def detect_fallback_momentum(
    df: pd.DataFrame,
    ma_len: int = 20,
    squeeze_len: int = 12,
    squeeze_threshold: float = 0.002
) -> List[Dict[str, Any]]:
    """
    Very simple momentum fallback:
    - Detects recent contraction: mean(high-low) over 'squeeze_len' < 'squeeze_threshold' (fraction of price).
    - If last close crosses MA(ma_len) upward -> LONG candidate, downward -> SHORT candidate.
    Swing bounds are approximated with recent min/max in the window.
    """
    out: List[Dict[str, Any]] = []
    if len(df) < max(ma_len, squeeze_len) + 2:
        return out

    tr = (df["high"] - df["low"]).rolling(squeeze_len).mean()
    price = df["close"]
    ma = price.ewm(span=ma_len, adjust=False).mean()
    window = df.iloc[-squeeze_len:]
    price_mid = price.iloc[-1]
    squeeze_ok = (tr.iloc[-1] / price_mid) < squeeze_threshold if price_mid else False

    # Cross detection on the last bar close
    crossed_up = price.iloc[-2] <= ma.iloc[-2] and price.iloc[-1] > ma.iloc[-1]
    crossed_down = price.iloc[-2] >= ma.iloc[-2] and price.iloc[-1] < ma.iloc[-1]

    swing_low = float(window["low"].min())
    swing_high = float(window["high"].max())
    atr_v = float(_atr_proxy(df, 14).iloc[-1])

    ts = df.index[-1]
    if squeeze_ok and crossed_up:
        out.append({"i": ts, "side": "LONG", "swing_low": swing_low, "swing_high": swing_high, "atr": atr_v})
    elif squeeze_ok and crossed_down:
        out.append({"i": ts, "side": "SHORT", "swing_low": swing_low, "swing_high": swing_high, "atr": atr_v})
    return out

def detect_candidates(
    df: pd.DataFrame,
    lookback: int = 40,
    use_fallback: bool = True,
    fallback_ma_len: int = 20,
    fallback_squeeze_len: int = 12,
    fallback_squeeze_threshold: float = 0.002
) -> List[Dict[str, Any]]:
    """
    Combined detector that first tries breakout, then (optionally) a momentum fallback.
    """
    out = detect_breakout_retest(df, lookback=lookback)
    if not out and use_fallback:
        out = detect_fallback_momentum(
            df,
            ma_len=fallback_ma_len,
            squeeze_len=fallback_squeeze_len,
            squeeze_threshold=fallback_squeeze_threshold,
        )
    return out
