"""
Lightweight setup detectors:
1) Breakout of N-bar channel (configurable lookback) — emits events across history,
2) Fallback momentum (SERIES): contraction "squeeze" + MA(20) cross — emits events across history.

These are intentionally simple to produce candidates both live and for dataset building.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np

def _atr_proxy(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return (df["high"] - df["low"]).rolling(n).mean()

def detect_breakout_retest(
    df: pd.DataFrame,
    lookback: int = 50
) -> List[Dict[str, Any]]:
    """
    Emits historical breakout events: close above rolling max or below rolling min.
    """
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
        atr_v = float(atr.iloc[i - 1]) if pd.notna(atr.iloc[i - 1]) else float((df["high"]-df["low"]).iloc[max(0,i-15):i].mean())

        if pd.isna(hi_prev) or pd.isna(lo_prev):
            continue

        if close_prev > hi_prev:
            out.append({"i": ts, "side": "LONG", "swing_low": float(lo_prev), "swing_high": float(hi_prev), "atr": atr_v})
        elif close_prev < lo_prev:
            out.append({"i": ts, "side": "SHORT", "swing_low": float(lo_prev), "swing_high": float(hi_prev), "atr": atr_v})
    return out

def detect_fallback_momentum_series(
    df: pd.DataFrame,
    ma_len: int = 20,
    squeeze_len: int = 12,
    squeeze_threshold: float = 0.002
) -> List[Dict[str, Any]]:
    """
    Series version of momentum fallback:
    - Squeeze: mean(high-low, squeeze_len)/price < threshold
    - Cross: close crosses EMA(ma_len) up/down.
    Emits events for every index where both conditions are true.
    """
    out: List[Dict[str, Any]] = []
    if len(df) < max(ma_len, squeeze_len) + 2:
        return out

    price = df["close"]
    ma = price.ewm(span=ma_len, adjust=False).mean()
    tr = (df["high"] - df["low"]).rolling(squeeze_len).mean()
    atr = _atr_proxy(df, 14)
    # booleans per-bar
    with np.errstate(divide="ignore", invalid="ignore"):
        squeeze = (tr / price).fillna(np.inf) < squeeze_threshold

    crossed_up = (price.shift(1) <= ma.shift(1)) & (price > ma)
    crossed_dn = (price.shift(1) >= ma.shift(1)) & (price < ma)

    # swings over the last squeeze window
    low_sw = df["low"].rolling(squeeze_len).min()
    high_sw = df["high"].rolling(squeeze_len).max()

    for idx in df.index:
        if not squeeze.loc[idx]:
            continue
        atr_v = float(atr.loc[idx]) if pd.notna(atr.loc[idx]) else float((df["high"]-df["low"]).loc[:idx].tail(14).mean())
        if crossed_up.loc[idx]:
            out.append({
                "i": idx, "side": "LONG",
                "swing_low": float(low_sw.loc[idx]),
                "swing_high": float(high_sw.loc[idx]),
                "atr": atr_v
            })
        elif crossed_dn.loc[idx]:
            out.append({
                "i": idx, "side": "SHORT",
                "swing_low": float(low_sw.loc[idx]),
                "swing_high": float(high_sw.loc[idx]),
                "atr": atr_v
            })
    return out

def detect_candidates(
    df: pd.DataFrame,
    lookback: int = 40,
    use_fallback: bool = True,
    fallback_ma_len: int = 20,
    fallback_squeeze_len: int = 12,
    fallback_squeeze_threshold: float = 0.002,
    series_mode: bool = True,
) -> List[Dict[str, Any]]:
    """
    Combined detector.
    - Breakout: always emits series of events.
    - Fallback: if series_mode=True, emits series; else only last-bar event (live mode).
    """
    out = detect_breakout_retest(df, lookback=lookback)
    if use_fallback:
        if series_mode:
            out += detect_fallback_momentum_series(
                df,
                ma_len=fallback_ma_len,
                squeeze_len=fallback_squeeze_len,
                squeeze_threshold=fallback_squeeze_threshold,
            )
        else:
            # non-series fallback is intentionally omitted in dataset build
            out += []
    # sort by timestamp to be deterministic
    out.sort(key=lambda x: x["i"])
    return out
