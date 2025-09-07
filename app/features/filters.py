"""Trading filters and feature helpers to boost signal quality."""
from __future__ import annotations

import math
import pandas as pd

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    # Lightweight DI/ADX approximation (good enough for filtering regimes)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series((up > down) & (up > 0), index=df.index) * up.clip(lower=0)
    minus_dm = pd.Series((down > up) & (down > 0), index=df.index) * down.clip(lower=0)
    tr = atr(df, length=1)
    plus_di = 100 * (plus_dm.ewm(span=length, adjust=False).mean() / tr.replace(0, 1e-9))
    minus_di = 100 * (minus_dm.ewm(span=length, adjust=False).mean() / tr.replace(0, 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)) * 100
    return dx.ewm(span=length, adjust=False).mean()

def trend_filter(df: pd.DataFrame, ema_len: int = 200) -> pd.Series:
    e = ema(df["close"], ema_len)
    return (df["close"] > e).map(lambda x: "up" if x else "down")

def volatility_filter(df: pd.DataFrame, atr_len: int = 14, min_atr_pct: float = 0.002) -> pd.Series:
    """Require ATR% >= threshold to avoid dead regimes."""
    a = atr(df, atr_len)
    atr_pct = a / df["close"]
    return (atr_pct >= min_atr_pct)
