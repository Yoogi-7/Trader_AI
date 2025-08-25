from __future__ import annotations
import pandas as pd
import numpy as np

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = pd.Series(tr).rolling(period, min_periods=period).mean()
    return atr.reindex(df.index).astype(float)

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values

    plus_di = 100 * (pd.Series(plus_dm).rolling(period, min_periods=period).sum().values / (atr + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period, min_periods=period).sum().values / (atr + 1e-12))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
    adx = pd.Series(dx).rolling(period, min_periods=period).mean()
    adx = pd.Series([np.nan] + adx.tolist(), index=df.index)
    return adx

def compute_regime(df: pd.DataFrame) -> dict:
    """
    Zwraca:
      {
        "trend": "trend"|"range",
        "vol":   "low"|"normal"|"high",
        "adx": float, "atr_pct": float
      }
    """
    if df is None or len(df) < 100:
        return {"trend":"range","vol":"normal","adx":0.0,"atr_pct":0.0}

    d = df.tail(300).copy()
    d.index = pd.to_datetime(d["ts_ms"], unit="ms", utc=True)
    adx = _adx(d, period=14)
    atr = _atr(d, period=14)
    close = d["close"]

    adx_last = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    atr_pct = float((atr.iloc[-1] / max(close.iloc[-1], 1e-9)) * 100.0) if not pd.isna(atr.iloc[-1]) else 0.0

    # percentyle ATR% z okna
    atr_pct_series = (atr / close * 100.0).dropna()
    if len(atr_pct_series) >= 30:
        p33 = float(np.percentile(atr_pct_series, 33))
        p67 = float(np.percentile(atr_pct_series, 67))
    else:
        p33, p67 = 0.5, 1.5  # bezpieczne defaulty

    trend = "trend" if adx_last >= 25.0 else "range"
    if atr_pct <= p33:
        vol = "low"
    elif atr_pct >= p67:
        vol = "high"
    else:
        vol = "normal"

    return {"trend":trend, "vol":vol, "adx":adx_last, "atr_pct":atr_pct}
