from __future__ import annotations
import pandas as pd
import numpy as np

def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    """
    Liczy AVWAP od zadanego indeksu do końca.
    df: columns ["high","low","close","volume"]
    Zwraca serię AVWAP tej samej długości (NaN przed anchor_idx).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    price_vol = tp * vol
    cs_price_vol = price_vol.copy()
    cs_vol = vol.copy()
    cs_price_vol.iloc[:anchor_idx] = 0.0
    cs_vol.iloc[:anchor_idx] = 0.0
    cs_price_vol = cs_price_vol.cumsum()
    cs_vol = cs_vol.cumsum()
    avwap = cs_price_vol / cs_vol.replace(0, np.nan)
    avwap.iloc[:anchor_idx] = np.nan
    return avwap
