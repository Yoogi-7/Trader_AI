"""
Higher-timeframe trend filter:
- Trend is 'up' if close > EMA200 and EMA200 slope > 0
- Trend is 'down' if close < EMA200 and EMA200 slope < 0
- Otherwise 'flat'
"""

import pandas as pd
from typing import Literal

Trend = Literal["up", "down", "flat"]

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def trend_status(df: pd.DataFrame, ema_len: int = 200) -> Trend:
    """
    df: DataFrame with at least 'close' column, indexed by timestamp.
    """
    if len(df) < ema_len + 2:
        return "flat"
    e = ema(df["close"], ema_len)
    # slope proxy: last EMA vs previous EMA
    slope_up = e.iloc[-1] > e.iloc[-2]
    slope_down = e.iloc[-1] < e.iloc[-2]
    price_above = df["close"].iloc[-1] > e.iloc[-1]
    price_below = df["close"].iloc[-1] < e.iloc[-1]

    if price_above and slope_up:
        return "up"
    if price_below and slope_down:
        return "down"
    return "flat"
