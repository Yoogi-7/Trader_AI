# core/features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(period, min_periods=period).mean()

def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std()
    upper = ma + mult * sd
    lower = ma - mult * sd
    return lower, ma, upper

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy cechy na bazie OHLCV. Nie wprowadza wycieku – wszystkie cechy są zshiftowane o 1 bar.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    out["rsi_14"] = rsi(close, 14)
    out["atr_14"] = atr(df, 14)

    lb, ma, ub = bollinger(close, 20, 2.0)
    out["bb_pos"] = (close - ma) / (ub - lb + 1e-12)

    out["vol_z"] = (volume - volume.rolling(20, min_periods=20).mean()) / (volume.rolling(20, min_periods=20).std() + 1e-12)
    out["rng"] = (high - low) / (close + 1e-12)

    sma_fast = close.rolling(20, min_periods=20).mean()
    sma_slow = close.rolling(60, min_periods=60).mean()
    out["sma_ratio"] = (sma_fast - sma_slow) / (sma_slow.abs() + 1e-12)

    # przesunięcie o 1 bar (cechy znane w momencie decyzji)
    out = out.shift(1)

    # usuwamy NaN-y z początku
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
