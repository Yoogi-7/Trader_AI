from __future__ import annotations
import pandas as pd
import numpy as np

def _ema(s: pd.Series, span: int) -> pd.Series:
    span = max(2, int(span))
    return s.ewm(span=span, adjust=False, min_periods=1).mean()

def _rsi(close: pd.Series, length: int) -> pd.Series:
    length = max(2, int(length))
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    length = max(2, int(length))
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=1).mean()

def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    n = len(close)
    fast = max(2, min(fast, max(2, n // 6)))
    slow = max(fast + 1, min(slow, max(3, n // 3)))
    signal = max(2, min(signal, max(2, n // 8)))
    macd_line = _ema(close, fast) - _ema(close, slow)
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca DF z kolumnami:
      ema21, ema50, ema200, rsi, macd, macd_signal, macd_hist, atr, vol_ma, vol_std, vol_z
    Odporne na krótkie serie: dynamiczne okna + brak globalnego dropna().
    """
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    n = len(out)
    len21  = min(21,  max(2, n // 8 or 2))
    len50  = min(50,  max(2, n // 5 or 2))
    len200 = min(200, max(2, n // 2 or 2))
    rsi_len = min(14, max(2, n // 10 or 2))
    atr_len = min(14, max(2, n // 10 or 2))

    out['ema21']  = _ema(out['close'], len21)
    out['ema50']  = _ema(out['close'], len50)
    out['ema200'] = _ema(out['close'], len200)

    out['rsi'] = _rsi(out['close'], rsi_len)

    macd_line, macd_sig, macd_hist = _macd(out['close'], fast=12, slow=26, signal=9)
    out['macd'] = macd_line
    out['macd_signal'] = macd_sig
    out['macd_hist'] = macd_hist

    out['atr'] = _atr(out['high'], out['low'], out['close'], atr_len)

    win = min(50, max(2, n))
    out['vol_ma']  = out['volume'].rolling(window=win, min_periods=1).mean()
    out['vol_std'] = out['volume'].rolling(window=win, min_periods=1).std(ddof=0).replace(0, np.nan)
    out['vol_z']   = (out['volume'] - out['vol_ma']) / (out['vol_std'].replace(0, np.nan))

    # ffill/bfill zamiast deprecated fillna(method=...)
    out = out.ffill().bfill()

    return out
