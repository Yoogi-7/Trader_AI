# core/features.py
from __future__ import annotations
import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = -diff.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    k_fast = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k_slow = k_fast.rolling(d, min_periods=d).mean()
    d_slow = k_slow.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"stoch_k": k_slow, "stoch_d": d_slow})


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_sig = _ema(macd, signal)
    macd_hist = macd - macd_sig
    return pd.DataFrame({"macd": macd, "macd_signal": macd_sig, "macd_hist": macd_hist})


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bollinger(close: pd.Series, period: int = 20, nstd: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std()
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    width = (upper - lower) / (ma.replace(0, np.nan)).abs()
    dist_mid = (close - ma) / (sd.replace(0, np.nan))
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_z": dist_mid})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buduje cechy dla modelu na podstawie OHLCV z kolumn:
    ['timestamp','open','high','low','close','volume'].
    Zwraca DataFrame z indeksem równym df.index (bez kolumny timestamp).
    """
    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    vol = data["volume"].astype(float)

    # Zwroty
    ret_1 = close.pct_change()
    logret_1 = np.log(close.replace(0, np.nan)).diff()

    # Z-score zwrotów
    def _z(x: pd.Series, win: int) -> pd.Series:
        m = x.rolling(win, min_periods=win).mean()
        s = x.rolling(win, min_periods=win).std()
        return (x - m) / s.replace(0, np.nan)

    zret_20 = _z(logret_1, 20)
    zret_50 = _z(logret_1, 50)

    # Średnie kroczące i ich relacje
    sma_10 = close.rolling(10, min_periods=10).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    sma_50 = close.rolling(50, min_periods=50).mean()
    ema_20 = _ema(close, 20)
    ema_50 = _ema(close, 50)

    sma_ratio_20 = close / sma_20.replace(0, np.nan)
    sma_ratio_50 = close / sma_50.replace(0, np.nan)
    ema_spread = (ema_20 - ema_50) / ema_50.replace(0, np.nan)

    # Zmienność / ATR
    atr14 = _atr(high, low, close, 14)
    atr_pct = atr14 / close.replace(0, np.nan)

    # RSI / MACD / Stochastic
    rsi14 = _rsi(close, 14)
    macd_df = _macd(close, 12, 26, 9)
    stoch_df = _stoch(high, low, close, 14, 3)

    # Bollinger
    bb = _bollinger(close, 20, 2.0)

    # Wolumen
    vol_ema_20 = _ema(vol, 20)
    vol_z20 = _z(vol, 20)

    feats = pd.DataFrame({
        # zwroty
        "ret_1": ret_1,
        "logret_1": logret_1,
        "zret_20": zret_20,
        "zret_50": zret_50,
        # MAs
        "sma_ratio_20": sma_ratio_20,
        "sma_ratio_50": sma_ratio_50,
        "ema_spread": ema_spread,
        # ATR / zmienność
        "atr_pct_14": atr_pct,
        # RSI / MACD / Stoch
        "rsi14": rsi14,
        "macd": macd_df["macd"],
        "macd_signal": macd_df["macd_signal"],
        "macd_hist": macd_df["macd_hist"],
        "stoch_k": stoch_df["stoch_k"],
        "stoch_d": stoch_df["stoch_d"],
        # Bollinger
        "bb_width": bb["bb_width"],
        "bb_z": bb["bb_z"],
        # Vol
        "vol_ema_20": vol_ema_20,
        "vol_z20": vol_z20,
    }, index=df.index)

    # Sprzątanie
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats
