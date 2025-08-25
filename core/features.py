from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
from core.indicators import ema, atr
from core.fibo import find_pivots, last_impulse_from_pivots, fib_retracements_and_extensions
from core.avwap import anchored_vwap

def _pct(a, b):
    return (b - a) / a * 100.0 if a else 0.0

def _safe(val, default=0.0):
    try:
        x = float(val)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except:
        return default

def compute_features(df: pd.DataFrame, direction: str, entry: float, sl: float, tp1: float, cfg: dict) -> Dict[str, float]:
    """
    Wylicza wektor cech na podstawie ostatnich N świec.
    Zakładamy df z kolumnami: ts_ms, open, high, low, close, volume (rosnąco po czasie).
    """
    if df.empty or len(df) < 60:
        return {}

    dfl = df.tail(max(cfg["models"]["meta"]["lookback_candles"], cfg["signals"]["lookback_candles"])).copy()
    dfl["ema20"] = ema(dfl["close"], 20)
    dfl["ema50"] = ema(dfl["close"], 50)
    dfl["ema100"] = ema(dfl["close"], 100)
    dfl["atr"] = atr(dfl, period=cfg["signals"]["atr_period"])

    close = float(dfl["close"].iloc[-1]); open_ = float(dfl["open"].iloc[-1])
    high = float(dfl["high"].iloc[-1]); low = float(dfl["low"].iloc[-1])
    atr_last = _safe(dfl["atr"].iloc[-1], 1e-9)

    # Proste cechy price/vol
    feats = {
        "ret_1": _pct(float(dfl["close"].iloc[-2]), close),
        "ret_3": _pct(float(dfl["close"].iloc[-4]), close) if len(dfl) >= 5 else 0.0,
        "ret_5": _pct(float(dfl["close"].iloc[-6]), close) if len(dfl) >= 7 else 0.0,
        "ret_10": _pct(float(dfl["close"].iloc[-11]), close) if len(dfl) >= 12 else 0.0,
        "atr_pct": _safe(atr_last / max(close, 1e-9) * 100.0),
        "body_pct": _pct(open_, close),
        "upper_wick_atr": _safe((high - max(open_, close)) / atr_last),
        "lower_wick_atr": _safe((min(open_, close) - low) / atr_last),
        "vol_z20": 0.0,
    }

    vol = dfl["volume"].tail(20)
    if len(vol) >= 5 and vol.std(ddof=1) > 0:
        feats["vol_z20"] = float((vol.iloc[-1] - vol.mean()) / vol.std(ddof=1))

    # Trend/regime
    ema20 = float(dfl["ema20"].iloc[-1]); ema50 = float(dfl["ema50"].iloc[-1])
    ema20_prev = float(dfl["ema20"].iloc[-5]) if len(dfl) >= 25 else ema20
    ema50_prev = float(dfl["ema50"].iloc[-5]) if len(dfl) >= 55 else ema50

    feats.update({
        "ema20_slope_pct5": _pct(ema20_prev, ema20),
        "ema50_slope_pct5": _pct(ema50_prev, ema50),
        "trend_strength": _safe(abs(ema20 - ema50) / atr_last),
        "ema20_above_50": 1.0 if ema20 > ema50 else 0.0,
    })

    # RR i dystanse
    if direction == "long":
        rr = _safe(((tp1 - entry) / entry) / (max((entry - sl) / entry, 1e-9)))
        dist_entry_close = _pct(close, entry)  # >0 gdy entry wyżej niż close
    else:
        rr = _safe(((entry - tp1) / entry) / (max((sl - entry) / entry, 1e-9)))
        dist_entry_close = _pct(entry, close)  # >0 gdy entry niżej niż close

    feats.update({
        "rr_tp1_sl": rr,
        "dist_entry_close_pct": dist_entry_close,
        "dist_sl_entry_pct": _pct(sl, entry) if direction == "long" else _pct(entry, sl),
        "dist_tp1_entry_pct": _pct(entry, tp1) if direction == "long" else _pct(tp1, entry),
    })

    # Fibo + AVWAP (od ostatniego impulsu)
    pivots = find_pivots(dfl, window=cfg["signals"]["fibo"]["piv_window"])
    imp = last_impulse_from_pivots(pivots)
    if imp:
        (p0_idx, p0_price, _), (p1_idx, p1_price, _) = imp
        fibs = fib_retracements_and_extensions(p0_price, p1_price)
        retr = fibs["retr"]
        level_0618 = retr["0.618"]; level_0786 = retr["0.786"]
        avwap_series = anchored_vwap(dfl, p0_idx)
        avwap_now = float(avwap_series.iloc[-1]) if pd.notna(avwap_series.iloc[-1]) else close

        feats.update({
            "dist_0618_close_pct": _pct(close, level_0618),
            "dist_0786_close_pct": _pct(close, level_0786),
            "dist_avwap_close_pct": _pct(close, avwap_now),
        })
    else:
        feats.update({
            "dist_0618_close_pct": 0.0,
            "dist_0786_close_pct": 0.0,
            "dist_avwap_close_pct": 0.0,
        })

    # Kierunek jako feature (pomaga modelowi)
    feats["dir_long"] = 1.0 if direction == "long" else 0.0
    return {k: float(_safe(v)) for k, v in feats.items()}
