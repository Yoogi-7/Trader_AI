# core/labeling.py
"""
Triple-barrier labeling + pomocnicze narzędzia.
Założenia:
- Dane z kolumnami: ["timestamp", "open", "high", "low", "close", "volume"].
- Czas w "timestamp" w strefie UTC lub lokalnej — nieistotne dla labeling.
- Możesz wybrać progi TP/SL jako wielokrotność ATR lub procent ceny.

Zwraca:
- DataFrame z kolumnami:
  ['tp_price','sl_price','horizon_end','label','outcome','time_to_outcome']
gdzie:
  label: 1 = TP, 0 = SL, -1 = horyzont (brak TP/SL w czasie),
  outcome: 'tp' / 'sl' / 'horizon'
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class TripleBarrierConfig:
    horizon_bars: int = 60               # ile świec w horyzoncie
    use_atr: bool = True                 # czy progi liczyć z ATR
    atr_period: int = 14                 # ATR do progów
    tp_mult: float = 2.0                 # mnożnik ATR lub procent (jeśli use_atr=False, to tp_mult=0.01 -> 1%)
    sl_mult: float = 1.0                 # j.w.
    percent_mode: bool = False           # jeśli True, tp/sl jako % (0.01 = 1%)
    side: str = "long"                   # 'long' lub 'short' — dla klasyfikacji jednoklasowej

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def triple_barrier_labels(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    df = df.copy()
    if cfg.use_atr:
        atr = _atr(df, cfg.atr_period)
        if cfg.side == "long":
            tp_dist = cfg.tp_mult * atr
            sl_dist = cfg.sl_mult * atr
        else:
            tp_dist = cfg.tp_mult * atr  # symetrycznie; niżej odwracamy logikę
            sl_dist = cfg.sl_mult * atr
    else:
        # procent od close
        if cfg.side == "long":
            tp_dist = df["close"] * cfg.tp_mult if cfg.percent_mode else cfg.tp_mult
            sl_dist = df["close"] * cfg.sl_mult if cfg.percent_mode else cfg.sl_mult
        else:
            tp_dist = df["close"] * cfg.tp_mult if cfg.percent_mode else cfg.tp_mult
            sl_dist = df["close"] * cfg.sl_mult if cfg.percent_mode else cfg.sl_mult

    # Wyznaczenie barier cenowych (na podstawie close jako ceny referencyjnej)
    if cfg.side == "long":
        df["tp_price"] = df["close"] + tp_dist
        df["sl_price"] = df["close"] - sl_dist
    else:  # short
        df["tp_price"] = df["close"] - tp_dist
        df["sl_price"] = df["close"] + sl_dist

    n = len(df)
    horizon_idx = np.arange(n) + cfg.horizon_bars
    horizon_idx[horizon_idx >= n] = n - 1
    df["horizon_end_idx"] = horizon_idx
    df["horizon_end"] = df["timestamp"].values[df["horizon_end_idx"]]

    # Dla każdego punktu startowego sprawdzamy w kolejnych świecach, co padnie najpierw.
    labels = np.full(n, -1, dtype=int)      # -1 = horizon
    outcomes = np.array(["horizon"] * n, dtype=object)
    time_to_outcome = np.full(n, np.nan, dtype=float)

    highs = df["high"].values
    lows = df["low"].values
    tp_prices = df["tp_price"].values
    sl_prices = df["sl_price"].values

    for i in range(n - cfg.horizon_bars):
        j_end = df["horizon_end_idx"].iat[i]
        # Start - kolejna świeca (latency bar zostawiamy do modułu execution; tu etykieta)
        # Labeling klasycznie patrzy od i+1 do j_end
        start = i + 1
        tp_hit = sl_hit = None
        for j in range(start, j_end + 1):
            # Priorytet: SL -> TP (konserwatywnie)
            if lows[j] <= sl_prices[i]:
                sl_hit = j
                break
            if highs[j] >= tp_prices[i]:
                tp_hit = j
                break
        if tp_hit is not None:
            labels[i] = 1
            outcomes[i] = "tp"
            time_to_outcome[i] = tp_hit - i
        elif sl_hit is not None:
            labels[i] = 0
            outcomes[i] = "sl"
            time_to_outcome[i] = sl_hit - i
        else:
            labels[i] = -1
            outcomes[i] = "horizon"
            time_to_outcome[i] = j_end - i

    out = df[["timestamp", "tp_price", "sl_price", "horizon_end"]].copy()
    out["label"] = labels
    out["outcome"] = outcomes
    out["time_to_outcome"] = time_to_outcome
    return out
