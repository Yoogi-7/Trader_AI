# core/signals.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class TripleBarrierConfig:
    horizon_bars: int = 60
    use_atr: bool = True
    atr_period: int = 14
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    percent_mode: bool = False  # jeśli True, tp/sl liczone % od ceny
    side: str = "long"          # "long" (obsługiwane w tej wersji)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = tr.rolling(period, min_periods=period).mean()
    return atr

def triple_barrier_labels(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    """
    Zwraca DataFrame z kolumną 'label' w {1,0,-1}:
      1 – TP trafiony przed SL w horyzoncie,
      0 – SL trafiony wcześniej albo nic nie trafione do końca,
     -1 – brak danych (za mało świec/horyzontu).
    """
    n = len(df)
    label = np.full(n, -1, dtype=int)
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    if cfg.use_atr:
        atr = _atr(df, cfg.atr_period)
        # deprecation fix:
        atr = atr.bfill().ffill()
        atr = atr.to_numpy()
        tp = close + cfg.tp_mult * atr
        sl = close - cfg.sl_mult * atr
    else:
        if cfg.percent_mode:
            tp = close * (1.0 + cfg.tp_mult)
            sl = close * (1.0 - cfg.sl_mult)
        else:
            tp = close + cfg.tp_mult
            sl = close - cfg.sl_mult

    H = int(cfg.horizon_bars)
    for i in range(n):
        j_end = i + H
        if j_end >= n:
            label[i] = -1
            continue
        # sprawdź przebicia w (i+1 .. j_end) w kolejności czasowej
        hit_tp = False
        hit_sl = False
        for j in range(i + 1, j_end + 1):
            if high[j] >= tp[i]:
                hit_tp = True
                break
            if low[j] <= sl[i]:
                hit_sl = True
                break
        if cfg.side == "long":
            if hit_tp and not hit_sl:
                label[i] = 1
            elif hit_sl and not hit_tp:
                label[i] = 0
            else:
                # nic nie trafione → traktuj jak 0 (konserwatywnie)
                label[i] = 0
        else:
            # na razie tylko long
            label[i] = -1

    return pd.DataFrame({"label": label})
