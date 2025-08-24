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
    percent_mode: bool = False  # jeśli True, tp/sl to procenty ceny
    side: str = "long"          # na razie wspieramy "long"

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(period, min_periods=period).mean()

def triple_barrier_labels(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    """
    Zwraca DataFrame: ['tp_price','sl_price','horizon_idx','label','event_idx'].
    label: 1=TP, 0=SL, -1=Horizon. Wymaga kolumn: timestamp, open, high, low, close, volume.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["tp_price","sl_price","horizon_idx","label","event_idx"])

    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    horizon = int(cfg.horizon_bars)

    if cfg.percent_mode:
        tp_off = cfg.tp_mult / 100.0 * close  # %
        sl_off = cfg.sl_mult / 100.0 * close
    else:
        if cfg.use_atr:
            atr = _atr(df, cfg.atr_period).fillna(method="bfill").fillna(method="ffill").to_numpy()
        else:
            atr = (df["high"].astype(float) - df["low"].astype(float)).rolling(14, min_periods=1).mean().to_numpy()
        tp_off = cfg.tp_mult * atr
        sl_off = cfg.sl_mult * atr

    tp_price = close + tp_off
    sl_price = close - sl_off

    label = np.full(n, -1, dtype=int)
    event_idx = np.full(n, -1, dtype=int)
    horizon_idx = (np.arange(n) + horizon).clip(max=n-1)

    # prosta pętla (wystarczająco szybka na kilkadziesiąt tysięcy barów)
    for i in range(n):
        j_end = int(horizon_idx[i])
        if i + 1 > j_end:
            continue
        # sprawdzaj czy w oknie trafiło TP/SL; wybierz pierwsze zdarzenie w czasie
        hh = high[i+1:j_end+1]
        ll = low[i+1:j_end+1]
        tp_hit = np.where(hh >= tp_price[i])[0]
        sl_hit = np.where(ll <= sl_price[i])[0]
        if tp_hit.size == 0 and sl_hit.size == 0:
            label[i] = -1
            event_idx[i] = j_end
            continue
        first_tp = tp_hit[0] if tp_hit.size else np.iinfo(np.int32).max
        first_sl = sl_hit[0] if sl_hit.size else np.iinfo(np.int32).max
        if first_tp < first_sl:
            label[i] = 1
            event_idx[i] = i + 1 + first_tp
        elif first_sl < first_tp:
            label[i] = 0
            event_idx[i] = i + 1 + first_sl
        else:
            # równoczesne trafienie — traktuj jako TP (możesz zmienić politykę wedle uznania)
            label[i] = 1
            event_idx[i] = i + 1 + min(first_tp, first_sl)

    return pd.DataFrame({
        "tp_price": tp_price,
        "sl_price": sl_price,
        "horizon_idx": horizon_idx,
        "label": label,
        "event_idx": event_idx,
    })
