"""
Labeling utilities using 1-minute reconstruction:
- For a candidate at time T on TF, we look forward a horizon (bars_horizon * TF minutes)
- We check intrabar path on 1m between (T, T + horizon]:
  * LONG: did high reach TP before low reached SL?
  * SHORT: did low reach TP before high reached SL?
Returns 1 if TP1-before-SL, else 0; None if not resolvable.
"""
from __future__ import annotations
import pandas as pd
from typing import Optional

def minutes_of(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unknown TF: {tf}")

def label_tp1_before_sl(
    df_1m: pd.DataFrame,
    side: str,
    entry: float,
    tp1: float,
    sl: float,
    start_ts,          # pandas Timestamp (right-edge of signal bar)
    tf: str,
    bars_horizon: int
) -> Optional[int]:
    horizon_min = minutes_of(tf) * bars_horizon
    # interval (start_ts, start_ts + horizon] in 1m
    s = df_1m[df_1m.index > start_ts]
    e = df_1m[df_1m.index <= start_ts + pd.Timedelta(minutes=horizon_min)]
    window = s.loc[:e.index.max()] if len(e) else s
    if window.empty:
        return None

    if side == "LONG":
        # Did we hit TP1 before SL?
        # find first index where high >= tp1 or low <= sl
        hit_tp = window.index[(window["high"] >= tp1)]
        hit_sl = window.index[(window["low"] <= sl)]
    else:
        hit_tp = window.index[(window["low"] <= tp1)]
        hit_sl = window.index[(window["high"] >= sl)]

    t_tp = hit_tp[0] if len(hit_tp) else None
    t_sl = hit_sl[0] if len(hit_sl) else None

    if t_tp is None and t_sl is None:
        return 0
    if t_tp is not None and (t_sl is None or t_tp <= t_sl):
        return 1
    return 0
