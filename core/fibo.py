# core/fibo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


@dataclass
class SwingPair:
    """Para swingów określająca kierunek (up lub down) i cenę high/low."""
    start_idx: int
    end_idx: int
    direction: str  # "up" (low->high) lub "down" (high->low)
    low_price: float
    high_price: float


FIB_BASE = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXT  = [1.272, 1.618, 2.0]


def find_fractal_swings(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
    """
    Wyszukuje swing-high i swing-low metodą fraktalną:
    high[i] > max(high[i-k ... i-1], high[i+1 ... i+k]) → swing high
    low[i]  < min(low[i-k ... i-1], low[i+1 ... i+k])  → swing low
    """
    high = df["high"].astype(float).to_numpy()
    low  = df["low"].astype(float).to_numpy()
    n = len(df)
    sh, sl = [], []
    for i in range(lookback, n - lookback):
        left_h  = high[i - lookback:i]
        right_h = high[i + 1:i + 1 + lookback]
        if high[i] > np.max(left_h) and high[i] > np.max(right_h):
            sh.append(i)

        left_l  = low[i - lookback:i]
        right_l = low[i + 1:i + 1 + lookback]
        if low[i] < np.min(left_l) and low[i] < np.min(right_l):
            sl.append(i)
    return sh, sl


def latest_swing_pair(df: pd.DataFrame, lookback: int = 5, prefer: str = "auto") -> Optional[SwingPair]:
    """
    Zwraca ostatnią parę swingów (low->high dla 'up' lub high->low dla 'down').
    prefer: "auto" | "up" | "down"
    """
    sh, sl = find_fractal_swings(df, lookback=lookback)
    if not sh and not sl:
        return None

    # Złóż listę pivotów w porządku czasu: (idx, typ)
    pivots = [(i, "H") for i in sh] + [(i, "L") for i in sl]
    pivots.sort(key=lambda x: x[0])
    if len(pivots) < 2:
        return None

    # Ostatnia para różnych typów
    for i in range(len(pivots) - 1, 0, -1):
        a_idx, a_t = pivots[i - 1]
        b_idx, b_t = pivots[i]
        if a_t != b_t:
            # kierunek: jeśli a=Low, b=High → up, jeśli a=High, b=Low → down
            if a_t == "L" and b_t == "H":
                direction = "up"
                low_price = float(df["low"].iloc[a_idx])
                high_price = float(df["high"].iloc[b_idx])
                return SwingPair(a_idx, b_idx, direction, low_price, high_price)
            elif a_t == "H" and b_t == "L":
                direction = "down"
                low_price = float(df["low"].iloc[b_idx])
                high_price = float(df["high"].iloc[a_idx])
                return SwingPair(a_idx, b_idx, direction, low_price, high_price)

    # fallback: jeśli prefer wymusza kierunek, spróbuj ułożyć odpowiednią parę
    if prefer in ("up", "down") and len(pivots) >= 2:
        # weź dwie ostatnie różne świece i dopasuj low/high
        a_idx, a_t = pivots[-2]
        b_idx, b_t = pivots[-1]
        if prefer == "up":
            lo = min(float(df["low"].iloc[a_idx]), float(df["low"].iloc[b_idx]))
            hi = max(float(df["high"].iloc[a_idx]), float(df["high"].iloc[b_idx]))
            lo_idx = a_idx if float(df["low"].iloc[a_idx]) <= float(df["low"].iloc[b_idx]) else b_idx
            hi_idx = a_idx if float(df["high"].iloc[a_idx]) >= float(df["high"].iloc[b_idx]) else b_idx
            if lo_idx < hi_idx:
                return SwingPair(lo_idx, hi_idx, "up", lo, hi)
        else:
            hi = max(float(df["high"].iloc[a_idx]), float(df["high"].iloc[b_idx]))
            lo = min(float(df["low"].iloc[a_idx]), float(df["low"].iloc[b_idx]))
            hi_idx = a_idx if float(df["high"].iloc[a_idx]) >= float(df["high"].iloc[b_idx]) else b_idx
            lo_idx = a_idx if float(df["low"].iloc[a_idx]) <= float(df["low"].iloc[b_idx]) else b_idx
            if hi_idx < lo_idx:
                return SwingPair(hi_idx, lo_idx, "down", lo, hi)
    return None


def fib_levels_from_swing(swing: SwingPair) -> pd.DataFrame:
    """
    Buduje poziomy Fibonacciego (retracement + extension) dla podanej fali.
    Dla 'up': retracement liczony od HIGH w dół; extension powyżej HIGH.
    Dla 'down': retracement liczony od LOW w górę; extension poniżej LOW.
    """
    lo, hi = swing.low_price, swing.high_price
    rng = hi - lo
    levels = []

    if rng <= 0:
        return pd.DataFrame(columns=["label", "ratio", "price", "kind"])

    if swing.direction == "up":
        # retracement: od high w dół do low
        for r in FIB_BASE:
            price = hi - rng * r
            levels.append(("RET " + f"{int(r*100)}%", r, price, "retracement"))
        # extensions: od high w górę
        for r in FIB_EXT:
            price = hi + rng * (r - 1.0)
            levels.append(("EXT " + f"{int(r*100)}%", r, price, "extension"))
    else:
        # retracement: od low w górę do high (ale w spadku retracementy są 'do góry')
        for r in FIB_BASE:
            price = lo + rng * r
            levels.append(("RET " + f"{int(r*100)}%", r, price, "retracement"))
        # extensions: poniżej low
        for r in FIB_EXT:
            price = lo - rng * (r - 1.0)
            levels.append(("EXT " + f"{int(r*100)}%", r, price, "extension"))

    df = pd.DataFrame(levels, columns=["label", "ratio", "price", "kind"]).sort_values("price", ascending=False).reset_index(drop=True)
    return df


def fib_levels_from_manual(low_price: float, high_price: float, direction: str) -> pd.DataFrame:
    """
    Wersja manualna – użytkownik podaje ceny i kierunek.
    """
    return fib_levels_from_swing(SwingPair(0, 0, direction, float(low_price), float(high_price)))
