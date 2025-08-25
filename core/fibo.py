from __future__ import annotations
import pandas as pd
from typing import List, Tuple, Dict

Pivot = Tuple[int, float, str]  # (index, price, "H"/"L")

def find_pivots(df: pd.DataFrame, window: int = 5) -> List[Pivot]:
    highs = df["high"].values
    lows = df["low"].values
    pivots: List[Pivot] = []
    n = len(df)
    for i in range(window, n - window):
        loc_max = highs[i] == max(highs[i-window:i+window+1])
        loc_min = lows[i] == min(lows[i-window:i+window+1])
        if loc_max:
            pivots.append((i, highs[i], "H"))
        if loc_min:
            pivots.append((i, lows[i], "L"))
    # posortuj po indeksie
    pivots.sort(key=lambda x: x[0])
    return pivots

def last_impulse_from_pivots(pivots: List[Pivot]) -> Tuple[Pivot, Pivot] | None:
    # bierz ostatnie dwa pivoty o przeciwnym typie (H po L lub L po H)
    for i in range(len(pivots) - 1, 0, -1):
        a = pivots[i-1]; b = pivots[i]
        if a[2] != b[2]:
            return a, b
    return None

def fib_retracements_and_extensions(p0: float, p1: float) -> Dict[str, Dict[str, float]]:
    up = p1 > p0
    dist = abs(p1 - p0)
    if up:
        retr = {
            "0.382": p1 - 0.382 * dist,
            "0.5":   p1 - 0.5   * dist,
            "0.618": p1 - 0.618 * dist,
            "0.786": p1 - 0.786 * dist,
        }
        ext = {
            "1.272": p1 + 0.272 * dist,
            "1.618": p1 + 0.618 * dist,
        }
    else:
        retr = {
            "0.382": p1 + 0.382 * dist,
            "0.5":   p1 + 0.5   * dist,
            "0.618": p1 + 0.618 * dist,
            "0.786": p1 + 0.786 * dist,
        }
        ext = {
            "1.272": p1 - 0.272 * dist,
            "1.618": p1 - 0.618 * dist,
        }
    return {"retr": retr, "ext": ext, "direction": "up" if up else "down"}
