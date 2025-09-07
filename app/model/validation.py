"""Walk-forward skeleton with Optuna (stub) and calibration hooks.

The goal is to give you a ready slot for real validation without bloating files.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

def split_walk_forward(n: int, folds: int = 5, min_train: int = 500) -> Tuple[list, list]:
    idx = np.arange(n)
    fold_size = (n - min_train) // folds if folds > 0 else n - min_train
    splits = []
    for i in range(folds):
        train_end = min_train + i * fold_size
        val_end = min(n, train_end + fold_size)
        if val_end <= train_end:
            break
        splits.append((idx[:train_end], idx[train_end:val_end]))
    return splits

def calibrate_threshold(p: np.ndarray, y: np.ndarray, costs: Dict[str, float] | None = None) -> float:
    """Return probability cutoff that maximizes expected value.


    costs: {'tp': +value, 'fp': -value, 'fn': -value, 'tn': +value}

    """
    if costs is None:
        costs = {'tp': 1.0, 'fp': -1.0, 'fn': -0.5, 'tn': 0.0}
    grid = np.linspace(0.3, 0.9, 61)
    best_t, best_ev = 0.5, -1e9
    for t in grid:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        ev = tp*costs['tp'] + fp*costs['fp'] + fn*costs['fn'] + tn*costs['tn']
        if ev > best_ev:
            best_ev, best_t = ev, t
    return float(best_t)
