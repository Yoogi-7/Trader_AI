# core/cv.py
from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np

def purged_time_series_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Dzieli dane na kolejne okna testowe (równe bloki), z purgem (embargo) po obu stronach okna testowego.
    Train = wszystko przed (test_start - embargo) + wszystko po (test_end + embargo).
    """
    if n_splits < 2:
        raise ValueError("n_splits >= 2 required")
    fold_size = n_samples // n_splits
    if fold_size < 1:
        raise ValueError("Too few samples for given n_splits")

    for k in range(n_splits):
        start = k * fold_size
        end = n_samples if k == n_splits - 1 else (k + 1) * fold_size
        test_idx = np.arange(start, end, dtype=int)

        # purged train
        left_end = max(0, start - embargo)
        right_start = min(n_samples, end + embargo)

        left_train = np.arange(0, left_end, dtype=int)
        right_train = np.arange(right_start, n_samples, dtype=int)

        train_idx = np.concatenate([left_train, right_train])
        # Na wszelki wypadek usuń duplikaty/sortuj
        train_idx = np.unique(train_idx)
        yield train_idx, test_idx
