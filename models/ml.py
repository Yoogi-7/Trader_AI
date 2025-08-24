# models/ml.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

@dataclass
class MLResult:
    model: object
    features: list[str]
    auc_mean: float
    auc_std: float
    brier: float

def time_series_fit_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, MLResult]:
    """
    Trenuje model w układzie TimeSeriesSplit na expanding window.
    Zwraca out-of-fold proby dla całego zbioru oraz wynik z ostatniego modelu (kalendarzowo najnowszy).
    """
    feats = X.columns.tolist()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba_oof = np.full(len(X), np.nan, dtype=float)
    aucs = []

    last_model = None

    for tr_idx, te_idx in tscv.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        base = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        )
        # kalibracja prawdopodobieństw
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr, ytr)

        p = clf.predict_proba(Xte)[:, 1]
        proba_oof[te_idx] = p

        try:
            aucs.append(roc_auc_score(yte, p))
        except ValueError:
            pass

        last_model = clf

    brier = brier_score_loss(y.iloc[~np.isnan(proba_oof)], proba_oof[~np.isnan(proba_oof)])

    result = MLResult(
        model=last_model,
        features=feats,
        auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
        auc_std=float(np.nanstd(aucs)) if aucs else float("nan"),
        brier=float(brier),
    )
    return proba_oof, result
