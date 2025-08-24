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
    oof_index: np.ndarray  # indeksy X/y, gdzie proba_oof nie NaN

def time_series_fit_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, MLResult]:
    """
    Trenuje model w schemacie TimeSeriesSplit (expanding window) z kalibracją.
    Zwraca tablicę OOF z prawdopodobieństwami oraz info o ostatnim modelu.
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
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr, ytr)

        p = clf.predict_proba(Xte)[:, 1]
        proba_oof[te_idx] = p

        try:
            aucs.append(roc_auc_score(yte, p))
        except ValueError:
            pass

        last_model = clf

    mask = ~np.isnan(proba_oof)
    brier = brier_score_loss(y.iloc[mask], proba_oof[mask])

    result = MLResult(
        model=last_model,
        features=feats,
        auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
        auc_std=float(np.nanstd(aucs)) if aucs else float("nan"),
        brier=float(brier),
        oof_index=np.where(mask)[0],
    )
    return proba_oof, result

def threshold_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """
    Metryki trafności dla danego progu.
    Zwraca precision/recall/F1/accuracy + liczności TP/FP/TN/FN.
    """
    pred_pos = (p >= thr).astype(int)
    TP = int(((pred_pos == 1) & (y_true == 1)).sum())
    FP = int(((pred_pos == 1) & (y_true == 0)).sum())
    TN = int(((pred_pos == 0) & (y_true == 0)).sum())
    FN = int(((pred_pos == 0) & (y_true == 1)).sum())
    pos = int((pred_pos == 1).sum())

    precision = (TP / pos) if pos > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    acc = (TP + TN) / max(1, len(y_true))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "predicted_positives": pos,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }

def precision_recall_table(y_true: np.ndarray, p: np.ndarray, steps: int = 101) -> pd.DataFrame:
    """
    Tabela precision/recall/f1/support w funkcji progu z zakresu [0,1].
    """
    thr_list = np.linspace(0.0, 1.0, steps)
    rows = []
    for thr in thr_list:
        m = threshold_metrics(y_true, p, thr)
        m["thr"] = float(thr)
        rows.append(m)
    df = pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)
    return df

def suggest_threshold_by_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Zwraca próg maksymalizujący F1.
    """
    df = precision_recall_table(y_true, p)
    best = df.loc[df["f1"].idxmax()]
    return float(best["thr"])

# ===== Expectancy (R) w funkcji progu =====

def expectancy_from_precision(precision: float, tp_mult: float, sl_mult: float, cost_R: float = 0.0) -> float:
    """
    Szacuje expectancy w jednostkach R:
      R_win = tp_mult / sl_mult
      R_loss = 1
      E[R] = precision * R_win - (1 - precision) * R_loss - cost_R
    cost_R – koszt łączny (prowizja/slippage/latency) w jednostkach R na trade.
    """
    R_win = float(tp_mult) / float(sl_mult) if sl_mult != 0 else 0.0
    R_loss = 1.0
    return float(precision) * R_win - (1.0 - float(precision)) * R_loss - float(cost_R)

def expectancy_table(
    y_true: np.ndarray,
    p: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    cost_R: float = 0.0,
    steps: int = 101,
) -> pd.DataFrame:
    """
    Zwraca tabelę metryk + expectancy (R) dla progów z [0,1].
    """
    df = precision_recall_table(y_true, p, steps=steps)
    df["expectancy_R"] = df["precision"].apply(lambda pr: expectancy_from_precision(pr, tp_mult, sl_mult, cost_R))
    return df

def suggest_threshold_by_expectancy(
    y_true: np.ndarray, p: np.ndarray, tp_mult: float, sl_mult: float, cost_R: float = 0.0
) -> float:
    """
    Zwraca próg maksymalizujący expectancy (R).
    """
    df = expectancy_table(y_true, p, tp_mult, sl_mult, cost_R)
    best = df.loc[df["expectancy_R"].idxmax()]
    return float(best["thr"])
