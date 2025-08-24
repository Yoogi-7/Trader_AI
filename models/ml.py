# models/ml.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from core.cv import purged_time_series_splits

# ========= Wyniki =========

@dataclass
class MLResult:
    model: object
    features: list[str]
    auc_mean: float
    auc_std: float
    brier: float
    oof_index: np.ndarray  # indeksy, gdzie proba_oof nie NaN

@dataclass
class MetaResult:
    model: object
    features: list[str]     # nazwy cech meta (w tym 'p_base')
    auc_mean: float
    auc_std: float
    brier: float
    oof_rows: np.ndarray    # lokalne indeksy w X_meta/y_meta (0..len-1)
    base_thr_used: float

# ========= Pomocnicze =========

class ConstantProba:
    """Awaryjny 'model' zwracający stałe prawdopodobieństwo."""
    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])

def _fit_final_model_calibrated(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> object:
    """Trenuje model na CAŁOŚCI i kalibruje z bezpiecznymi fallbackami."""
    y = y.astype(int)
    if y.nunique() < 2 or len(y) < 10:
        # zbyt mało/próbki jednoklasowe → stała proba
        return ConstantProba(float(y.mean() if len(y) else 0.5))

    base = RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=random_state,
    )
    base.fit(X, y)

    # Bezpieczny wybór kalibracji
    min_class = y.value_counts().min()
    # preferuj isotonic przy większych próbkach, w innym razie sigmoid
    method = "isotonic" if (len(y) >= 200 and min_class >= 20) else "sigmoid"
    cv = 3 if min_class >= 3 else 2

    try:
        cal = CalibratedClassifierCV(base, method=method, cv=cv)
        cal.fit(X, y)
        return cal
    except Exception:
        # Gdyby kalibracja zawiodła — użyj niekalibrowanego RF
        return base

# ========= Model bazowy (OOF bez kalibracji), finalny z kalibracją =========

def time_series_fit_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: int = 0,
    random_state: int = 42,
) -> Tuple[np.ndarray, MLResult]:
    """
    OOF: trenowanie bez kalibracji (stabilność na małych foldach).
    Finalny model: trenowany na całości + bezpieczna kalibracja.
    """
    feats = X.columns.tolist()
    proba_oof = np.full(len(X), np.nan, dtype=float)
    aucs = []
    last_model = None

    y = y.astype(int)

    for tr_idx, te_idx in purged_time_series_splits(len(X), n_splits=n_splits, embargo=embargo):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        if len(ytr) < 10 or ytr.nunique() < 2:
            # Fallback: stała proba = udział klasy 1 w train
            p_const = float(ytr.mean() if len(ytr) else 0.5)
            p = np.full(len(Xte), p_const, dtype=float)
            proba_oof[te_idx] = p
            # AUC dla stałej predykcji nie ma sensu → pomijamy
            last_model = ConstantProba(p_const)
            continue

        base = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        )
        base.fit(Xtr, ytr)
        p = base.predict_proba(Xte)[:, 1]
        proba_oof[te_idx] = p

        try:
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass

        last_model = base  # tymczasowo

    # Finalny model na całości + kalibracja
    final_model = _fit_final_model_calibrated(X, y, random_state=random_state)

    mask = ~np.isnan(proba_oof)
    brier = brier_score_loss(y.iloc[mask], proba_oof[mask]) if mask.any() else float("nan")

    result = MLResult(
        model=final_model,
        features=feats,
        auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
        auc_std=float(np.nanstd(aucs)) if aucs else float("nan"),
        brier=float(brier),
        oof_index=np.where(mask)[0],
    )
    return proba_oof, result

# ========= Metryki/progi =========

def threshold_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
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
    thr_list = np.linspace(0.0, 1.0, steps)
    rows = []
    for thr in thr_list:
        m = threshold_metrics(y_true, p, float(thr))
        m["thr"] = float(thr)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)

def suggest_threshold_by_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    df = precision_recall_table(y_true, p, steps=501)
    best = df.loc[df["f1"].idxmax()]
    return float(best["thr"])

def suggest_threshold_for_precision(
    y_true: np.ndarray,
    p: np.ndarray,
    target_precision: float = 0.70,
    min_signals: int = 30,
) -> Optional[float]:
    df = precision_recall_table(y_true, p, steps=1001)
    df = df[df["predicted_positives"] >= int(min_signals)]
    if df.empty:
        return None
    ok = df[df["precision"] >= float(target_precision)]
    if not ok.empty:
        best = ok.sort_values(["predicted_positives", "thr"], ascending=[False, True]).iloc[0]
        return float(best["thr"])
    return float(df.loc[df["precision"].idxmax(), "thr"])

# ========= Expectancy =========

def expectancy_from_precision(precision: float, tp_mult: float, sl_mult: float, cost_R: float = 0.0) -> float:
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
    df = precision_recall_table(y_true, p, steps=steps)
    df["expectancy_R"] = df["precision"].apply(lambda pr: expectancy_from_precision(pr, tp_mult, sl_mult, cost_R))
    return df

def suggest_threshold_by_expectancy(
    y_true: np.ndarray, p: np.ndarray, tp_mult: float, sl_mult: float, cost_R: float = 0.0
) -> float:
    df = expectancy_table(y_true, p, tp_mult, sl_mult, cost_R, steps=501)
    best = df.loc[df["expectancy_R"].idxmax()]
    return float(best["thr"])

# ========= META-labeling =========

def build_meta_frame(
    X: pd.DataFrame,
    y: pd.Series,
    p_base_oof: np.ndarray,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Meta-cecha na start: p_base (OOF)."""
    mask = ~np.isnan(p_base_oof)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return pd.DataFrame(), pd.Series(dtype=int)

    X_meta = pd.DataFrame({"p_base": p_base_oof[mask]}, index=X.index[idx])
    y_meta = y.loc[X_meta.index].astype(int)
    return X_meta, y_meta

def meta_time_series_fit_predict_proba(
    X_meta: pd.DataFrame,
    y_meta: pd.Series,
    n_splits: int = 5,
    embargo: int = 0,
    random_state: int = 42,
) -> Tuple[np.ndarray, MetaResult]:
    """Meta OOF + finalny meta model (z kalibracją bezpieczną)."""
    feats = X_meta.columns.tolist()
    proba_oof = np.full(len(X_meta), np.nan, dtype=float)
    aucs = []
    last_model = None

    y_meta = y_meta.astype(int)

    for tr_idx, te_idx in purged_time_series_splits(len(X_meta), n_splits=n_splits, embargo=embargo):
        Xtr, Xte = X_meta.iloc[tr_idx], X_meta.iloc[te_idx]
        ytr, yte = y_meta.iloc[tr_idx], y_meta.iloc[te_idx]

        if len(ytr) < 10 or ytr.nunique() < 2:
            p_const = float(ytr.mean() if len(ytr) else 0.5)
            p = np.full(len(Xte), p_const, dtype=float)
            proba_oof[te_idx] = p
            last_model = ConstantProba(p_const)
            continue

        base = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=random_state,
        )
        base.fit(Xtr, ytr)
        p = base.predict_proba(Xte)[:, 1]
        proba_oof[te_idx] = p

        try:
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass

        last_model = base

    # finalny meta model na całości + kalibracja
    final_model = _fit_final_model_calibrated(X_meta, y_meta, random_state=random_state)

    mask = ~np.isnan(proba_oof)
    brier = brier_score_loss(y_meta.iloc[mask], proba_oof[mask]) if mask.any() else float("nan")

    meta_res = MetaResult(
        model=final_model,
        features=feats,
        auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
        auc_std=float(np.nanstd(aucs)) if aucs else float("nan"),
        brier=float(brier),
        oof_rows=np.where(mask)[0],
        base_thr_used=float("nan"),
    )
    return proba_oof, meta_res

# ========= Kombinacja BASE+META (gating) =========

def combined_metrics_for_thresholds(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_meta: np.ndarray,
    thr_base: float,
    thr_meta: float,
) -> dict:
    take_mask = (p_base >= thr_base) & (p_meta >= thr_meta)
    if take_mask.sum() == 0:
        return {
            "predicted_positives": 0, "TP": 0, "FP": 0, "FN": int((y_true == 1).sum()),
            "TN": int((y_true == 0).sum()), "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": float((y_true == 0).mean())
        }
    y_sel = y_true[take_mask]
    TP = int((y_sel == 1).sum())
    FP = int((y_sel == 0).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / max(1, int((y_true == 1).sum()))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (y_sel == 1).mean()
    return {
        "predicted_positives": int(take_mask.sum()),
        "TP": TP,
        "FP": FP,
        "FN": int((y_true == 1).sum()) - TP,
        "TN": int((y_true == 0).sum()) - FP,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }

def suggest_meta_threshold_for_precision(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_meta: np.ndarray,
    thr_base: float,
    target_precision: float = 0.70,
    min_signals: int = 30,
) -> Optional[float]:
    thr_list = np.linspace(0.0, 1.0, 1001)
    best_thr = None
    best_count = -1
    best_prec = 0.0

    base_mask = (p_base >= thr_base)
    if base_mask.sum() < min_signals:
        return None

    for thr_m in thr_list:
        take = base_mask & (p_meta >= thr_m)
        cnt = int(take.sum())
        if cnt < min_signals:
            continue
        y_sel = y_true[take]
        if len(y_sel) == 0:
            continue
        prec = (y_sel == 1).mean()
        if prec >= target_precision:
            if cnt > best_count or (cnt == best_count and (best_thr is None or thr_m < best_thr)):
                best_thr = float(thr_m)
                best_count = cnt
                best_prec = float(prec)

    if best_thr is not None:
        return best_thr

    # fallback: maks precision przy min_signals
    for thr_m in thr_list[::-1]:
        take = base_mask & (p_meta >= thr_m)
        cnt = int(take.sum())
        if cnt >= min_signals:
            y_sel = y_true[take]
            if len(y_sel) == 0:
                continue
            prec = (y_sel == 1).mean()
            if prec > best_prec:
                best_prec = float(prec)
                best_thr = float(thr_m)
    return best_thr
