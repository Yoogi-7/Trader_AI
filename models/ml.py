# models/ml.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

from core.cv import purged_time_series_splits


@dataclass
class MLResult:
    model: object
    features: list[str]
    auc_mean: float
    auc_std: float
    brier: float
    oof_index: np.ndarray
    # konformalny próg (opcjonalnie)
    conformal_tau: Optional[float] = None
    conformal_alpha: Optional[float] = None


@dataclass
class MetaResult:
    model: object
    features: list[str]
    auc_mean: float
    auc_std: float
    brier: float
    oof_rows: np.ndarray


class ConstantProba:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _balanced_sample_weight(y: pd.Series) -> np.ndarray:
    """Wagi ~ zbalansowane: w_c = N / (2 * N_c)."""
    y = y.astype(int)
    N = len(y)
    vc = y.value_counts()
    w = y.map({c: N / (2.0 * vc[c]) for c in vc.index}).astype(float).to_numpy()
    return w


def _fit_final_model_calibrated(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> object:
    y = y.astype(int)
    if y.nunique() < 2 or len(y) < 20:
        return ConstantProba(float(y.mean() if len(y) else 0.5))

    base = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.06,
        max_iter=600,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        max_bins=255,
    )
    sw = _balanced_sample_weight(y)
    base.fit(X, y, sample_weight=sw)

    try:
        min_class = y.value_counts().min()
        method = "isotonic" if (len(y) >= 400 and min_class >= 40) else "sigmoid"
        cv = 3 if min_class >= 3 else 2
        cal = CalibratedClassifierCV(base, method=method, cv=cv)
        cal.fit(X, y)
        return cal
    except Exception:
        return base


def time_series_fit_predict_proba(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: int = 0,
    random_state: int = 42,
    alpha_for_conformal: Optional[float] = None,
) -> Tuple[np.ndarray, MLResult]:
    """
    OOF z HGBClassifier + finalny model z kalibracją.
    Dodatkowo (opcjonalnie) wylicza konformalny próg tau z OOF:
      s_i = 1 - p_hat_i[y_i], tau = quantile_{1-alpha}(s).
      Do sygnałów przyjmujemy warunek p >= max(thr, 1 - tau).
    """
    feats = X.columns.tolist()
    y = y.astype(int)
    proba_oof = np.full(len(X), np.nan, dtype=float)
    aucs: list[float] = []
    last_model = None

    for tr_idx, te_idx in purged_time_series_splits(len(X), n_splits=n_splits, embargo=embargo):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        if len(ytr) < 20 or ytr.nunique() < 2:
            p_const = float(ytr.mean() if len(ytr) else 0.5)
            proba_oof[te_idx] = p_const
            last_model = ConstantProba(p_const)
            continue

        clf = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=500,
            l2_regularization=0.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            max_bins=255,
        )
        sw = _balanced_sample_weight(ytr)
        clf.fit(Xtr, ytr, sample_weight=sw)

        p = clf.predict_proba(Xte)[:, 1]
        proba_oof[te_idx] = p

        try:
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass

        last_model = clf

    final_model = _fit_final_model_calibrated(X, y, random_state=random_state)

    mask = ~np.isnan(proba_oof)
    brier = brier_score_loss(y.iloc[mask], proba_oof[mask]) if mask.any() else float("nan")

    conformal_tau = None
    if alpha_for_conformal is not None and mask.any():
        # nonconformity s = 1 - p(true_label) na OOF
        p_oof = proba_oof[mask]
        y_oof = y.iloc[mask].to_numpy()
        p_true = np.where(y_oof == 1, p_oof, 1.0 - p_oof)
        s = 1.0 - p_true
        alpha = float(np.clip(alpha_for_conformal, 1e-4, 0.99))
        q = np.quantile(s, 1.0 - alpha, method="higher")
        conformal_tau = float(q)

    result = MLResult(
        model=final_model,
        features=feats,
        auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
        auc_std=float(np.nanstd(aucs)) if aucs else float("nan"),
        brier=float(brier),
        oof_index=np.where(mask)[0],
        conformal_tau=conformal_tau,
        conformal_alpha=float(alpha_for_conformal) if alpha_for_conformal is not None else None,
    )
    return proba_oof, result


# ====== Metryki i progi ======

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
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "predicted_positives": pos,
            "precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": float(acc)}


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
    y_true: np.ndarray, p: np.ndarray, target_precision: float = 0.70, min_signals: int = 30
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


# ====== Expectancy (R i $) ======

def expectancy_from_precision(precision: float, tp_mult: float, sl_mult: float, cost_R: float = 0.0) -> float:
    R_win = float(tp_mult) / float(sl_mult) if sl_mult != 0 else 0.0
    R_loss = 1.0
    return float(precision) * R_win - (1.0 - float(precision)) * R_loss - float(cost_R)


def expectancy_table(
    y_true: np.ndarray, p: np.ndarray, tp_mult: float, sl_mult: float, cost_R: float = 0.0, steps: int = 101
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


def suggest_threshold_for_min_profit(
    y_true: np.ndarray,
    p: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    risk_dollars: float,
    min_profit_dollars: float,
    cost_R: float = 0.0,
    min_signals: int = 30,
) -> Tuple[Optional[float], bool]:
    df = expectancy_table(y_true, p, tp_mult, sl_mult, cost_R, steps=1001)
    df = df[df["predicted_positives"] >= int(min_signals)]
    if df.empty:
        return (None, False)
    df["exp_usd"] = df["expectancy_R"] * float(risk_dollars)
    ok = df[df["exp_usd"] >= float(min_profit_dollars)]
    if not ok.empty:
        cand = ok.sort_values(["predicted_positives", "expectancy_R"], ascending=[False, False]).iloc[0]
        return (float(cand["thr"]), True)
    best = df.sort_values(["exp_usd", "predicted_positives"], ascending=[False, False]).iloc[0]
    return (float(best["thr"]), False)


# ====== META (na p_base) ======

def build_meta_frame(X: pd.DataFrame, y: pd.Series, p_base_oof: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
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
    feats = X_meta.columns.tolist()
    proba_oof = np.full(len(X_meta), np.nan, dtype=float)
    aucs: list[float] = []

    y_meta = y_meta.astype(int)

    for tr_idx, te_idx in purged_time_series_splits(len(X_meta), n_splits=n_splits, embargo=embargo):
        Xtr, Xte = X_meta.iloc[tr_idx], X_meta.iloc[te_idx]
        ytr, yte = y_meta.iloc[tr_idx], y_meta.iloc[te_idx]

        if len(ytr) < 20 or ytr.nunique() < 2:
            p_const = float(ytr.mean() if len(ytr) else 0.5)
            proba_oof[te_idx] = p_const
            continue

        clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            max_bins=255,
        )
        sw = _balanced_sample_weight(ytr)
        clf.fit(Xtr, ytr, sample_weight=sw)
        p = clf.predict_proba(Xte)[:, 1]
        proba_oof[te_idx] = p

        try:
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass

    # final
    if len(y_meta) < 20 or y_meta.nunique() < 2:
        final_model = ConstantProba(float(y_meta.mean() if len(y_meta) else 0.5))
    else:
        clf_final = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=600,
            early_stopping=True, validation_fraction=0.1,
            random_state=random_state, max_bins=255
        )
        sw = _balanced_sample_weight(y_meta)
        clf_final.fit(X_meta, y_meta, sample_weight=sw)
        try:
            min_class = y_meta.value_counts().min()
            method = "isotonic" if (len(y_meta) >= 400 and min_class >= 40) else "sigmoid"
            cv = 3 if min_class >= 3 else 2
            cal = CalibratedClassifierCV(clf_final, method=method, cv=cv)
            cal.fit(X_meta, y_meta)
            final_model = cal
        except Exception:
            final_model = clf_final

    mask = ~np.isnan(proba_oof)
    brier = brier_score_loss(y_meta.iloc[mask], proba_oof[mask]) if mask.any() else float("nan")

    meta_res = MetaResult(
        model=final_model,
        features=feats,
        auc_mean=float(np.nanmean(aucs)) if aucs else float("nan"),
        auc_std=float(np.nanstd(aucs)) if aucs else float("nan"),
        brier=float(brier),
        oof_rows=np.where(mask)[0],
    )
    return proba_oof, meta_res


# ====== Kombinacja Base+Meta ======

def combined_metrics_for_thresholds(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_meta: np.ndarray,
    thr_base: float,
    thr_meta: float,
) -> dict:
    take_mask = (p_base >= thr_base) & (p_meta >= thr_meta)
    if take_mask.sum() == 0:
        return {"predicted_positives": 0, "TP": 0, "FP": 0, "FN": int((y_true == 1).sum()),
                "TN": int((y_true == 0).sum()), "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": float((y_true == 0).mean())}
    y_sel = y_true[take_mask]
    TP = int((y_sel == 1).sum())
    FP = int((y_sel == 0).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / max(1, int((y_true == 1).sum()))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (y_sel == 1).mean()
    return {"predicted_positives": int(take_mask.sum()), "TP": TP, "FP": FP,
            "FN": int((y_true == 1).sum()) - TP, "TN": int((y_true == 0).sum()) - FP,
            "precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": float(acc)}
