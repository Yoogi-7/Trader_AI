import os
import sqlite3
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, brier_score_loss
from xgboost import XGBClassifier
from core.features import compute_features

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_signals_for_training(conn: sqlite3.Connection, exchange: str) -> pd.DataFrame:
    q = """
    SELECT id, ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, tp2, status
    FROM signals
    WHERE exchange=? AND status IN ('TP','SL','TP1_TRAIL')
    ORDER BY ts_ms ASC
    """
    return pd.read_sql_query(q, conn, params=(exchange,))

def read_lookback_df(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str, ts_ms: int, lookback: int) -> pd.DataFrame:
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe=? AND ts_ms<=?
    ORDER BY ts_ms DESC
    LIMIT ?
    """
    df = pd.read_sql_query(q, conn, params=(exchange, symbol, timeframe, ts_ms, lookback))
    if df.empty:
        return df
    return df.sort_values("ts_ms").reset_index(drop=True)

def build_dataset(conn: sqlite3.Connection, cfg: dict) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    exchange = cfg["exchange"]["id"]
    lookback = max(cfg["models"]["meta"]["lookback_candles"], cfg["signals"]["lookback_candles"]) + 10
    sigs = read_signals_for_training(conn, exchange)
    rows = []
    for _, s in sigs.iterrows():
        df = read_lookback_df(conn, exchange, s["symbol"], s["timeframe"], int(s["ts_ms"]), lookback)
        if df.empty or len(df) < 60:
            continue
        feats = compute_features(df, s["direction"], float(s["entry"]), float(s["sl"]), float(s["tp1"]), cfg)
        if not feats:
            continue
        y = 1 if s["status"] in ("TP", "TP1_TRAIL") else 0
        feats["y"] = y
        rows.append(feats)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), []

    data = pd.DataFrame(rows)
    feature_names = [c for c in data.columns if c != "y"]
    X = data[feature_names].astype(float).fillna(0.0)
    y = data["y"].astype(int)
    return X, y, feature_names

def purged_ts_cv_indices(n: int, n_splits: int = 5, embargo: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Prosty purged walk-forward:
      - dzieli dane na n_splits kolejnymi blokami czasowymi,
      - usuwa 'embargo' obserwacji po każdym train, by nie przeciekała informacja.
    Zwraca listę (train_idx, test_idx).
    """
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    idx = np.arange(n)
    offsets = np.cumsum(fold_sizes)
    splits = []
    start = 0
    for k in range(n_splits):
        stop = offsets[k]
        test_idx = idx[start:stop]
        train_end = max(0, start - embargo)
        train_idx = idx[:train_end]
        if len(train_idx) == 0:
            # pierwszy fold bez train – pomijamy
            start = stop
            continue
        splits.append((train_idx, test_idx))
        start = stop
    return splits

def train_and_save(X: pd.DataFrame, y: pd.Series, feature_names: list[str], model_path: str):
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )
    # Purged CV
    splits = purged_ts_cv_indices(len(X), n_splits=5, embargo=20)
    aucs, precs, briers = [], [], []
    thr = 0.6
    for tr, te in splits:
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        briers.append(brier_score_loss(y.iloc[te], p))
        precs.append(precision_score(y.iloc[te], (p >= thr).astype(int), zero_division=0))
    if aucs:
        print(f"[CV] AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f} | "
              f"prec@{thr}={np.mean(precs):.3f} | Brier={np.mean(briers):.3f}")

    # final fit
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": clf, "features": feature_names}, model_path)
    print(f"Saved model to {model_path} with {len(feature_names)} features.")

def main():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    model_path = cfg["models"]["meta"]["model_path"]
    with sqlite3.connect(db_path, timeout=60) as conn:
        X, y, feat_names = build_dataset(conn, cfg)
    need = int(cfg["models"]["meta"]["min_signals_for_training"])
    if X.empty or len(y) < need:
        print(f"Not enough data to train. Got {len(y)} samples (need {need}).")
        return
    train_and_save(X, y, feat_names, model_path)

if __name__ == "__main__":
    main()
