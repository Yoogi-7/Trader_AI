"""
Train LightGBM classifier for p_hit (TP1-before-SL).
- Loads dataset from build_dataset.py
- Simple time-based split (80/20)
- Saves model to data/models/p_hit_lgbm.joblib
"""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from app.config import DATA_DIR

DATASET_PATH = DATA_DIR / "datasets" / "btcusdt_tf10_30.parquet"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "p_hit_lgbm.joblib"

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run app.model.build_dataset first.")
    return pd.read_parquet(DATASET_PATH)

def train() -> Path:
    df = load_dataset()
    df = df.sort_values("ts")
    feats = [c for c in df.columns if c.startswith("f_")]
    X = df[feats]
    y = df["y"]

    # time-based split
    split_idx = int(len(df) * 0.8)
    X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
    X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]

    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    if len(X_te):
        proba = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        print(f"AUC (time-split): {auc:.3f}")

    joblib.dump((clf, feats), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return MODEL_PATH

if __name__ == "__main__":
    train()
