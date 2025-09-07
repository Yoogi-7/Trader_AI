"""
Model loading and probability prediction helper.
- Gracefully falls back to a constant 0.60 if model is missing.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np

from app.config import DATA_DIR

MODEL_PATH = DATA_DIR / "models" / "p_hit_lgbm.joblib"
_loaded = {"model": None, "feats": None}

def load_model():
    if _loaded["model"] is None and MODEL_PATH.exists():
        model, feats = joblib.load(MODEL_PATH)
        _loaded["model"] = model
        _loaded["feats"] = feats
    return _loaded["model"], _loaded["feats"]

def predict_proba(features: dict) -> float:
    model, feats = load_model()
    if model is None:
        return 0.60  # fallback
    x = np.array([[features.get(k, 0.0) for k in feats]])
    p = float(model.predict_proba(x)[0, 1])
    # clamp conservative
    return max(0.01, min(0.99, p))
