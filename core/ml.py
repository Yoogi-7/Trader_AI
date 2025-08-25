from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from core.features import compute_features

_MODEL = None
_MODEL_FEATURES = None

def load_meta_model(model_path: str) -> Tuple[object, list[str]]:
    global _MODEL, _MODEL_FEATURES
    if _MODEL is not None and _MODEL_FEATURES is not None:
        return _MODEL, _MODEL_FEATURES
    if not os.path.exists(model_path):
        return None, None
    bundle = joblib.load(model_path)
    _MODEL = bundle["model"]
    _MODEL_FEATURES = bundle["features"]
    return _MODEL, _MODEL_FEATURES

def predict_pwin_from_df(df: pd.DataFrame, sig: dict, cfg: dict, model, feature_names: list[str]) -> float:
    feats = compute_features(df, sig["direction"], sig["entry"], sig["sl"], sig["tp1"], cfg)
    if not feats:
        return 0.5
    # ułóż w ustalonej kolejności; brakujące -> 0.0
    x = np.array([[feats.get(f, 0.0) for f in feature_names]], dtype=float)
    proba = float(model.predict_proba(x)[0, 1])  # prawdopodobieństwo klasy "win"
    return proba
