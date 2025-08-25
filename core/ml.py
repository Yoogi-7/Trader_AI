from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from core.features import compute_features

_MODEL = None
_MODEL_FEATURES: Optional[List[str]] = None
_CAL = None

def load_meta_model(model_path: str) -> Tuple[object, List[str] | None]:
    """Ładuje główny model (XGB/LightGBM) oraz listę cech."""
    global _MODEL, _MODEL_FEATURES
    if _MODEL is not None and _MODEL_FEATURES is not None:
        return _MODEL, _MODEL_FEATURES
    if not os.path.exists(model_path):
        return None, None
    bundle = joblib.load(model_path)
    _MODEL = bundle["model"]
    _MODEL_FEATURES = bundle["features"]
    return _MODEL, _MODEL_FEATURES

def load_meta_calibrator(cal_path: str):
    """Ładuje kalibrator (IsotonicRegression/Platt)."""
    global _CAL
    if _CAL is not None:
        return _CAL
    if not os.path.exists(cal_path):
        return None
    _CAL = joblib.load(cal_path)["calibrator"]
    return _CAL

def _predict_raw_proba(df: pd.DataFrame, sig: dict, cfg: dict, model, feat_names: list[str]) -> float:
    feats = compute_features(df, sig["direction"], sig["entry"], sig["sl"], sig["tp1"], cfg)
    if not feats:
        return 0.5
    x = np.array([[feats.get(f, 0.0) for f in feat_names]], dtype=float)
    proba = float(model.predict_proba(x)[0, 1])
    return proba

def predict_pwin(df: pd.DataFrame, sig: dict, cfg: dict, model, feat_names: list[str]) -> float:
    """Zwraca skalibrowane p(win) jeśli kalibrator jest dostępny i włączony."""
    p = _predict_raw_proba(df, sig, cfg, model, feat_names)
    cal_cfg = cfg.get("models", {}).get("meta", {}).get("calibration", {})
    if not cal_cfg or not cal_cfg.get("enabled", False):
        return p
    cal = load_meta_calibrator(cal_cfg.get("path", "models/meta_cal.pkl"))
    if cal is None:
        return p
    # isotonic/Platt w sklearn przyjmuje wektor
    p_cal = float(np.clip(cal.predict([[p]])[0], 0.0, 1.0))
    return p_cal
