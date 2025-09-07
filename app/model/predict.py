"""LightGBM probability wrapper (lazy loading).

If a trained model file exists at ``app/model/model.pkl``, it will be loaded
on first call and used to return the positive-class probability.
Otherwise, we fall back to a simple baseline (``BASELINE_P_HIT``) or the
``p_base`` value provided in the `features` dict.

Keep this file small and readable—no heavy dependencies at import time.
"""
from __future__ import annotations

from pathlib import Path

MODEL_PATH = Path(__file__).with_name("model.pkl")
BASELINE_P_HIT: float = 0.60  # sensible default when model is missing

_model = None  # loaded on first use


def predict_proba(features: dict) -> float:
    """Return probability of success (0..1).

    The function attempts to:
    1) Load a model from ``MODEL_PATH`` lazily.
    2) If the model exposes ``predict_proba`` (sklearn/lightgbm style),
       return the probability for class 1.
    3) If the model exposes ``predict`` and returns a float within [0, 1],
       pass it through as probability.
    4) Otherwise, return ``features.get("p_base", BASELINE_P_HIT)``.

    Parameters
    ----------
    features : dict
        A flat dict of numeric features (already engineered upstream).

    Returns
    -------
    float
        Probability in range [0.0, 1.0]. Guaranteed to be a float.
    """
    global _model

    # 1) Lazy-load model if available
    if _model is None and MODEL_PATH.exists():
        # Import here to avoid heavy deps at import time
        import joblib  # type: ignore[import-not-found]
        _model = joblib.load(MODEL_PATH)

    # 2) Use the model if we have one
    if _model is not None:
        # Import numpy only if we actually infer with the model
        import numpy as np  # type: ignore[import-not-found]

        # Order of features must match the model's training schema.
        # For MVP we just pass values in insertion order.
        x = np.array([list(features.values())], dtype=float)

        proba = getattr(_model, "predict_proba", None)
        if callable(proba):
            try:
                p = proba(x)
                # sklearn-style: shape (n, 2). Take column for class 1.
                if hasattr(p, "__getitem__"):
                    p1 = float(p[0][1])
                    if 0.0 <= p1 <= 1.0:
                        return p1
            except Exception:
                # fall through to other strategies/baseline
                pass

        predict = getattr(_model, "predict", None)
        if callable(predict):
            try:
                y = predict(x)
                y0 = float(y[0]) if hasattr(y, "__getitem__") else float(y)
                if 0.0 <= y0 <= 1.0:
                    return y0
            except Exception:
                # fall through to baseline
                pass

    # 3) Baseline fallback
    try:
        return float(features.get("p_base", BASELINE_P_HIT))
    except Exception:
        return BASELINE_P_HIT
