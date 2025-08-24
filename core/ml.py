from __future__ import annotations
from typing import Dict, Any
import json
import os
import pandas as pd
import numpy as np

# UWAGA: brak importu z core.signals -> unikamy cyklicznego importu
try:
    import joblib
except Exception:
    joblib = None  # pozwoli działać bez modelu (np. tylko reguły)

def _models_dir(cfg: Dict) -> str:
    # Domyślnie folder "models" w katalogu projektu
    return cfg.get("ml", {}).get("models_dir", "models")

def load_model(cfg: Dict) -> Dict[str, Any]:
    """
    Ładuje model (np. LogisticRegression) oraz metadane z plików:
      - models/lr_winprob.joblib
      - models/lr_winprob_meta.json
    Zwraca dict: {"pipe": model, "features": [...]}.
    Jeśli model nie istnieje lub brak joblib — zwraca pusty obiekt (inference będzie pomijał ML).
    """
    mdir = _models_dir(cfg)
    model_path = os.path.join(mdir, cfg.get("ml", {}).get("model_file", "lr_winprob.joblib"))
    meta_path  = os.path.join(mdir, cfg.get("ml", {}).get("meta_file",  "lr_winprob_meta.json"))

    if joblib is None or not os.path.exists(model_path) or not os.path.exists(meta_path):
        return {"pipe": None, "features": []}

    pipe = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta.get("features", [])
    return {"pipe": pipe, "features": feats}

def infer_proba(cfg: Dict, df: pd.DataFrame, ts: pd.Timestamp, direction: str, mtf_ok: bool) -> float:
    """
    Zwraca prawdopodobieństwo wygranej (proba[win]) na bazie cech z meta.
    Wymagane kolumny: rsi, macd_hist, ema200, close, vol_z, atr.
    Dodatkowe cechy: trend_up (close>ema200 & ema200 wzrostowa), htf_ok.
    """
    obj = load_model(cfg)
    pipe = obj.get("pipe")
    feats_names = obj.get("features", [])

    if pipe is None or not isinstance(df, pd.DataFrame) or ts not in df.index:
        # Brak modelu – zwracamy neutralne 0.5 (pozwala działać logice zewnętrznej)
        return 0.5

    row = df.loc[ts]
    # Bezpieczne ekstrakcje
    ema200 = float(row.get("ema200", np.nan)) if "ema200" in df.columns else np.nan
    close  = float(row.get("close", np.nan))

    # Pochylenie ema200: różnica z poprzedniej świecy
    ema200_slope_up = False
    try:
        ema200_slope_up = bool(df["ema200"].diff().loc[ts] > 0)
    except Exception:
        ema200_slope_up = False

    feats = {
        "rsi": float(row.get("rsi", 50.0)),
        "macdh": float(row.get("macd_hist", 0.0)),
        "dist_ema200_bps": float(((close - ema200) / ema200) * 1e4) if np.isfinite(ema200) else 0.0,
        "vol_z": float(row.get("vol_z", 0.0)),
        "atr_bps": float(row.get("atr", 0.0) / close * 1e4) if (close and close != 0) else 0.0,
        "trend_up": float(1.0 if (np.isfinite(ema200) and close > ema200 and ema200_slope_up) else 0.0),
        "htf_ok": float(1.0 if mtf_ok else 0.0),
        # kierunek jako cecha binarna (opcjonalnie – jeśli meta zawiera tę nazwę)
        "dir_long": float(1.0 if direction == "long" else 0.0),
    }

    x = pd.DataFrame([{k: feats.get(k, 0.0) for k in feats_names}])
    try:
        p = float(pipe.predict_proba(x)[0, 1])
    except Exception:
        # Model może być np. klasyfikator bez predict_proba – fallback
        y = pipe.predict(x)
        p = float(y[0]) if hasattr(y, "__len__") else float(y)
        # znormalizuj do [0,1]
        p = max(0.0, min(1.0, p))
    return p

def train_and_save(*args, **kwargs):
    """
    Bezpieczny stub – jeśli chcesz uczyć model w locie, możesz
    zaimplementować tutaj trening i zapis modelu + meta.
    Pozostawiam jako opcjonalne, aby dashboard nie rzucał ImportError.
    """
    raise NotImplementedError("Training routine is not implemented in this build.")
