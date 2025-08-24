# core/utils.py
from __future__ import annotations
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def save_model_artifacts(model: Any, meta: dict, out_dir: str, base_name: str = "model") -> tuple[str, str]:
    """
    Zapisuje model (joblib/pickle kompatybilny) + meta.json w katalogu out_dir.
    Zwraca (model_path, meta_path).
    """
    ensure_dir(out_dir)
    try:
        import joblib  # scikit-learn ma joblib jako zależność
    except Exception as e:
        raise RuntimeError("Brak 'joblib'. Zainstaluj: python -m pip install joblib") from e

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(out_dir, f"{base_name}_{ts}.joblib")
    meta_path = os.path.join(out_dir, f"{base_name}_{ts}.json")

    joblib.dump(model, model_path)

    meta_to_save = {}
    for k, v in meta.items():
        if is_dataclass(v):
            meta_to_save[k] = asdict(v)
        else:
            try:
                json.dumps(v)
                meta_to_save[k] = v
            except TypeError:
                meta_to_save[k] = str(v)

    meta_to_save["saved_at_utc"] = utc_now_iso()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_to_save, f, indent=2, ensure_ascii=False)

    return model_path, meta_path

def load_model_artifacts(model_path: str, meta_path: str | None = None) -> tuple[Any, dict | None]:
    """
    Wczytuje model i (opcjonalnie) metadane. Jeśli meta_path brak, spróbuje <model>.json.
    """
    try:
        import joblib
    except Exception as e:
        raise RuntimeError("Brak 'joblib'. Zainstaluj: python -m pip install joblib") from e

    model = joblib.load(model_path)
    meta = None
    if meta_path is None:
        base, _ = os.path.splitext(model_path)
        guess = base + ".json"
        if os.path.exists(guess):
            meta_path = guess

    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, meta
