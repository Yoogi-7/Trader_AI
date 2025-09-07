"""
Build a supervised dataset from historical BTCUSDT data:
- Detect candidates on signal TF (series mode),
- Compute features (TF + HTF),
- Propose levels, then label TP1-before-SL using 1m path and TF horizon,
- Save a Parquet dataset for training.
This is a compact, single-symbol builder (BTCUSDT) for MVP.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from app.config import POLICY, DATA_DIR
from app.data.store import path_raw
from app.data.resample import resample_to_tf
from app.rules.detectors import detect_candidates
from app.rules.levels import propose_levels
from app.features.engine import make_features
from app.labels.hits import label_tp1_before_sl

SYMBOL = "BTCUSDT"
DATASET_PATH = DATA_DIR / "datasets" / "btcusdt_tf10_30.parquet"
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

TF_ORDER = ["10m", "15m", "30m", "1h", "2h", "4h"]
def next_two_higher_tfs(tf: str):
    i = TF_ORDER.index(tf)
    return TF_ORDER[min(i+1, len(TF_ORDER)-1)], TF_ORDER[min(i+2, len(TF_ORDER)-1)]

def build() -> Path:
    df_1m = pd.read_parquet(path_raw(SYMBOL))

    rows: List[Dict[str, Any]] = []
    det = POLICY["detector_defaults"]

    # Use only 10m and 30m to keep dataset small for MVP
    for tf in ["10m", "30m"]:
        df_tf = resample_to_tf(df_1m, tf)
        h1, h2 = next_two_higher_tfs(tf)
        df_h1 = resample_to_tf(df_1m, h1)
        df_h2 = resample_to_tf(df_1m, h2)

        lookback = POLICY["tf_params"][tf].get("lookback", det["breakout_lookback"])
        # IMPORTANT: series_mode=True to emit events over history
        cands = detect_candidates(
            df_tf,
            lookback=lookback,
            use_fallback=det["use_fallback"],
            fallback_ma_len=det["fallback_ma_len"],
            fallback_squeeze_len=det["fallback_squeeze_len"],
            fallback_squeeze_threshold=det["fallback_squeeze_threshold"],
            series_mode=True,
        )
        if not cands:
            continue

        feats_all = make_features(df_tf, df_h1, df_h2)

        for c in cands:
            ts = c["i"]
            if ts not in feats_all.index:
                continue

            tf_cfg = POLICY["tf_params"][tf]
            entry, sl, tp1, tp2 = propose_levels(
                c["side"], c["swing_low"], c["swing_high"], c["atr"],
                k_atr_sl=tf_cfg["k_atr_sl"]
            )
            y = label_tp1_before_sl(
                df_1m=df_1m, side=c["side"], entry=entry, tp1=tp1, sl=sl,
                start_ts=ts, tf=tf, bars_horizon=tf_cfg["bars_horizon"]
            )
            if y is None:
                continue

            feat_row = feats_all.loc[ts].to_dict()
            row = {"ts": ts, "tf": tf, "side": c["side"], "y": int(y)}
            row.update({f"f_{k}": float(v) for k, v in feat_row.items()})
            rows.append(row)

    if not rows:
        raise RuntimeError("No dataset rows were generated. Try running backfill (365+ days) or loosening detector settings.")
    df_ds = pd.DataFrame(rows).sort_values("ts")
    df_ds.to_parquet(DATASET_PATH)
    return DATASET_PATH

if __name__ == "__main__":
    p = build()
    print(f"Dataset saved to {p}")
