import os
import sqlite3
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from core.features import compute_features
from core.ml import load_meta_model

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_train_eval(conn, cfg):
    exch = cfg["exchange"]["id"]
    q = """
    SELECT id, ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, status
    FROM signals
    WHERE exchange=? AND status IN ('TP','SL','TP1_TRAIL')
    ORDER BY ts_ms ASC
    """
    return pd.read_sql_query(q, conn, params=(exch,))

def read_lookback(conn, exchange, symbol, timeframe, ts_ms, lookback):
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe=? AND ts_ms<=?
    ORDER BY ts_ms DESC
    LIMIT ?
    """
    d = pd.read_sql_query(q, conn, params=(exchange, symbol, timeframe, ts_ms, lookback))
    return d.sort_values("ts_ms").reset_index(drop=True)

def main():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    model_path = cfg["models"]["meta"]["model_path"]
    cal_path = cfg["models"]["meta"]["calibration"]["path"]
    lookback = max(cfg["models"]["meta"]["lookback_candles"], cfg["signals"]["lookback_candles"]) + 10

    model, feat_names = load_meta_model(model_path)
    if model is None:
        print("Model not found. Train it first: python train_meta.py")
        return

    conn = sqlite3.connect(db_path, timeout=60)
    try:
        df = read_train_eval(conn, cfg)
        if len(df) < 200:
            print(f"Not enough samples to calibrate: {len(df)} < 200")
            return

        split = int(len(df) * 0.75)
        hold = df.iloc[split:].copy()

        Xp, y = [], []
        for _, s in hold.iterrows():
            dfl = read_lookback(conn, cfg["exchange"]["id"], s["symbol"], s["timeframe"], int(s["ts_ms"]), lookback)
            if dfl.empty or len(dfl) < 60:
                continue
            feats = compute_features(dfl, s["direction"], float(s["entry"]), float(s["sl"]), float(s["tp1"]), cfg)
            if not feats:
                continue
            x = [feats.get(f, 0.0) for f in feat_names]
            Xp.append(x)
            y.append(1 if s["status"] in ("TP","TP1_TRAIL") else 0)

        if len(Xp) < 100:
            print("Too few holdout samples to calibrate.")
            return

        Xp = np.array(Xp, dtype=float)
        y = np.array(y, dtype=int)
        p_raw = model.predict_proba(Xp)[:,1]
        brier_before = brier_score_loss(y, p_raw)

        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_raw, y)
        p_cal = cal.predict(p_raw)
        brier_after = brier_score_loss(y, p_cal)

        os.makedirs(os.path.dirname(cal_path), exist_ok=True)
        joblib.dump({"calibrator": cal}, cal_path)
        print(f"Saved calibrator to {cal_path} | Brier before={brier_before:.3f}, after={brier_after:.3f}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
