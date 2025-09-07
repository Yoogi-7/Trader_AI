"""
Feature engineering for TRADER_AI.
- Works on resampled TF df and higher TFs (h1, h2)
- Produces numeric features suitable for LightGBM
All texts and comments are in English.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr_proxy(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return (df["high"] - df["low"]).rolling(n).mean()

def make_features(df_tf: pd.DataFrame, df_h1: pd.DataFrame, df_h2: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the entire TF frame (aligned to df_tf index).
    """
    out = pd.DataFrame(index=df_tf.index.copy())
    # TF-level features
    out["close"] = df_tf["close"]
    out["ema20"] = _ema(df_tf["close"], 20)
    out["ema50"] = _ema(df_tf["close"], 50)
    out["ema200"] = _ema(df_tf["close"], 200)
    out["rsi14"] = _rsi(df_tf["close"], 14)
    out["atrp14"] = _atr_proxy(df_tf, 14) / df_tf["close"]

    # Higher TF summaries (aligned by last value carry-forward)
    for name, d in [("h1", df_h1), ("h2", df_h2)]:
        e200 = _ema(d["close"], 200)
        up = (d["close"] > e200) & (e200.diff() > 0)
        down = (d["close"] < e200) & (e200.diff() < 0)
        feat = pd.DataFrame({
            f"{name}_ema200_pos": (d["close"] - e200) / d["close"],
            f"{name}_ema200_slope": e200.pct_change().fillna(0),
            f"{name}_trend_up": up.astype(int),
            f"{name}_trend_down": down.astype(int)
        }, index=d.index)
        # reindex to df_tf
        feat = feat.reindex(df_tf.index, method="ffill")
        out = out.join(feat, how="left")

    # Price position vs EMAs
    out["pos_ema20"] = (df_tf["close"] - out["ema20"]) / df_tf["close"]
    out["pos_ema50"] = (df_tf["close"] - out["ema50"]) / df_tf["close"]
    out["pos_ema200"] = (df_tf["close"] - out["ema200"]) / df_tf["close"]

    # Fill/clip
    out = out.fillna(method="ffill").fillna(method="bfill")
    out = out.replace([np.inf, -np.inf], 0)
    return out

def latest_feature_row(df_tf: pd.DataFrame, df_h1: pd.DataFrame, df_h2: pd.DataFrame) -> dict:
    feats = make_features(df_tf, df_h1, df_h2)
    return feats.iloc[-1].to_dict()
