"""Scanning pipeline using Binance OHLCV + ATR/ADX/trend filters.

Steps per symbol/TF:
  1) Backfill 1m from Binance and persist (resampled) OHLCV to SQLite (table `ohlcv`).
  2) Build features: EMA200 trend, ATR, ADX, basic momentum.
  3) Detect setup: breakout OR momentum fallback with filters (trend+volatility+ADX).
  4) Propose levels: ATR-based SL/TP, compute RR.
  5) Estimate p_hit (baseline or via model.predict_proba if available).
  6) Compute EV with fee and slippage; keep only EV > 0.
  7) Return list of signal dicts (and optionally persist to `signals` via db.insert_signals).

Keep this module compact and dependency-light.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, List, Optional, Tuple

import math
import pandas as pd

from app.data.exchange import backfill_ohlcv_1m, resample, normalize_symbol
from app.storage.market import upsert_ohlcv
from app.storage.db import SignalDTO, insert_signals
from app.model.predict import predict_proba

# -------------------------- Feature helpers -------------------------------

def _tf_to_rule(tf: str) -> str:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return tf.replace("m", "min")
    return tf  # e.g., 1h, 2h, 4h are fine

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
    tr = atr(df, length=1).replace(0, 1e-9)
    plus_di = 100 * (plus_dm.ewm(span=length, adjust=False).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(span=length, adjust=False).mean() / tr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)) * 100
    return dx.ewm(span=length, adjust=False).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema200"] = ema(out["close"], 200)
    out["atr"] = atr(out, 14)
    out["adx"] = adx(out, 14)
    out["trend"] = (out["close"] > out["ema200"]).map(lambda x: "up" if x else "down")
    out["roc"] = out["close"].pct_change(3)
    out["vol_ok"] = (out["atr"] / out["close"]) >= 0.002  # 0.2% ATR floor
    return out

# -------------------------- Detection & levels ----------------------------

def detect_setup(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Simple breakout with momentum fallback.

    Returns dict with keys: side, swing_low, swing_high, atr
    or None if no setup.
    """
    if len(df) < 50:
        return None
    last = df.iloc[-1]
    window = df.iloc[-20:]  # 20 bars lookback

    # Require volatility and regime
    if not bool(last["vol_ok"]):
        return None
    if float(last["adx"]) < 18:  # skip dead trend
        return None

    high20 = window["high"].max()
    low20 = window["low"].min()
    side: Optional[str] = None
    # breakout
    if last["close"] > high20 and last["trend"] == "up":
        side = "long"
    elif last["close"] < low20 and last["trend"] == "down":
        side = "short"
    else:
        # momentum fallback: trend + ROC sign
        if last["trend"] == "up" and last["roc"] > 0:
            side = "long"
        elif last["trend"] == "down" and last["roc"] < 0:
            side = "short"

    if not side:
        return None

    return {
        "side": side,
        "swing_low": float(window["low"].min()),
        "swing_high": float(window["high"].max()),
        "atr": float(last["atr"]),
    }

def propose_levels(side: str, price: float, atr_val: float, k_sl: float, k_tp: float) -> Tuple[float, float]:
    """Return (sl, tp) based on ATR multipliers around current price."""
    if side == "long":
        sl = price - k_sl * atr_val
        tp = price + k_tp * atr_val
    else:
        sl = price + k_sl * atr_val
        tp = price - k_tp * atr_val
    return float(sl), float(tp)

# -------------------------- EV & costs ------------------------------------

def estimate_ev(side: str, price: float, sl: float, tp: float, p_hit: float, fee_bps: float, slippage_bps: float) -> float:
    """Compute simple EV per 1 notional.

    fee and slippage are applied on entry+exit roughly as bps of price.
    """
    risk = abs(price - sl)
    reward = abs(tp - price)
    ev = p_hit * reward - (1.0 - p_hit) * risk
    # subtract costs (entry + exit): fee twice, slippage twice
    cost = (fee_bps + slippage_bps) * 2 * price / 10_000.0
    return float(ev - cost)

# -------------------------- Main API --------------------------------------

def scan_symbols(
    symbols: List[str],
    tfs: List[str],
    days: int = 120,
    fee_bps: float = 6.0,         # 0.06% taker typical
    slippage_bps: float = 2.0,    # 0.02% conservative
    k_sl_atr: float = 1.5,
    k_tp_atr: float = 2.5,
    persist_signals: bool = True,
    persist_ohlcv: bool = True,
) -> List[Dict[str, Any]]:
    """Run full scan and (optionally) persist results to DB."""
    out: List[Dict[str, Any]] = []
    for raw_symbol in symbols:
        sym = normalize_symbol(raw_symbol)
        # 1) backfill 1m
        df_1m = backfill_ohlcv_1m(sym, days=days)
        if df_1m.empty:
            continue
        for tf in tfs:
            rule = _tf_to_rule(tf)
            df_tf = resample(df_1m, rule)
            if df_tf.empty or len(df_tf) < 200:
                continue
            # 2) features
            feat = build_features(df_tf)
            # 3) detect
            setup = detect_setup(feat)
            if not setup:
                continue
            last = feat.iloc[-1]
            price = float(last["close"])
            # 4) levels
            sl, tp = propose_levels(setup["side"], price, setup["atr"], k_sl_atr, k_tp_atr)
            rr = abs(tp - price) / max(abs(price - sl), 1e-9)
            # 5) p_hit via model or baseline
            features_for_model = {
                "rr": rr,
                "atr_pct": float(last["atr"] / price),
                "adx": float(last["adx"]),
                "trend_up": 1.0 if last["trend"] == "up" else 0.0,
                "roc": float(last["roc"]),
                "p_base": 0.60,
            }
            p_hit = float(predict_proba(features_for_model))
            # 6) EV with costs
            ev = estimate_ev(setup["side"], price, sl, tp, p_hit, fee_bps, slippage_bps)
            if ev <= 0:
                continue
            # (optional) persist ohlcv
            if persist_ohlcv:
                rows = [
                    dict(ts=int(r.ts), symbol=sym, timeframe=tf,
                         open=float(r.open), high=float(r.high), low=float(r.low),
                         close=float(r.close), volume=float(r.volume))
                    for r in df_tf.tail(500).itertuples(index=False)  # limit writes
                ]
                try:
                    upsert_ohlcv(rows)
                except Exception:
                    pass
            signal = {
                "ts": int(last.name.value // 10**9) if hasattr(last.name, "value") else int(last["ts"]),
                "symbol": sym.replace("/", ""),
                "timeframe": tf,
                "direction": setup["side"],
                "entry": price,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "ev": ev,
                "fee": fee_bps / 10_000.0,
                "slippage": slippage_bps / 10_000.0,
                "p_hit": p_hit,
                "detector": "breakout_or_momo",
                "trend": last["trend"],
                "model": "lgbm_or_baseline",
                "meta": {
                    "atr": setup["atr"],
                    "adx": float(last["adx"]),
                    "roc": float(last["roc"]),
                },
            }
            out.append(signal)
    # 7) optional persist signals
    if persist_signals and out:
        dtos = [SignalDTO(**s) for s in out]
        try:
            insert_signals(dtos)
        except Exception:
            pass
    return out
