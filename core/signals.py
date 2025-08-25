from __future__ import annotations
import sqlite3, json, time
from typing import Optional, Dict, Any
import pandas as pd

from core.indicators import atr
from core.fibo import find_pivots, last_impulse_from_pivots, fib_retracements_and_extensions
from core.avwap import anchored_vwap
from core.risk import Fees, tp_net_pct, sizing_and_leverage

DDL_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,               -- znacznik czasu świecy bazowej sygnału
  exchange TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  direction TEXT NOT NULL,              -- 'long' / 'short'
  entry REAL NOT NULL,
  sl REAL NOT NULL,
  tp1 REAL NOT NULL,
  tp2 REAL NOT NULL,
  leverage REAL NOT NULL,
  risk_pct REAL NOT NULL,
  position_notional REAL NOT NULL,
  confidence REAL NOT NULL,
  rationale TEXT NOT NULL,              -- JSON array[str]
  status TEXT NOT NULL DEFAULT 'PENDING'
);
CREATE INDEX IF NOT EXISTS idx_signals_recent ON signals(exchange, symbol, timeframe, ts_ms DESC);
"""

def ensure_signals_schema(conn: sqlite3.Connection):
    for stmt in DDL_SIGNALS.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")

def read_candles(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe=?
    ORDER BY ts_ms DESC
    LIMIT ?
    """
    cur = conn.execute(q, (exchange, symbol, timeframe, limit))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts_ms","open","high","low","close","volume"])
    df = df.sort_values("ts_ms").reset_index(drop=True)
    return df

def _confidence_from_confluence(entry: float, avwap_now: float, df_atr: float) -> float:
    if pd.isna(avwap_now) or df_atr <= 0:
        return 0.5
    # im bliżej AVWAP (<=0.5 ATR) tym wyższa pewność
    dist_atr = abs(entry - avwap_now) / df_atr
    if dist_atr <= 0.5:
        return 0.8
    elif dist_atr <= 1.0:
        return 0.7
    else:
        return 0.6

def generate_signal(df: pd.DataFrame, cfg: dict) -> Optional[Dict[str, Any]]:
    if df.empty or len(df) < cfg["signals"]["lookback_candles"]:
        return None

    lookback = cfg["signals"]["lookback_candles"]
    window = cfg["signals"]["fibo"]["piv_window"]
    atr_period = cfg["signals"]["atr_period"]
    atr_mult_sl = cfg["signals"]["atr_mult_sl"]

    dfl = df.tail(lookback).copy()
    dfl["atr"] = atr(dfl, period=atr_period)
    pivots = find_pivots(dfl, window=window)
    if len(pivots) < 2:
        return None

    last_imp = last_impulse_from_pivots(pivots)
    if not last_imp:
        return None
    a, b = last_imp  # (idx, price, "H"/"L"), (idx, price, "H"/"L")
    p0_idx, p0_price, _ = a
    p1_idx, p1_price, _ = b
    up = p1_price > p0_price

    fibs = fib_retracements_and_extensions(p0_price, p1_price)
    retr = fibs["retr"]; ext = fibs["ext"]

    # Entry na 0.618 (trend); SL poniżej 0.786 (long) / powyżej 0.786 (short), z buforem ATR
    last_close = dfl["close"].iloc[-1]
    last_ts = int(dfl["ts_ms"].iloc[-1])

    atr_last = float(dfl["atr"].iloc[-1])
    if atr_last <= 0:
        return None

    if up:
        direction = "long"
        entry = retr["0.618"]
        sl_base = min(retr["0.786"], dfl["low"].iloc[p0_idx])
        sl = sl_base - atr_mult_sl * atr_last * 0.5
        tp1 = ext["1.272"]
        tp2 = ext["1.618"]
    else:
        direction = "short"
        entry = retr["0.618"]
        sl_base = max(retr["0.786"], dfl["high"].iloc[p0_idx])
        sl = sl_base + atr_mult_sl * atr_last * 0.5
        tp1 = ext["1.272"]
        tp2 = ext["1.618"]

    # AVWAP od punktu początku impulsu
    avwap_series = anchored_vwap(dfl, p0_idx)
    avwap_now = float(avwap_series.iloc[-1]) if not pd.isna(avwap_series.iloc[-1]) else float("nan")

    # Warunek min. 2% netto do TP1
    fees_cfg = cfg["signals"]["fees"]
    fees = Fees(taker_pct=fees_cfg["taker_pct"], maker_pct=fees_cfg["maker_pct"], assume_maker=fees_cfg["assume_maker"])
    tp1_net = tp_net_pct(entry, tp1, fees=fees, slippage_pct=cfg["signals"]["slippage_pct"])
    if tp1_net < cfg["signals"]["min_tp_net_pct"]:
        return None

    # Risk & sizing
    mode = cfg["risk"]["mode"]
    risk_pct = float(cfg["risk"]["risk_pct_by_mode"][mode])
    equity = 10000.0  # na start stała wartość; w przyszłości pobór z konta/ustawień
    position_notional, leverage = sizing_and_leverage(
        equity=equity, risk_pct=risk_pct, entry=entry, sl=sl,
        max_leverage=float(cfg["risk"]["max_leverage"]),
        liquidation_buffer=float(cfg["risk"]["liquidation_buffer"])
    )
    if leverage <= 0 or position_notional <= 0:
        return None

    confidence = _confidence_from_confluence(entry, avwap_now, atr_last)
    rationale = [
        f"Impulse {'up' if up else 'down'}: pivots {p0_idx}->{p1_idx}",
        "Fibo 0.618 entry / 0.786 SL / 1.272-1.618 TP",
        "AVWAP anchor @ start impulsu",
        f"ATR({atr_period})={atr_last:.4f}, TP1_net={tp1_net:.3f}%"
    ]

    return {
        "ts_ms": last_ts,
        "direction": direction,
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "leverage": float(leverage),
        "risk_pct": float(risk_pct),
        "position_notional": float(position_notional),
        "confidence": float(confidence),
        "rationale": rationale
    }

def insert_signal(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str, sig: Dict[str, Any]):
    conn.execute("""
        INSERT INTO signals
        (ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, tp2, leverage, risk_pct, position_notional, confidence, rationale, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
    """, (
        sig["ts_ms"], exchange, symbol, timeframe, sig["direction"], sig["entry"], sig["sl"], sig["tp1"], sig["tp2"],
        sig["leverage"], sig["risk_pct"], sig["position_notional"], sig["confidence"], json.dumps(sig["rationale"])
    ))
