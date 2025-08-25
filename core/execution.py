from __future__ import annotations
import sqlite3
from typing import List, Tuple, Optional
import pandas as pd

def timeframe_to_ms(tf: str) -> int:
    m = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000
    }
    if tf not in m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return m[tf]

def _read_signals_pending(conn: sqlite3.Connection, exchange: str) -> pd.DataFrame:
    q = """
    SELECT id, ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, tp2, status
    FROM signals
    WHERE exchange=? AND status='PENDING'
    ORDER BY ts_ms ASC
    """
    df = pd.read_sql_query(q, conn, params=(exchange,))
    return df

def _read_candles_since(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str, ts_from: int, limit: int = 2000) -> pd.DataFrame:
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe=? AND ts_ms>=?
    ORDER BY ts_ms ASC
    LIMIT ?
    """
    df = pd.read_sql_query(q, conn, params=(exchange, symbol, timeframe, ts_from, limit))
    return df

def _first_touch_index(df: pd.DataFrame, price: float) -> Optional[int]:
    for i, row in df.iterrows():
        if row["low"] <= price <= row["high"]:
            return i
    return None

def _decide_exit_on_bar(row, direction: str, tp: float, sl: float, bar_policy: str) -> Optional[str]:
    # Zwraca "TP" / "SL" / None (brak wybicia)
    hit_tp = row["high"] >= tp if direction == "long" else row["low"] <= tp
    hit_sl = row["low"] <= sl   if direction == "long" else row["high"] >= sl
    if hit_tp and hit_sl:
        return "SL" if bar_policy == "conservative" else "TP"
    if hit_tp:
        return "TP"
    if hit_sl:
        return "SL"
    return None

def _pnl_pct(entry: float, exit_price: float, direction: str) -> float:
    if direction == "long":
        return (exit_price / entry - 1.0) * 100.0
    else:
        return (entry / exit_price - 1.0) * 100.0

def simulate_and_update(conn: sqlite3.Connection, cfg: dict):
    exchange = cfg["exchange"]["id"]
    validity = int(cfg["signals"]["validity_candles"])
    bar_policy = cfg["signals"].get("bar_policy", "conservative")
    usd_per_trade = float(cfg["signals"]["evaluation"]["usd_per_trade"])

    sigs = _read_signals_pending(conn, exchange)
    if sigs.empty:
        return

    for _, s in sigs.iterrows():
        symbol = s["symbol"]; tf = s["timeframe"]
        direction = s["direction"]; entry = float(s["entry"])
        sl = float(s["sl"]); tp = float(s["tp1"])
        ts_from = int(s["ts_ms"])

        # bierz świece od chwili sygnału (włącznie z kolejną)
        df = _read_candles_since(conn, exchange, symbol, tf, ts_from)
        if df.empty or len(df) < 2:
            continue

        # Szukamy pierwszego dotknięcia ENTRY po sygnale
        entry_idx = _first_touch_index(df.iloc[1:], entry)  # od kolejnej świecy
        if entry_idx is None or entry_idx > validity:
            # nie weszło w oknie ważności
            conn.execute(
                "UPDATE signals SET status='EXPIRED', opened_ts_ms=NULL, closed_ts_ms=?, exit_price=NULL, pnl_usd=0, pnl_pct=0 WHERE id=?",
                (int(df["ts_ms"].iloc[min(validity, len(df)-1)]), int(s["id"]))
            )
            continue

        # Po wejściu sprawdzamy kolejne świece (włącznie z tą samą)
        sub = df.iloc[entry_idx:]  # od świecy entry
        exit_status = None; exit_ts = None; exit_price = None
        for i, row in sub.iterrows():
            dec = _decide_exit_on_bar(row, direction, tp, sl, bar_policy)
            if dec is not None:
                exit_status = dec
                exit_ts = int(row["ts_ms"])
                if dec == "TP":
                    exit_price = tp
                else:
                    exit_price = sl
                break

        if exit_status is None:
            # w oknie ważności nie padł ani TP ani SL → traktujemy jako EXPIRED (bez PnL)
            last_i = min(entry_idx + validity, len(df) - 1)
            conn.execute(
                "UPDATE signals SET status='EXPIRED', opened_ts_ms=?, closed_ts_ms=?, exit_price=NULL, pnl_usd=0, pnl_pct=0 WHERE id=?",
                (int(df["ts_ms"].iloc[entry_idx]), int(df["ts_ms"].iloc[last_i]), int(s["id"]))
            )
            continue

        # policz PnL w trybie $100 per trade
        pnl_pct = _pnl_pct(entry, exit_price, direction)
        pnl_usd = usd_per_trade * pnl_pct / 100.0

        conn.execute("""
            UPDATE signals
            SET status=?, opened_ts_ms=?, closed_ts_ms=?, exit_price=?, pnl_usd=?, pnl_pct=?
            WHERE id=?
        """, (exit_status, int(df["ts_ms"].iloc[entry_idx]), exit_ts, float(exit_price), float(pnl_usd), float(pnl_pct), int(s["id"])))
