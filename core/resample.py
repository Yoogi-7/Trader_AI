from __future__ import annotations
import pandas as pd
import sqlite3
from typing import List

def _rule(tf: str) -> str:
    # pandas: 'min' zamiast 'T'
    mapping = {
        "1m":  "1min",
        "3m":  "3min",
        "5m":  "5min",
        "15m": "15min",
        "30m": "30min",
        "1h":  "60min",
        "2h":  "120min",
        "4h":  "240min",
    }
    return mapping.get(tf, tf)

def _agg():
    return {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume":"sum",
    }

def _to_dt(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df.set_index("ts").sort_index()

def _read_1m(conn: sqlite3.Connection, exchange: str, symbol: str, lookback_minutes: int) -> pd.DataFrame:
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe='1m'
      AND ts_ms >= (SELECT COALESCE(MAX(ts_ms),0) FROM ohlcv WHERE exchange=? AND symbol=? AND timeframe='1m') - ?*60*1000
    ORDER BY ts_ms ASC
    """
    return pd.read_sql_query(q, conn, params=(exchange, symbol, exchange, symbol, lookback_minutes))

def _write_tf(conn: sqlite3.Connection, exchange: str, symbol: str, tf: str, df: pd.DataFrame):
    if df.empty:
        return
    rows = [
        (exchange, symbol, tf, int(ix.value//10**6), float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
        for ix, r in df.iterrows()
    ]
    conn.executemany("""
        INSERT OR REPLACE INTO ohlcv (exchange, symbol, timeframe, ts_ms, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)

def resample_symbols(conn: sqlite3.Connection, exchange: str, symbols: List[str], base_tf: str, out_tfs: List[str], lookback_minutes: int = 720):
    if base_tf != "1m":
        raise ValueError("Ten resampler zakłada base_tf='1m'")
    for symbol in symbols:
        df_1m = _read_1m(conn, exchange, symbol, lookback_minutes)
        if df_1m.empty:
            continue
        s = _to_dt(df_1m)
        agg = _agg()
        for tf in out_tfs:
            rule = _rule(tf)
            r = s[["open","high","low","close","volume"]].resample(rule, label="left", closed="left").agg(agg).dropna(how="any")
            if not r.empty:
                _write_tf(conn, exchange, symbol, tf, r)
