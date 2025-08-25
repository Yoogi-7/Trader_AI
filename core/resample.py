from __future__ import annotations
import sqlite3
from typing import List, Iterable
import pandas as pd
import numpy as np

# Mapowanie TF -> reguła pandas
TF_RULE = {
    "1m": "1T",
    "3m": "3T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}

def _tf_ms(tf: str) -> int:
    m = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    if tf not in m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return m[tf]

def _read_last_ts(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str) -> int | None:
    cur = conn.execute("""
        SELECT ts_ms FROM ohlcv
        WHERE exchange=? AND symbol=? AND timeframe=?
        ORDER BY ts_ms DESC LIMIT 1
    """, (exchange, symbol, timeframe))
    row = cur.fetchone()
    return int(row[0]) if row else None

def _read_base_since(conn: sqlite3.Connection, exchange: str, symbol: str, base_tf: str, since_ms: int) -> pd.DataFrame:
    cur = conn.execute("""
        SELECT ts_ms, open, high, low, close, volume
        FROM ohlcv
        WHERE exchange=? AND symbol=? AND timeframe=? AND ts_ms >= ?
        ORDER BY ts_ms ASC
    """, (exchange, symbol, base_tf, since_ms))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts_ms","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("ts")
    return df

def _upsert_rows(conn: sqlite3.Connection, rows: list[tuple]):
    # rows: (exchange, symbol, timeframe, ts_ms, o,h,l,c,v)
    conn.executemany("""
        INSERT OR REPLACE INTO ohlcv
        (exchange, symbol, timeframe, ts_ms, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

def _resample_df(df_1m: pd.DataFrame, out_tf: str) -> pd.DataFrame:
    rule = TF_RULE[out_tf]
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    r = df_1m[["open","high","low","close","volume"]].resample(rule, label="left", closed="left").agg(agg).dropna()
    r["ts_ms"] = (r.index.view("int64") // 10**6).astype("int64")
    return r

def resample_symbols(
    conn: sqlite3.Connection,
    exchange: str,
    symbols: Iterable[str],
    base_tf: str,
    out_tfs: Iterable[str],
    lookback_minutes: int = 720,  # ile minut wstecz brać 1m do przeliczeń (domyślnie 12h)
):
    if base_tf not in TF_RULE:
        raise ValueError(f"Base TF not supported: {base_tf}")
    for tf in out_tfs:
        if tf not in TF_RULE:
            raise ValueError(f"Out TF not supported: {tf}")

    # policz od kiedy wczytać 1m (wystarczy okno + 1 out-bar)
    import time
    now_ms = int(time.time() * 1000)
    base_tf_ms = _tf_ms(base_tf)
    since_base_default = now_ms - lookback_minutes * 60_000

    for symbol in symbols:
        # Wczytaj bazę 1m
        df_base = _read_base_since(conn, exchange, symbol, base_tf, since_base_default)
        if df_base.empty:
            continue

        for out_tf in out_tfs:
            last_ts_target = _read_last_ts(conn, exchange, symbol, out_tf)
            # Jeśli mamy już jakieś bary docelowe, zacznij od kolejnego
            if last_ts_target is not None:
                need_from_ms = last_ts_target + _tf_ms(out_tf)
                # musimy mieć 1m >= need_from_ms
                df_in = df_base[df_base["ts_ms"] >= need_from_ms] if "ts_ms" in df_base.columns else df_base[df_base.index >= pd.to_datetime(need_from_ms, unit="ms", utc=True)]
                if df_in.empty:
                    continue
                df_1m = df_in
            else:
                df_1m = df_base

            # Gdy _read_base_since użył indeksu czasu, zapewnij kolumnę ts_ms
            if "ts_ms" not in df_1m.columns:
                df_1m = df_1m.copy()
                df_1m["ts_ms"] = (df_1m.index.view("int64") // 10**6).astype("int64")

            r = _resample_df(df_1m, out_tf)
            if r.empty:
                continue

            rows = []
            for ts_ms, row in r.iterrows():
                rows.append((
                    exchange, symbol, out_tf, int(row["ts_ms"]),
                    float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"])
                ))
            if rows:
                _upsert_rows(conn, rows)
