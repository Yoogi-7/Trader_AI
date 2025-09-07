"""Market data persistence (OHLCV) with coverage and resume helpers — defensive version."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Iterable
from sqlalchemy import text
import pandas as pd

from app.storage.db import ENGINE, init_db
from app.data.exchange import iter_ohlcv_1m, resample, normalize_symbol

def ensure_market_tables() -> None:
    init_db()
    with ENGINE.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (ts, symbol, timeframe)
            )
            """
        ))
        conn.execute(text(
            """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
            ON ohlcv (symbol, timeframe, ts)
            """
        ))

def upsert_ohlcv(rows: List[Dict[str, Any]]) -> int:
    ensure_market_tables()
    if not rows:
        return 0
    sql = text(
        """
        INSERT INTO ohlcv (ts, symbol, timeframe, open, high, low, close, volume)
        VALUES (:ts, :symbol, :timeframe, :open, :high, :low, :close, :volume)
        ON CONFLICT(ts, symbol, timeframe) DO UPDATE SET
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume
        """
    )
    with ENGINE.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def upsert_ohlcv_df(df: pd.DataFrame, symbol: str, timeframe: str, batch_size: int = 5000) -> int:
    if df is None or df.empty:
        return 0
    total = 0
    recs = df.to_dict(orient="records")
    for i in range(0, len(recs), batch_size):
        chunk = recs[i:i+batch_size]
        for r in chunk:
            r["symbol"] = symbol
            r["timeframe"] = timeframe
        total += upsert_ohlcv(chunk)
    return total

def get_last_ts(symbol: str, timeframe: str) -> Optional[int]:
    ensure_market_tables()
    with ENGINE.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT ts FROM ohlcv
                WHERE symbol = :symbol AND timeframe = :timeframe
                ORDER BY ts DESC
                LIMIT 1
                """
            ),
            {"symbol": symbol, "timeframe": timeframe},
        ).fetchone()
        return int(row[0]) if row else None

def get_coverage(symbol: Optional[str] = None, timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return coverage per (symbol, timeframe): min_ts, max_ts, count. Never raises if table missing."""
    ensure_market_tables()
    clauses = ["1=1"]
    params: Dict[str, Any] = {}
    if symbol:
        clauses.append("symbol = :symbol"); params["symbol"] = symbol
    if timeframe:
        clauses.append("timeframe = :timeframe"); params["timeframe"] = timeframe
    sql = f"""
        SELECT symbol, timeframe, MIN(ts) AS min_ts, MAX(ts) AS max_ts, COUNT(*) AS cnt
        FROM ohlcv
        WHERE {' AND '.join(clauses)}
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """
    with ENGINE.connect() as conn:
        res = conn.execute(text(sql), params)
        rows = [dict(r) for r in res]
    # If empty, return empty list (dashboard will show 'No rows yet')
    return rows

def resume_backfill(
    symbols: Iterable[str],
    tfs: Iterable[str],
    days: int = 730,
    limit: int = 1000,
    sleep_ms: int = 150,
) -> Dict[str, Any]:
    import time
    ensure_market_tables()
    now_ms = int(time.time() * 1000)
    since_ms_default = now_ms - days * 24 * 60 * 60 * 1000

    total_1m, total_tf = 0, 0
    details: List[Dict[str, Any]] = []

    def tf_rule(tf: str) -> str:
        return tf.strip().lower().replace("m", "min") if tf.endswith("m") else tf.strip().lower()

    for raw in symbols:
        sym = normalize_symbol(raw)
        # 1) 1m streaming backfill (resume)
        last_1m = get_last_ts(sym, "1m")
        start_ms = max(since_ms_default, (last_1m + 60) * 1000) if last_1m else since_ms_default

        import pandas as pd
        from app.data.exchange import iter_ohlcv_1m, resample  # local import to avoid circulars

        frames = []
        for df_chunk in iter_ohlcv_1m(sym, since_ms=start_ms, until_ms=None, limit=limit, sleep_ms=sleep_ms):
            if df_chunk.empty:
                continue
            n = upsert_ohlcv_df(df_chunk, symbol=sym, timeframe="1m")
            total_1m += n
            frames.append(df_chunk)

        # Build in-memory 1m window for TFs
        if frames:
            df_1m = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
        else:
            df_1m = pd.DataFrame(columns=["ts","open","high","low","close","volume"])  # no new data

        for tf in tfs:
            if tf == "1m":
                continue
            # Pull small window to ensure TF completion
            last_tf = get_last_ts(sym, tf)
            if last_tf and not df_1m.empty:
                since_cut = last_tf - 5 * 24 * 3600  # 5-day buffer
                df_win = df_1m[df_1m["ts"] >= since_cut].copy()
            else:
                df_win = df_1m.copy()
            if df_win.empty:
                continue
            df_res = resample(df_win, tf_rule(tf))
            if df_res.empty:
                continue
            n_tf = upsert_ohlcv_df(df_res, symbol=sym, timeframe=tf)
            total_tf += n_tf
            details.append({"symbol": sym, "tf": tf, "rows": int(n_tf)})

    return {"inserted_1m": int(total_1m), "inserted_tf": int(total_tf), "details": details}
