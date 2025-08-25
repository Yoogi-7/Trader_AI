import os
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from contextlib import contextmanager
import yaml

@contextmanager
def db_conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def export_symbol_tf(conn, out_root, exchange, symbol, timeframe, chunk_days=30):
    # Pobierz min/max
    row = conn.execute("""
        SELECT MIN(ts_ms) AS min_ts, MAX(ts_ms) AS max_ts
        FROM ohlcv WHERE exchange=? AND symbol=? AND timeframe=?
    """, (exchange, symbol, timeframe)).fetchone()
    if not row or row["min_ts"] is None:
        return 0
    min_ts, max_ts = int(row["min_ts"]), int(row["max_ts"])

    os.makedirs(out_root, exist_ok=True)
    # Partycje katalogowe
    base = os.path.join(
        out_root,
        f"exchange={exchange}",
        f"symbol={symbol.replace('/','_').replace(':','_')}",
        f"timeframe={timeframe}",
    )
    os.makedirs(base, exist_ok=True)

    # Pętla po dniach
    from datetime import datetime, timedelta, timezone
    start = datetime.fromtimestamp(min_ts/1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end   = datetime.fromtimestamp(max_ts/1000, tz=timezone.utc)

    written = 0
    cursor = start
    while cursor <= end:
        next_day = cursor + timedelta(days=1)
        ts_from = int(cursor.timestamp() * 1000)
        ts_to   = int(next_day.timestamp() * 1000)

        rows = conn.execute("""
            SELECT ts_ms, open, high, low, close, volume
            FROM ohlcv
            WHERE exchange=? AND symbol=? AND timeframe=? AND ts_ms>=? AND ts_ms<? 
            ORDER BY ts_ms ASC
        """, (exchange, symbol, timeframe, ts_from, ts_to)).fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=["ts_ms","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            table = pa.Table.from_pandas(df)
            out_dir = os.path.join(base, f"dt={cursor.strftime('%Y-%m-%d')}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "data.parquet")
            pq.write_table(table, out_path)
            written += len(df)

        cursor = next_day

    return written

def main():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    out_root = os.path.join(cfg["app"]["data_dir"], "parquet")

    with db_conn(db_path) as conn:
        exchange = cfg["exchange"]["id"]
        symbols = cfg["symbols"]
        tfs = cfg.get("timeframes", ["1m"])
        total = 0
        for sym in symbols:
            for tf in tfs:
                written = export_symbol_tf(conn, out_root, exchange, sym, tf)
                print(f"[EXPORT] {sym} {tf}: {written} rows")
                total += written
        print(f"Done. Total rows: {total}. Output: {out_root}")

if __name__ == "__main__":
    main()
