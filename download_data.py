import os
import time
import math
import yaml
import ccxt
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

# ---------- Helpers ----------
def timeframe_to_ms(tf: str) -> int:
    m = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
        "1d": 86_400_000
    }
    if tf not in m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return m[tf]

def now_ms() -> int:
    return int(time.time() * 1000)

def floor_ms(ts: int, tf_ms: int) -> int:
    return (ts // tf_ms) * tf_ms

def parse_start_date_iso(s: str | None) -> int | None:
    if not s:
        return None
    # YYYY-MM-DD jako UTC 00:00
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

@contextmanager
def db_conn(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

# ---------- DB schema ----------
DDL_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    exchange   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    ts_ms      INTEGER NOT NULL,   -- open time in ms
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe, ts_ms)
);
"""

DDL_CHECKPOINT = """
CREATE TABLE IF NOT EXISTS sync_checkpoint (
    exchange   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    last_ts_ms INTEGER NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe)
);
"""

def upsert_ohlcv_rows(conn, rows):
    conn.executemany("""
        INSERT OR REPLACE INTO ohlcv
        (exchange, symbol, timeframe, ts_ms, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

def read_checkpoint(conn, exchange_id, symbol, timeframe):
    cur = conn.execute("""
        SELECT last_ts_ms FROM sync_checkpoint
        WHERE exchange=? AND symbol=? AND timeframe=?
    """, (exchange_id, symbol, timeframe))
    row = cur.fetchone()
    return row["last_ts_ms"] if row else None

def write_checkpoint(conn, exchange_id, symbol, timeframe, last_ts_ms):
    conn.execute("""
        INSERT INTO sync_checkpoint (exchange, symbol, timeframe, last_ts_ms)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(exchange, symbol, timeframe)
        DO UPDATE SET last_ts_ms=excluded.last_ts_ms
    """, (exchange_id, symbol, timeframe, last_ts_ms))

# ---------- Exchange ----------
def make_exchange(exchange_id: str, rate_limit_ms: int):
    cls = getattr(ccxt, exchange_id)
    ex = cls({"enableRateLimit": True, "rateLimit": rate_limit_ms})
    ex.load_markets()
    return ex

def fetch_ohlcv(ex, symbol, timeframe, since_ms=None, limit=1000):
    return ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)

# ---------- Backfill & Incremental ----------
def do_backfill(conn, ex, exchange_id, symbol, timeframe, start_ms, batch_limit):
    tf_ms = timeframe_to_ms(timeframe)
    latest_closed = floor_ms(now_ms() - tf_ms, tf_ms)
    cursor = start_ms

    while cursor < latest_closed:
        candles = fetch_ohlcv(ex, symbol, timeframe, since_ms=cursor, limit=batch_limit)
        if not candles:
            break

        rows = []
        for ts, o, h, l, c, v in candles:
            ts_aligned = floor_ms(ts, tf_ms)
            if ts_aligned <= latest_closed:
                rows.append((exchange_id, symbol, timeframe, ts_aligned, o, h, l, c, v))

        if rows:
            upsert_ohlcv_rows(conn, rows)
            max_ts = max(r[3] for r in rows)
            write_checkpoint(conn, exchange_id, symbol, timeframe, max_ts)
            cursor = max_ts + tf_ms
        else:
            # jeśli pusto, przesuń się o jedno okno
            cursor += batch_limit * tf_ms

        time.sleep(0.2)  # mały jitter

def do_incremental(conn, ex, exchange_id, symbol, timeframe, overlap_candles=1, batch_limit=1000):
    tf_ms = timeframe_to_ms(timeframe)
    latest_closed = floor_ms(now_ms() - tf_ms, tf_ms)

    last_ts = read_checkpoint(conn, exchange_id, symbol, timeframe)
    if last_ts is None:
        # brak checkpointu → zaczynamy 2 dni wstecz (bezpieczny default)
        last_ts = latest_closed - 2 * 24 * 60 * 60 * 1000

    since = max(last_ts - overlap_candles * tf_ms, 0)

    while since < latest_closed:
        candles = fetch_ohlcv(ex, symbol, timeframe, since_ms=since, limit=batch_limit)
        if not candles:
            break

        rows = []
        for ts, o, h, l, c, v in candles:
            ts_aligned = floor_ms(ts, tf_ms)
            if ts_aligned <= latest_closed:
                rows.append((exchange_id, symbol, timeframe, ts_aligned, o, h, l, c, v))

        if rows:
            upsert_ohlcv_rows(conn, rows)
            max_ts = max(r[3] for r in rows)
            write_checkpoint(conn, exchange_id, symbol, timeframe, max_ts)
            since = max_ts + tf_ms
        else:
            break

        time.sleep(0.15)

# ---------- CLI ----------
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_schema(conn):
    conn.execute(DDL_OHLCV)
    conn.execute(DDL_CHECKPOINT)

def main():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    os.makedirs(cfg["app"]["data_dir"], exist_ok=True)

    with db_conn(db_path) as conn:
        ensure_schema(conn)

        ex = make_exchange(cfg["exchange"]["id"], cfg["exchange"]["rate_limit_ms"])
        symbols = cfg["symbols"]
        # Backfill robimy dla WSZYSTKICH TF z configu (w tym 1m i wyższych, jeśli chcesz historycznie)
        tfs = cfg.get("timeframes", ["1m"])

        batch_limit = cfg["backfill"]["batch_limit"]

        start_date = cfg["backfill"].get("start_date") or ""
        start_ms_from_date = parse_start_date_iso(start_date)
        if start_ms_from_date is not None:
            start_ms = start_ms_from_date
        else:
            start_days_ago = int(cfg["backfill"]["start_days_ago"])
            start_ms = now_ms() - start_days_ago * 24 * 60 * 60 * 1000

        for symbol in symbols:
            for tf in tfs:
                print(f"[BACKFILL] {symbol} {tf} from {start_date or (str(cfg['backfill']['start_days_ago']) + 'd ago')}")
                try:
                    do_backfill(conn, ex, cfg["exchange"]["id"], symbol, tf, start_ms, batch_limit)
                except Exception as e:
                    print(f"[BACKFILL ERR] {symbol} {tf}: {e}")

        print("Backfill done. Run scheduler for incrementals.")

if __name__ == "__main__":
    main()
