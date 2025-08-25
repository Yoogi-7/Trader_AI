import yaml, sqlite3, ccxt
from apscheduler.schedulers.blocking import BlockingScheduler
from contextlib import contextmanager
from download_data import (
    DDL_OHLCV, DDL_CHECKPOINT, do_incremental
)
from core.signals import ensure_signals_schema
from scan_signals import scan_once as scan_signals_once

@contextmanager
def db_conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def ensure_schema(conn):
    conn.execute(DDL_OHLCV)
    conn.execute(DDL_CHECKPOINT)
    ensure_signals_schema(conn)

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_exchange(exchange_id: str, rate_limit_ms: int):
    cls = getattr(ccxt, exchange_id)
    ex = cls({"enableRateLimit": True, "rateLimit": rate_limit_ms})
    ex.load_markets()
    return ex

def job_incremental():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    with db_conn(db_path) as conn:
        ensure_schema(conn)
        ex = make_exchange(cfg["exchange"]["id"], cfg["exchange"]["rate_limit_ms"])
        symbols = cfg["symbols"]
        tfs = cfg["timeframes"]
        batch_limit = cfg["backfill"]["batch_limit"]
        overlap = int(cfg["incremental"]["overlap_candles"])
        for symbol in symbols:
            for tf in tfs:
                try:
                    do_incremental(conn, ex, cfg["exchange"]["id"], symbol, tf, overlap_candles=overlap, batch_limit=batch_limit)
                except Exception as e:
                    print(f"[INCR ERR] {symbol} {tf}: {e}")

def job_scan_signals():
    try:
        scan_signals_once()
    except Exception as e:
        print(f"[SCAN ERR] {e}")

if __name__ == "__main__":
    cfg = load_config()
    incr_every = int(cfg["incremental"]["run_seconds"])
    scan_every = int(cfg["signals"]["scan_seconds"])
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(job_incremental, "interval", seconds=incr_every, id="incremental_sync", max_instances=1, coalesce=True)
    sched.add_job(job_scan_signals, "interval", seconds=scan_every, id="scan_signals", max_instances=1, coalesce=True)
    print(f"Scheduler started. Incr: {incr_every}s, Scan: {scan_every}s")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
