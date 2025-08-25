import yaml, sqlite3, ccxt
from apscheduler.schedulers.blocking import BlockingScheduler
from contextlib import contextmanager
from download_data import DDL_OHLCV, DDL_CHECKPOINT, do_incremental
from core.signals import ensure_signals_schema
from scan_signals import scan_once as scan_signals_once
from core.resample import resample_symbols
from core.execution import simulate_and_update

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
        base_tf = cfg["data"]["base_timeframe"]
        batch_limit = cfg["backfill"]["batch_limit"]
        overlap = int(cfg["incremental"]["overlap_candles"])
        for symbol in symbols:
            try:
                do_incremental(conn, ex, cfg["exchange"]["id"], symbol, base_tf,
                               overlap_candles=overlap, batch_limit=batch_limit)
            except Exception as e:
                print(f"[INCR ERR] {symbol} {base_tf}: {e}")

def job_resample():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    with db_conn(db_path) as conn:
        exchange = cfg["exchange"]["id"]
        symbols = cfg["symbols"]
        base_tf = cfg["data"]["base_timeframe"]
        out_tfs = cfg["data"]["derived_timeframes"]
        lookback = int(cfg["data"]["resample"]["lookback_minutes"])
        try:
            resample_symbols(conn, exchange, symbols, base_tf, out_tfs, lookback_minutes=lookback)
        except Exception as e:
            print(f"[RESAMPLE ERR] {e}")

def job_scan_signals():
    try:
        scan_signals_once()
    except Exception as e:
        print(f"[SCAN ERR] {e}")

def job_execute():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    with db_conn(db_path) as conn:
        try:
            simulate_and_update(conn, cfg)
        except Exception as e:
            print(f"[EXEC ERR] {e}")

if __name__ == "__main__":
    cfg = load_config()
    incr_every = int(cfg["incremental"]["run_seconds"])
    scan_every = int(cfg["signals"]["scan_seconds"])
    resm_every = int(cfg["data"]["resample"]["run_seconds"])

    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(job_incremental, "interval", seconds=incr_every, id="incremental_sync", max_instances=1, coalesce=True)
    sched.add_job(job_resample,   "interval", seconds=resm_every, id="resample",         max_instances=1, coalesce=True)
    sched.add_job(job_scan_signals,"interval", seconds=scan_every, id="scan_signals",    max_instances=1, coalesce=True)
    # egzekucja co tyle samo co skan (możesz rozdzielić)
    sched.add_job(job_execute,    "interval", seconds=scan_every, id="execute_signals", max_instances=1, coalesce=True)

    print(f"Scheduler started. incr={incr_every}s, resample={resm_every}s, scan={scan_every}s, exec={scan_every}s")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
