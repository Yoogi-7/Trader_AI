import sqlite3
import yaml
from contextlib import contextmanager
from core.signals import ensure_signals_schema, read_candles, generate_signal, insert_signal

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

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def scan_once():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    with db_conn(db_path) as conn:
        ensure_signals_schema(conn)
        exch = cfg["exchange"]["id"]
        for sym in cfg["symbols"]:
            for tf in cfg["timeframes"]:
                try:
                    df = read_candles(conn, exch, sym, tf, limit=max(600, cfg["signals"]["lookback_candles"] + 50))
                    sig = generate_signal(df, cfg)
                    if sig:
                        insert_signal(conn, exch, sym, tf, sig)
                        print(f"[SIGNAL] {sym} {tf} {sig['direction']} entry={sig['entry']:.4f} tp1_net≈ok")
                except Exception as e:
                    print(f"[ERR] {sym} {tf}: {e}")

if __name__ == "__main__":
    scan_once()
