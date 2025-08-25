import sqlite3, yaml, pandas as pd

def load_config():
    with open("config.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    db = cfg["app"]["db_path"]
    con = sqlite3.connect(db, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        # OHLCV 1m
        print("=== OHLCV (1m) per symbol ===")
        for sym in cfg["symbols"]:
            r = con.execute("SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM ohlcv WHERE exchange=? AND symbol=? AND timeframe='1m'",
                            (cfg["exchange"]["id"], sym)).fetchone()
            cnt, tmin, tmax = r
            print(f"{sym:16s}  rows={cnt:8d}  min_ts={tmin}  max_ts={tmax}")
        # Derived TFs
        print("\n=== OHLCV (derived) last ts per TF ===")
        for tf in cfg["timeframes"]:
            r = con.execute("SELECT COUNT(*) FROM ohlcv WHERE exchange=? AND timeframe=?",
                            (cfg["exchange"]["id"], tf)).fetchone()
            print(f"{tf:4s} rows={r[0]}")
        # Signals
        print("\n=== SIGNALS ===")
        r = con.execute("SELECT COUNT(*) FROM signals").fetchone()
        closed = con.execute("SELECT COUNT(*) FROM signals WHERE status IN ('TP','SL','TP1_TRAIL')").fetchone()[0]
        filt = con.execute("SELECT COUNT(*) FROM signals WHERE status='FILTERED'").fetchone()[0]
        print(f"total={r[0]}, closed={closed}, filtered={filt}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
