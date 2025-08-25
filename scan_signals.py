import sqlite3
import yaml
from contextlib import contextmanager

from core.schema import ensure_base_schema, migrate_signals_schema
from core.signals import read_candles, generate_signal, insert_signal
from core.ml import load_meta_model, predict_pwin_from_df

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
        # jedyne miejsce gdzie dotykamy schematu
        ensure_base_schema(conn)
        migrate_signals_schema(conn)

        exch = cfg["exchange"]["id"]

        model, feat_names = (None, None)
        use_meta = bool(cfg.get("models", {}).get("meta", {}).get("enabled", False))
        threshold = float(cfg.get("models", {}).get("meta", {}).get("threshold", 0.6))
        model_path = cfg.get("models", {}).get("meta", {}).get("model_path", "models/meta_xgb.pkl")
        if use_meta:
            model, feat_names = load_meta_model(model_path)

        for sym in cfg["symbols"]:
            for tf in cfg["timeframes"]:
                try:
                    df = read_candles(conn, exch, sym, tf, limit=max(600, cfg["signals"]["lookback_candles"] + 50))
                    sig = generate_signal(df, cfg)
                    if not sig:
                        continue

                    status_override = None
                    if use_meta and model is not None and feat_names is not None:
                        p = predict_pwin_from_df(df, sig, cfg, model, feat_names)
                        sig["ml_p"] = float(p)
                        sig["ml_model"] = "xgb_v1"
                        if p < threshold:
                            status_override = "FILTERED"
                            print(f"[FILTER] {sym} {tf} {sig['direction']} p={p:.2f} < {threshold}")
                        else:
                            print(f"[PASS]   {sym} {tf} {sig['direction']} p={p:.2f} ≥ {threshold}")

                    insert_signal(conn, exch, sym, tf, sig, status_override=status_override)
                except Exception as e:
                    print(f"[ERR] {sym} {tf}: {e}")

if __name__ == "__main__":
    scan_once()
