import argparse
import sqlite3
import yaml
from contextlib import contextmanager
from datetime import datetime, timezone
import pandas as pd

from core.signals import ensure_signals_schema, read_candles, generate_signal, insert_signal
from core.ml import load_meta_model, predict_pwin_from_df
from core.execution import simulate_and_update

@contextmanager
def db_conn(db_path: str):
    conn = sqlite3.connect(db_path, timeout=120)
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

def dt_to_ms(s: str | None) -> int | None:
    if not s:
        return None
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def read_candles_range(conn, exchange, symbol, timeframe, ts_from=None, ts_to=None) -> pd.DataFrame:
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe=?
    """
    params = [exchange, symbol, timeframe]
    if ts_from is not None:
        q += " AND ts_ms>=?"
        params.append(int(ts_from))
    if ts_to is not None:
        q += " AND ts_ms<=?"
        params.append(int(ts_to))
    q += " ORDER BY ts_ms ASC"
    df = pd.read_sql_query(q, conn, params=params)
    return df

def retro_scan_symbol_tf(conn, cfg, symbol, timeframe, ts_from, ts_to, stride=1, limit_signals=None):
    exchange = cfg["exchange"]["id"]
    lookback = cfg["signals"]["lookback_candles"]

    use_meta   = bool(cfg.get("models", {}).get("meta", {}).get("enabled", False))
    threshold  = float(cfg.get("models", {}).get("meta", {}).get("threshold", 0.6))
    model_path = cfg.get("models", {}).get("meta", {}).get("model_path", "models/meta_xgb.pkl")
    model, feat_names = (None, None)
    if use_meta:
        model, feat_names = load_meta_model(model_path)

    df = read_candles_range(conn, exchange, symbol, timeframe, ts_from, ts_to)
    if df.empty or len(df) < (lookback + 5):
        return 0

    inserted = 0
    # przesuwamy okno po historii, co 'stride' świec
    start_idx = lookback
    end_idx = len(df) - 2  # zostaw min. jedną świecę po sygnale
    for i in range(start_idx, end_idx, stride):
        dfi = df.iloc[: i + 1].copy()  # dane "do teraz"
        sig = generate_signal(dfi, cfg)
        if not sig:
            continue
        # generujemy tylko, jeśli sygnał dotyczy ostatniej świecy tego wycinka
        if int(sig["ts_ms"]) != int(dfi["ts_ms"].iloc[-1]):
            continue

        status_override = None
        if use_meta and model is not None and feat_names is not None:
            p = predict_pwin_from_df(dfi, sig, cfg, model, feat_names)
            sig["ml_p"] = float(p); sig["ml_model"] = "xgb_v1"
            if p < threshold:
                status_override = "FILTERED"

        try:
            insert_signal(conn, exchange, symbol, timeframe, sig, status_override=status_override)
            inserted += 1
            if limit_signals and inserted >= limit_signals:
                break
        except Exception as e:
            print(f"[INSERT ERR] {symbol} {timeframe} i={i}: {e}")

    # po wstawieniu – od razu symulujemy egzekucję PENDING
    try:
        simulate_and_update(conn, cfg)
    except Exception as e:
        print(f"[EXEC ERR] {symbol} {timeframe}: {e}")

    return inserted

def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Retro-scan historycznych sygnałów")
    parser.add_argument("--start", type=str, default=cfg["backfill"].get("start_date", None), help="YYYY-MM-DD")
    parser.add_argument("--end",   type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--stride", type=int, default=1, help="co ile świec próbować (1=każda)")
    parser.add_argument("--limit-per-pair", type=int, default=None, help="limit wstawek sygnałów na parę/TF")
    parser.add_argument("--symbols", nargs="*", default=cfg["symbols"])
    parser.add_argument("--timeframes", nargs="*", default=cfg["timeframes"])
    args = parser.parse_args()

    ts_from = dt_to_ms(args.start)
    ts_to   = dt_to_ms(args.end)

    with db_conn(cfg["app"]["db_path"]) as conn:
        ensure_signals_schema(conn)
        total = 0
        for sym in args.symbols:
            for tf in args.timeframes:
                n = retro_scan_symbol_tf(conn, cfg, sym, tf, ts_from, ts_to, stride=args.stride, limit_signals=args.limit_per_pair)
                print(f"[RETRO] {sym} {tf}: +{n} sygnałów")
                total += n
        print(f"[RETRO] DONE: inserted={total}")

if __name__ == "__main__":
    main()
