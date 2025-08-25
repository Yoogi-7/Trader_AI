import sqlite3
import pandas as pd
import yaml
from core.performance import metrics_from_signals, equity_curve

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_signals(conn, exchange, symbol=None, timeframe=None):
    q = """
    SELECT ts_ms, closed_ts_ms, symbol, timeframe, status, pnl_usd, pnl_pct
    FROM signals
    WHERE exchange=?
    """
    params = [exchange]
    if symbol:
        q += " AND symbol=?"; params.append(symbol)
    if timeframe:
        q += " AND timeframe=?"; params.append(timeframe)
    q += " ORDER BY ts_ms ASC"
    df = pd.read_sql_query(q, conn, params=params)
    return df

def main():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    exchange = cfg["exchange"]["id"]

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        df = load_signals(conn, exchange)
    finally:
        conn.close()

    print(f"Loaded {len(df)} signals.")

    m = metrics_from_signals(df)
    print("--- METRYKI (całość) ---")
    for k, v in m.items():
        print(f"{k}: {v}")

    curve = equity_curve(df)
    out_path = f"{cfg['app']['data_dir']}/equity_curve.csv"
    curve.to_csv(out_path, index=False)
    print(f"Equity curve -> {out_path}")

if __name__ == "__main__":
    main()
