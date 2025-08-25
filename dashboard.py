import os
import sqlite3
import pandas as pd
import streamlit as st
import yaml

from core.performance import metrics_from_signals, equity_curve

st.set_page_config(page_title="Trader AI – Dashboard", layout="wide")

@st.cache_resource
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def table_exists(conn, name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None

@st.cache_resource
def get_conn_ro(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=2000;")
    return conn

def read_recent_candles(conn, exchange, symbol, timeframe, limit=300):
    if not table_exists(conn, "ohlcv"):
        return pd.DataFrame()
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe=?
    ORDER BY ts_ms DESC
    LIMIT ?
    """
    cur = conn.execute(q, (exchange, symbol, timeframe, limit))
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["ts_ms", "open","high","low","close","volume"])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.sort_values("ts")
    return df

def read_signals(conn, exchange, symbol=None, timeframe=None, limit=20000):
    if not table_exists(conn, "signals"):
        return pd.DataFrame()
    q = """
    SELECT id, ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, tp2,
           leverage, risk_pct, position_notional, confidence, rationale, status,
           opened_ts_ms, closed_ts_ms, exit_price, pnl_usd, pnl_pct, tp1_hit, exit_reason,
           ml_p, ml_model
    FROM signals
    {where}
    ORDER BY ts_ms ASC
    LIMIT ?
    """
    conds = ["exchange=?"]; params = [exchange]
    if symbol:    conds.append("symbol=?");    params.append(symbol)
    if timeframe: conds.append("timeframe=?"); params.append(timeframe)
    where = "WHERE " + " AND ".join(conds)
    cur = conn.execute(q.format(where=where), params + [limit])
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [
        "id","ts_ms","exchange","symbol","timeframe","direction","entry","sl","tp1","tp2",
        "leverage","risk_pct","position_notional","confidence","rationale","status",
        "opened_ts_ms","closed_ts_ms","exit_price","pnl_usd","pnl_pct","tp1_hit","exit_reason",
        "ml_p","ml_model"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["opened_ts"] = pd.to_datetime(df["opened_ts_ms"], unit="ms", utc=True)
    df["closed_ts"] = pd.to_datetime(df["closed_ts_ms"], unit="ms", utc=True)
    return df

def compute_threshold_table(df: pd.DataFrame):
    # liczymy coverage i winrate na zamkniętych tradach, per próg ml_p
    if df.empty or "ml_p" not in df.columns:
        return pd.DataFrame(columns=["threshold","coverage_pct","winrate_pct","n_closed"])

    closed = df[df["status"].isin(["TP","SL","TP1_TRAIL"])].copy()
    closed = closed.dropna(subset=["ml_p"])
    if closed.empty:
        return pd.DataFrame(columns=["threshold","coverage_pct","winrate_pct","n_closed"])

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    total = len(closed)
    rows = []
    for t in thresholds:
        subset = closed[closed["ml_p"] >= t]
        cov = (len(subset) / total * 100.0) if total else 0.0
        wins = subset["status"].isin(["TP","TP1_TRAIL"]).sum()
        closed_n = len(subset)
        winrate = (wins / closed_n * 100.0) if closed_n else 0.0
        rows.append({"threshold": t, "coverage_pct": cov, "winrate_pct": winrate, "n_closed": closed_n})
    return pd.DataFrame(rows)

def hist_ml_p(df: pd.DataFrame, bins=20):
    if df.empty or "ml_p" not in df.columns:
        return pd.DataFrame(columns=["bin","count"])
    s = df["ml_p"].dropna()
    if s.empty:
        return pd.DataFrame(columns=["bin","count"])
    import numpy as np
    counts, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    out = pd.DataFrame({"bin": centers, "count": counts})
    return out

def main():
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    conn = get_conn_ro(db_path)

    st.sidebar.header("Ustawienia")
    exchange = cfg["exchange"]["id"]
    symbol = st.sidebar.selectbox("Symbol", ["(wszystkie)"] + cfg["symbols"], index=0)
    timeframe = st.sidebar.selectbox("Timeframe", ["(wszystkie)"] + cfg["timeframes"], index=0)

    sym_filter = None if symbol == "(wszystkie)" else symbol
    tf_filter  = None if timeframe == "(wszystkie)" else timeframe

    st.title("📈 Trader AI – Dashboard")

    tab1, tab2 = st.tabs(["🎯 Sygnały & Wykres", "📊 Wyniki"])
    with tab1:
        col1, col2 = st.columns([2,1])
        with col1:
            sel_symbol = cfg["symbols"][0] if symbol == "(wszystkie)" else symbol
            sel_tf = cfg["timeframes"][1] if timeframe == "(wszystkie)" else timeframe
            st.subheader(f"Candles: {sel_symbol} – {sel_tf}")

            if not table_exists(conn, "ohlcv"):
                st.info(
                    "Baza nie zainicjalizowana.\n\n"
                    "1) `python init_db.py`\n"
                    "2) `python download_data.py`\n"
                    "3) `python scheduler_run.py`"
                )
            else:
                df = read_recent_candles(conn, exchange, sel_symbol, sel_tf, limit=500)
                if df.empty:
                    st.info("Brak danych OHLCV. Uruchom backfill i scheduler.")
                else:
                    st.line_chart(df.set_index("ts")[["close"]])
                    st.dataframe(df.tail(50), use_container_width=True)

        with col2:
            st.subheader("Sygnały")
            s = read_signals(conn, exchange, sym_filter, tf_filter, limit=300)
            if s.empty:
                st.write("Brak sygnałów.")
            else:
                view = s[[
                    "ts","symbol","timeframe","direction","entry","sl","tp1","tp2",
                    "leverage","confidence","status","exit_reason","tp1_hit","ml_p",
                    "opened_ts","closed_ts","exit_price","pnl_usd","pnl_pct"
                ]].copy()
                st.dataframe(view, use_container_width=True)

    with tab2:
        st.subheader("Metryki & Equity")
        df_all = read_signals(conn, exchange, sym_filter, tf_filter, limit=200000)
        m = metrics_from_signals(df_all)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", m["trades"])
        c2.metric("Winrate", f"{m['winrate']:.1f}%")
        c3.metric("PnL łącznie (USD)", f"{m['pnl_usd']:.2f}")
        c4.metric("Max DD (USD)", f"{m['max_dd_usd']:.2f}")

        curve = equity_curve(df_all)
        if not curve.empty:
            st.line_chart(curve.set_index("idx")[["equity"]])
            st.caption("Equity zsumowane po kolejnych transakcjach (tryb $100/trade).")
        else:
            st.info("Brak danych do wyświetlenia equity.")

        st.subheader("Coverage vs Winrate (wg progu `ml_p`)")
        tbl = compute_threshold_table(df_all)
        st.dataframe(tbl, use_container_width=True)

        st.subheader("Histogram `ml_p`")
        hist = hist_ml_p(df_all[df_all["status"].isin(["TP","SL","TP1_TRAIL"])], bins=20)
        if not hist.empty:
            st.bar_chart(hist.set_index("bin"))
        else:
            st.write("Brak danych `ml_p` lub brak zamkniętych transakcji.")

if __name__ == "__main__":
    main()
