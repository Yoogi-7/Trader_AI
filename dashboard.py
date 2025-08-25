import sqlite3
import pandas as pd
import streamlit as st
import yaml
from datetime import datetime, timezone

st.set_page_config(page_title="Trader AI – Dashboard", layout="wide")

@st.cache_resource
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_resource
def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn

def read_recent_candles(conn, exchange, symbol, timeframe, limit=300):
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

def read_signals(conn, exchange, symbol=None, timeframe=None, limit=50):
    q = """
    SELECT ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, tp2, leverage, risk_pct, position_notional, confidence, rationale, status
    FROM signals
    {where}
    ORDER BY ts_ms DESC
    LIMIT ?
    """
    conds = ["exchange=?"]
    params = [exchange]
    if symbol:
        conds.append("symbol=?"); params.append(symbol)
    if timeframe:
        conds.append("timeframe=?"); params.append(timeframe)
    where = "WHERE " + " AND ".join(conds)
    cur = conn.execute(q.format(where=where), params + [limit])
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts_ms","exchange","symbol","timeframe","direction","entry","sl","tp1","tp2","leverage","risk_pct","position_notional","confidence","rationale","status"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df

def main():
    cfg = load_config()
    conn = get_conn(cfg["app"]["db_path"])

    st.sidebar.header("Ustawienia")
    exchange = cfg["exchange"]["id"]
    symbol = st.sidebar.selectbox("Symbol", ["(wszystkie)"] + cfg["symbols"], index=0)
    timeframe = st.sidebar.selectbox("Timeframe", ["(wszystkie)"] + cfg["timeframes"], index=0)

    st.title("📈 Trader AI – Dashboard")

    col1, col2 = st.columns([2,1])
    with col1:
        sel_symbol = cfg["symbols"][0] if symbol == "(wszystkie)" else symbol
        sel_tf = cfg["timeframes"][1] if timeframe == "(wszystkie)" else timeframe
        st.subheader(f"Candles: {sel_symbol} – {sel_tf}")
        df = read_recent_candles(conn, exchange, sel_symbol, sel_tf, limit=500)
        if df.empty:
            st.info("Brak danych. Uruchom backfill:\n`python download_data.py`\nPotem scheduler:\n`python scheduler_run.py`")
        else:
            st.line_chart(df.set_index("ts")[["close"]])
            st.dataframe(df.tail(50), use_container_width=True)

    with col2:
        st.subheader("Sygnały")
        sym_filter = None if symbol == "(wszystkie)" else symbol
        tf_filter = None if timeframe == "(wszystkie)" else timeframe
        s = read_signals(conn, exchange, sym_filter, tf_filter, limit=100)
        if s.empty:
            st.write("Brak sygnałów.")
        else:
            # krótki podgląd
            view = s[["ts","symbol","timeframe","direction","entry","sl","tp1","leverage","risk_pct","confidence","status"]].copy()
            st.dataframe(view, use_container_width=True)

if __name__ == "__main__":
    main()
