import sqlite3
import pandas as pd
import streamlit as st
import yaml
from datetime import datetime

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
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert("UTC")
        df = df.sort_values("ts")
    return df

def main():
    cfg = load_config()
    conn = get_conn(cfg["app"]["db_path"])

    st.sidebar.header("Ustawienia")
    exchange = cfg["exchange"]["id"]
    symbol = st.sidebar.selectbox("Symbol", cfg["symbols"], index=0)
    timeframe = st.sidebar.selectbox("Timeframe", cfg["timeframes"], index=1)

    st.title("📈 Trader AI – Dashboard")
    st.caption("Podgląd danych (SQLite) + miejsce na sygnały AI.")

    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader(f"Candles: {symbol} – {timeframe}")
        df = read_recent_candles(conn, exchange, symbol, timeframe, limit=500)
        if df.empty:
            st.info("Brak danych. Uruchom backfill:  \n`python download_data.py`  \na następnie scheduler:  \n`python scheduler_run.py`")
        else:
            st.line_chart(df.set_index("ts")[["close"]])
            st.dataframe(df.tail(50), use_container_width=True)

    with col2:
        st.subheader("Sygnały (placeholder)")
        st.write("Tu wyświetlimy sygnały: entry/TP/SL/lewar/sizing + confidence.")
        st.write("W kroku 2 podłączymy detektory + meta-model.")

if __name__ == "__main__":
    main()
