import sqlite3
import pandas as pd
import streamlit as st
import yaml

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

def read_signals(conn, exchange, symbol=None, timeframe=None, limit=100):
    q = """
    SELECT ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, tp2,
           leverage, risk_pct, position_notional, confidence, rationale, status,
           opened_ts_ms, closed_ts_ms, exit_price, pnl_usd, pnl_pct
    FROM signals
    {where}
    ORDER BY ts_ms DESC
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
    df = pd.DataFrame(rows, columns=[
        "ts_ms","exchange","symbol","timeframe","direction","entry","sl","tp1","tp2",
        "leverage","risk_pct","position_notional","confidence","rationale","status",
        "opened_ts_ms","closed_ts_ms","exit_price","pnl_usd","pnl_pct"
    ])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["opened_ts"] = pd.to_datetime(df["opened_ts_ms"], unit="ms", utc=True)
    df["closed_ts"] = pd.to_datetime(df["closed_ts_ms"], unit="ms", utc=True)
    return df

def summary(conn, exchange):
    q = """
    SELECT
      COUNT(*)                              AS total,
      SUM(CASE WHEN status IN ('TP','SL') THEN 1 ELSE 0 END) AS closed,
      SUM(CASE WHEN status='TP' THEN 1 ELSE 0 END)           AS wins,
      COALESCE(SUM(CASE WHEN status IN ('TP','SL') THEN pnl_usd ELSE 0 END),0) AS pnl_usd,
      COALESCE(AVG(CASE WHEN status IN ('TP','SL') THEN pnl_pct END),0)        AS avg_pct
    FROM signals
    WHERE exchange=?
    """
    r = conn.execute(q, (exchange,)).fetchone()
    if not r: return {"total":0,"closed":0,"wins":0,"pnl_usd":0.0,"avg_pct":0.0}
    total, closed, wins, pnl_usd, avg_pct = r
    winrate = (wins/closed*100.0) if closed else 0.0
    return {"total":total, "closed":closed, "wins":wins, "winrate":winrate, "pnl_usd":pnl_usd, "avg_pct":avg_pct}

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
        tf_filter  = None if timeframe == "(wszystkie)" else timeframe
        s = read_signals(conn, exchange, sym_filter, tf_filter, limit=200)
        if s.empty:
            st.write("Brak sygnałów.")
        else:
            view = s[[
                "ts","symbol","timeframe","direction","entry","sl","tp1",
                "leverage","confidence","status","opened_ts","closed_ts","exit_price","pnl_usd","pnl_pct"
            ]].copy()
            st.dataframe(view, use_container_width=True)

        st.subheader("Wyniki (tryb $100/trade)")
        agg = summary(conn, exchange)
        m1, m2, m3 = st.columns(3)
        m1.metric("Sygnały łącznie", agg["total"])
        m2.metric("Winrate", f"{agg['winrate']:.1f}%")
        m3.metric("PnL łącznie (USD)", f"{agg['pnl_usd']:.2f}")
        st.caption(f"Śr. wynik na trade: {agg['avg_pct']:.3f}%  •  Uwaga: PnL liczony w trybie stałego nominału $100.")
        
if __name__ == "__main__":
    main()
