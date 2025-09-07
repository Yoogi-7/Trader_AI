import os
import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TRADER_AI — Dashboard", layout="wide")
st.title("TRADER_AI — Dashboard")

api_base = st.sidebar.text_input("API base URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8080"))
st.sidebar.markdown("**Endpoints used:** `/market/coverage`, `/ingest/binance/resume`")

st.header("📊 Database coverage")
col1, col2 = st.columns(2)
with col1:
    symbol_filter = st.text_input("Filter symbol (optional)", "")
with col2:
    tf_filter = st.text_input("Filter timeframe (optional)", "")

if st.button("Refresh coverage", type="primary"):
    try:
        params = {}
        if symbol_filter: params["symbol"] = symbol_filter
        if tf_filter: params["timeframe"] = tf_filter
        r = requests.get(f"{api_base}/market/coverage", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data:
            df = pd.DataFrame(data)
            if "min_ts" in df.columns:
                df["min_dt"] = pd.to_datetime(df["min_ts"], unit="s")
            if "max_ts" in df.columns:
                df["max_dt"] = pd.to_datetime(df["max_ts"], unit="s")
            st.dataframe(df[["symbol","timeframe","cnt","min_ts","max_ts","min_dt","max_dt"]], use_container_width=True)
        else:
            st.info("No rows in ohlcv table yet.")
    except Exception as e:
        st.error(f"Failed to load coverage: {e}")

st.header("⬇️ Resume backfill (2 years by default)")
c1, c2, c3 = st.columns(3)
symbols = c1.text_input("Symbols (comma-separated)", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
tfs = c2.text_input("TFs (comma-separated)", "1m,10m,15m,30m,1h,2h,4h")
days = c3.number_input("Days", min_value=1, max_value=3650, value=730, step=1)
c4, c5, c6 = st.columns(3)
limit = c4.number_input("Exchange limit", min_value=100, max_value=1500, value=1000, step=50)
sleep_ms = c5.number_input("Sleep ms", min_value=0, max_value=1000, value=150, step=10)
run_button = c6.button("Run resume backfill", type="primary")

if run_button:
    try:
        params = {
            "symbols": symbols,
            "tfs": tfs,
            "days": int(days),
            "limit": int(limit),
            "sleep_ms": int(sleep_ms),
        }
        with st.spinner("Backfilling... this can take a while."):
            r = requests.post(f"{api_base}/ingest/binance/resume", params=params, timeout=3600)
            r.raise_for_status()
            result = r.json()
        st.success("Done.")
        st.json(result)
    except Exception as e:
        st.error(f"Backfill error: {e}")

st.caption("Tip: keep the API running: `uvicorn app.api.main:app --reload --port 8080`")
