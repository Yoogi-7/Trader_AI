# dashboard.py
# Streamlit dashboard: Backtest + ML + Predykcje

from __future__ import annotations

import io
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.execution import ExecConfig, backtest_trades
from core.backtest import WFConfig, walk_forward_backtest, equity_curve, metrics
from core.features import make_features
from core.labeling import TripleBarrierConfig, triple_barrier_labels
from models.ml import time_series_fit_predict_proba

st.set_page_config(
    page_title="Trader AI — Backtest & ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helpers: Binance fetch + resample ----------
@st.cache_data(show_spinner=True, ttl=60 * 5)
def fetch_binance_klines(symbol: str, interval_code: str, start_str: str, end_str: str | None) -> pd.DataFrame:
    try:
        from binance.client import Client
    except Exception as e:
        raise RuntimeError("Brak pakietu 'python-binance'. Zainstaluj: python -m pip install python-binance") from e

    client = Client()  # publiczne
    klines = client.get_historical_klines(symbol, interval_code, start_str, end_str)
    if not klines:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["timestamp","open","high","low","close","volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.astype({"open":"float","high":"float","low":"float","close":"float","volume":"float"})
    return df.sort_values("timestamp").reset_index(drop=True)

def resample_to(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if df.empty or not rule:
        return df.copy()
    g = df.set_index("timestamp")
    out = pd.DataFrame({
        "open": g["open"].resample(rule).first(),
        "high": g["high"].resample(rule).max(),
        "low": g["low"].resample(rule).min(),
        "close": g["close"].resample(rule).last(),
        "volume": g["volume"].resample(rule).sum(),
    }).dropna().reset_index()
    return out

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Dane")
symbol = st.sidebar.text_input("Symbol (Binance)", "BTCUSDT")
interval_label = st.sidebar.selectbox("Interwał", ["1m","5m","10m (agregacja)","15m","30m","1h"], index=2)
days = st.sidebar.slider("Zakres (dni wstecz)", 1, 365, 60, 1)

if interval_label == "10m (agregacja)":
    fetch_interval, resample_rule = "1m", "10T"
else:
    fetch_interval, resample_rule = interval_label, None

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=days)
start_arg = start_dt.strftime("%d %b, %Y %H:%M:%S")
end_arg = None

st.sidebar.header("🎯 Triple-Barrier (etykiety dla ML)")
horizon_bars = st.sidebar.number_input("Horyzont (bary)", 5, 1000, 60, 5)
tp_mult = st.sidebar.number_input("TP * ATR", 0.1, 10.0, 2.0, 0.1)
sl_mult = st.sidebar.number_input("SL * ATR", 0.1, 10.0, 1.0, 0.1)

st.sidebar.header("🛠 Egzekucja")
fee_bp = st.sidebar.number_input("Prowizja (bp/strona)", 0.0, 50.0, 1.0, 0.1)
slip_ticks = st.sidebar.number_input("Slippage (ticki)", 0, 20, 1, 1)
tick_size = st.sidebar.number_input("Tick size", 0.0001, 100.0, 0.1, step=0.0001, format="%.4f")
latency = st.sidebar.number_input("Latency (bary)", 0, 5, 1, 1)
capital_ref = st.sidebar.number_input("Kapitał referencyjny [$]", 10.0, 100000.0, 100.0, 10.0)
risk_pct = st.sidebar.number_input("Ryzyko na trade [%]", 0.1, 10.0, 1.0, 0.1) / 100.0

st.sidebar.header("🧪 Walk-forward")
min_train_bars = st.sidebar.number_input("Min. train (bary)", 100, 200000, 5000, 100)
step_bars = st.sidebar.number_input("Krok testu (bary)", 50, 50000, 1000, 50)

st.sidebar.header("🤖 ML")
decision_thr = st.sidebar.slider("Próg decyzji p(win)", 0.50, 0.80, 0.55, 0.01)

col1, col2 = st.sidebar.columns(2)
fetch_btn = col1.button("📥 Pobierz")
run_btn = col2.button("▶️ Backtest")
ml_btn = st.sidebar.button("🤖 Trenuj ML + Predykcja")

# ---------- State ----------
if "data_df" not in st.session_state:
    st.session_state["data_df"] = pd.DataFrame()
if "ml_info" not in st.session_state:
    st.session_state["ml_info"] = None
if "ml_proba" not in st.session_state:
    st.session_state["ml_proba"] = None

st.title("🤖 Trader AI — Backtest + ML")

# ---------- Fetch ----------
if fetch_btn:
    with st.spinner(f"Pobieram {symbol} ({fetch_interval})…"):
        try:
            df_raw = fetch_binance_klines(symbol, fetch_interval, start_arg, end_arg)
            df = resample_to(df_raw, resample_rule)
            st.session_state["data_df"] = df
            st.success(f"Pobrano {len(df)} świec ({interval_label}).")
        except Exception as e:
            st.exception(e)

df = st.session_state["data_df"]

tabs = st.tabs(["🧱 Dane", "🧪 Backtest (reguły)", "🤖 ML (training)", "🔔 Predykcje ML"])

# ---------- TAB: Dane ----------
with tabs[0]:
    st.subheader("Podgląd danych")
    if df.empty:
        st.info("Brak danych – pobierz je z panelu bocznego.")
    else:
        st.dataframe(df.head(200))

# ---------- TAB: Backtest regułowy (demo) ----------
def demo_signal_fn_factory(tp_mult: float, sl_mult: float, horizon_bars: int):
    def signal_fn(_df: pd.DataFrame, te_slice: slice) -> pd.DataFrame:
        slc = _df.iloc[te_slice]
        close = slc["close"].astype(float)
        sma20 = close.rolling(20, min_periods=20).mean()
        high = slc["high"].astype(float)
        low = slc["low"].astype(float)
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        atr = tr.rolling(14, min_periods=14).mean()

        cond = (close > sma20) & (atr > atr.median())
        idxs = slc.index[cond].tolist()
        if not idxs:
            return pd.DataFrame(columns=["idx","side","tp","sl","horizon_bars"])

        tp = (close.loc[idxs] + tp_mult * atr.loc[idxs]).values
        sl = (close.loc[idxs] - sl_mult * atr.loc[idxs]).values
        sig = pd.DataFrame({
            "idx": idxs,
            "side": "long",
            "tp": tp,
            "sl": sl,
            "horizon_bars": int(horizon_bars),
        })
        sig = sig[sig["idx"] < te_slice.stop - 1].reset_index(drop=True)
        return sig
    return signal_fn

with tabs[1]:
    if st.button("Uruchom backtest (demo reguły)"):
        if df.empty:
            st.warning("Najpierw pobierz dane.")
        else:
            exec_cfg = ExecConfig(
                latency_bar=int(latency), fee_bp=float(fee_bp),
                slippage_ticks=int(slip_ticks), tick_size=float(tick_size),
                contract_value=1.0, use_trailing=False, time_stop_bars=int(horizon_bars)
            )
            wf_cfg = WFConfig(min_train_bars=int(min_train_bars), step_bars=int(step_bars))
            signal_fn = demo_signal_fn_factory(tp_mult=float(tp_mult), sl_mult=float(sl_mult), horizon_bars=int(horizon_bars))

            with st.spinner("Liczenie…"):
                trades, wf_table = walk_forward_backtest(
                    df=df, signal_fn=signal_fn, exec_cfg=exec_cfg, wf_cfg=wf_cfg,
                    capital_ref=float(capital_ref), risk_pct=float(risk_pct),
                )
            m = metrics(trades)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", m["trades"])
            c2.metric("Winrate", f"{m['winrate']*100:.1f}%")
            c3.metric("PF", f"{m['profit_factor']:.2f}" if np.isfinite(m["profit_factor"]) else "∞")
            c4.metric("MaxDD", f"{m['max_dd']:.2f}")

            st.subheader("Equity curve")
            ec = equity_curve(trades)
            if ec.empty:
                st.info("Brak transakcji.")
            else:
                fig = plt.figure(figsize=(12, 4))
                plt.plot(ec["timestamp_exit"], ec["equity"])
                plt.xlabel("Czas"); plt.ylabel("Equity [$]"); plt.title("Krzywa kapitału (netto)")
                st.pyplot(fig)

            st.subheader("Walk-forward – okna")
            st.dataframe(wf_table)

            st.subheader("Trade Log")
            st.dataframe(pd.DataFrame(trades))

# ---------- TAB: ML training ----------
with tabs[2]:
    if st.button("Trenuj model (features + triple-barrier)"):
        if df.empty:
            st.warning("Najpierw pobierz dane.")
        else:
            with st.spinner("Buduję cechy i etykiety…"):
                feats = make_features(df)
                # triple-barrier na bazie ATR
                tb = triple_barrier_labels(df.assign(timestamp=df["timestamp"]), 
                                           cfg=TripleBarrierConfig(
                                               horizon_bars=int(horizon_bars),
                                               use_atr=True, atr_period=14,
                                               tp_mult=float(tp_mult), sl_mult=float(sl_mult),
                                               percent_mode=False, side="long"
                                           ))
                # label = 1 (tp), 0 (sl), ignorujemy -1 (horizon) dla prostoty
                y = tb["label"].replace({-1: np.nan})
                data = pd.concat([feats, y.rename("label")], axis=1).dropna()
                X = data.drop(columns=["label"])
                y = data["label"].astype(int)

            with st.spinner("Trenuję poprzez TimeSeriesSplit + kalibracja…"):
                proba_oof, info = time_series_fit_predict_proba(X, y, n_splits=5)

            st.session_state["ml_info"] = info
            st.session_state["ml_proba"] = pd.DataFrame({
                "timestamp": df.loc[data.index, "timestamp"].values,
                "proba": proba_oof
            }).reset_index(drop=True)

            st.success(f"Trening OK. AUC mean={info.auc_mean:.3f} (±{info.auc_std:.3f}), Brier={info.brier:.4f}")

            # wykres kalibracji
            from sklearn.calibration import calibration_curve
            p = proba_oof[~np.isnan(proba_oof)]
            y_valid = y.iloc[~np.isnan(proba_oof)]
            frac_pos, mean_pred = calibration_curve(y_valid, p, n_bins=10, strategy="quantile")

            fig = plt.figure(figsize=(5, 4))
            plt.plot(mean_pred, frac_pos, marker="o", label="Model")
            plt.plot([0,1], [0,1], linestyle="--", label="Ideal")
            plt.xlabel("Przewidywane p(win)"); plt.ylabel("Rzeczywisty odsetek wygranych")
            plt.title("Kalibracja"); plt.legend()
            st.pyplot(fig)

            st.subheader("Out-of-fold p(win)")
            st.line_chart(st.session_state["ml_proba"].set_index("timestamp"))

# ---------- TAB: ML predictions ----------
with tabs[3]:
    if st.button("Generuj sygnały z ML"):
        if df.empty or st.session_state["ml_info"] is None:
            st.warning("Najpierw pobierz dane i wytrenuj model (zakładka ML).")
        else:
            info = st.session_state["ml_info"]
            # generujemy cechy i przepuszczamy przez wytrenowany ostatni model
            feats = make_features(df).dropna()
            last_idx = feats.index
            p_all = info.model.predict_proba(feats)[:, 1]
            pred = pd.DataFrame({"timestamp": df.loc[last_idx, "timestamp"].values, "proba": p_all}, index=last_idx)

            # decyzje wg progu
            hits = pred[pred["proba"] >= float(decision_thr)]
            if hits.empty:
                st.info("Brak sygnałów powyżej progu.")
            else:
                # zbuduj DataFrame sygnałów do backtestu egzekucji
                # TP/SL z panelu, wejście na następnej świecy
                sig = pd.DataFrame({
                    "idx": hits.index.astype(int),
                    "side": "long",
                    "tp": (df["close"].iloc[hits.index] + float(tp_mult) * (df["high"] - df["low"]).rolling(14, min_periods=14).mean().iloc[hits.index]).values,
                    "sl": (df["close"].iloc[hits.index] - float(sl_mult) * (df["high"] - df["low"]).rolling(14, min_periods=14).mean().iloc[hits.index]).values,
                    "horizon_bars": int(horizon_bars),
                }).reset_index(drop=True)

                exec_cfg = ExecConfig(
                    latency_bar=int(latency), fee_bp=float(fee_bp),
                    slippage_ticks=int(slip_ticks), tick_size=float(tick_size),
                    contract_value=1.0, use_trailing=False, time_stop_bars=int(horizon_bars)
                )
                with st.spinner("Symuluję egzekucję sygnałów ML…"):
                    trades = backtest_trades(df, sig, exec_cfg, capital_ref=float(capital_ref), risk_pct=float(risk_pct))

                tdf = pd.DataFrame(trades)
                st.subheader(f"Sygnały ML (próg {decision_thr:.2f})")
                if tdf.empty:
                    st.info("Brak transakcji po egzekucji.")
                else:
                    tdf["success"] = tdf["net_pnl"] > 0
                    st.dataframe(tdf)

                    m = metrics(trades)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Trades", m["trades"])
                    c2.metric("Winrate", f"{m['winrate']*100:.1f}%")
                    c3.metric("PF", f"{m['profit_factor']:.2f}" if np.isfinite(m["profit_factor"]) else "∞")
                    c4.metric("MaxDD", f"{m['max_dd']:.2f}")

                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    csv_buf = io.StringIO(); tdf.to_csv(csv_buf, index=False)
                    st.download_button("⬇️ trades_ml.csv", csv_buf.getvalue(), file_name=f"trades_ml_{symbol}_{ts}.csv", mime="text/csv")
