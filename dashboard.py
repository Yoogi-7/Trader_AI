# dashboard.py
# Streamlit dashboard: Backtest + ML + Predykcje, PR-curve, expectancy (R) oraz CACHE danych.

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
from core.signals import TripleBarrierConfig, triple_barrier_labels
from core.utils import save_model_artifacts, load_model_artifacts
from core.cache import ensure_range_cached, slice_range, clear_cache
from models.ml import (
    time_series_fit_predict_proba,
    threshold_metrics,
    precision_recall_table,
    suggest_threshold_by_f1,
    expectancy_table,
    suggest_threshold_by_expectancy,
)

st.set_page_config(
    page_title="Trader AI — Backtest & ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Resampling helper ----------
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
st.sidebar.header("⚙️ Dane (z cache na dysku)")
symbol = st.sidebar.text_input("Symbol (Binance)", "BTCUSDT")
interval_label = st.sidebar.selectbox("Interwał", ["1m","5m","10m (agregacja)","15m","30m","1h"], index=2)
days = st.sidebar.slider("Zakres (dni wstecz)", 1, 365, 60, 1)

# Mapa interwału: dla "10m (agregacja)" pobieramy 1m do cache i agregujemy
if interval_label == "10m (agregacja)":
    fetch_interval, resample_rule = "1m", "10T"
else:
    fetch_interval, resample_rule = interval_label, None

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=days)

# cache actions
c1, c2 = st.sidebar.columns(2)
fetch_btn = c1.button("📥 Pobierz/odśwież")
clear_btn = c2.button("🗑️ Wyczyść cache (dla interwału)")

if clear_btn:
    ok, path = clear_cache(symbol, fetch_interval)
    if ok:
        st.sidebar.success(f"Usunięto cache: {path}")
    else:
        st.sidebar.info("Brak pliku cache do usunięcia.")

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

st.sidebar.header("🤖 ML")
decision_thr = st.sidebar.slider("Próg decyzji p(win)", 0.50, 0.90, 0.55, 0.01)
cost_R = st.sidebar.number_input("Koszt na trade [R] (fee+slip+latency)", 0.0, 1.0, 0.00, 0.01)
ml_btn = st.sidebar.button("🤖 Trenuj ML + Predykcja")

# ---------- State ----------
if "data_df" not in st.session_state:
    st.session_state["data_df"] = pd.DataFrame()
if "ml_info" not in st.session_state:
    st.session_state["ml_info"] = None
if "ml_proba" not in st.session_state:
    st.session_state["ml_proba"] = None
if "ml_oof_labels" not in st.session_state:
    st.session_state["ml_oof_labels"] = None
if "thr_suggested" not in st.session_state:
    st.session_state["thr_suggested"] = None
if "loaded_model" not in st.session_state:
    st.session_state["loaded_model"] = None
if "loaded_meta" not in st.session_state:
    st.session_state["loaded_meta"] = None
if "exp_thr_suggested" not in st.session_state:
    st.session_state["exp_thr_suggested"] = None

st.title("🤖 Trader AI — Backtest + ML (z cache)")

# ---------- FETCH (z CACHE) ----------
def fetch_with_cache(symbol: str, fetch_interval: str, start_dt: datetime, end_dt: datetime, resample_rule: str | None) -> pd.DataFrame:
    """
    1) ensure_range_cached() dociąga brakujące świece do cache dla fetch_interval,
    2) tniemy do wymaganego zakresu,
    3) opcjonalnie agregujemy (np. 1m -> 10m).
    """
    try:
        full_cache = ensure_range_cached(symbol, fetch_interval, start_dt, end_dt)
    except Exception as e:
        raise RuntimeError(f"Problem z pobieraniem/cachingiem ({symbol} {fetch_interval}): {e}") from e
    df = slice_range(full_cache, start_dt, end_dt)
    df = resample_to(df, resample_rule)
    return df

if fetch_btn:
    with st.spinner(f"Sprawdzam cache i dociągam brakujące dane ({symbol}, {fetch_interval})…"):
        df = fetch_with_cache(symbol, fetch_interval, start_dt, end_dt, resample_rule)
        st.session_state["data_df"] = df
        st.success(f"Dane gotowe: {len(df)} świec (cache + inkrementalne pobranie).")

df = st.session_state["data_df"]

tabs = st.tabs([
    "🧱 Dane",
    "🧪 Backtest (reguły)",
    "🤖 ML (training)",
    "📊 Trafność OOF + PR + Expectancy",
    "💾 Model: Zapis/Wczytaj",
    "🔔 Predykcje ML",
])

# ---------- TAB: Dane ----------
with tabs[0]:
    st.subheader("Podgląd danych")
    if df.empty:
        st.info("Brak danych – kliknij „Pobierz/odśwież”.")
    else:
        st.dataframe(df.head(200))

# ---------- TAB: Backtest regułowy (demo)
def demo_signal_fn_factory(tp_mult_v: float, sl_mult_v: float, horizon_v: int):
    def signal_fn(_df: pd.DataFrame, te_slice: slice) -> pd.DataFrame:
        slc = _df.iloc[te_slice]
        close = slc["close"].astype(float)
        high = slc["high"].astype(float)
        low = slc["low"].astype(float)
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        atr = tr.rolling(14, min_periods=14).mean()
        sma20 = close.rolling(20, min_periods=20).mean()
        cond = (close > sma20) & (atr > atr.median())
        idxs = slc.index[cond].tolist()
        if not idxs:
            return pd.DataFrame(columns=["idx","side","tp","sl","horizon_bars"])
        tp_vals = (close.loc[idxs] + tp_mult_v * atr.loc[idxs]).values
        sl_vals = (close.loc[idxs] - sl_mult_v * atr.loc[idxs]).values
        sig = pd.DataFrame({
            "idx": idxs,
            "side": "long",
            "tp": tp_vals,
            "sl": sl_vals,
            "horizon_bars": int(horizon_v),
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
            wf_cfg = WFConfig(min_train_bars=int(min_train_bars := 5000), step_bars=int(step_bars := 1000))
            signal_fn = demo_signal_fn_factory(tp_mult, sl_mult, horizon_bars)

            with st.spinner("Liczenie…"):
                trades, wf_table = walk_forward_backtest(
                    df=df, signal_fn=signal_fn, exec_cfg=exec_cfg, wf_cfg=wf_cfg,
                    capital_ref=float(capital_ref), risk_pct=float(risk_pct),
                )
            m = metrics(trades)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", m["trades"])
            c2.metric("Winrate", f"{m['winrate']*100:.1f}%")
            pf = m["profit_factor"]; c3.metric("PF", f"{pf:.2f}" if np.isfinite(pf) else "∞")
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
    if ml_btn:
        if df.empty:
            st.warning("Najpierw pobierz dane.")
        else:
            with st.spinner("Buduję cechy i etykiety…"):
                feats = make_features(df)
                tb = triple_barrier_labels(
                    df.assign(timestamp=df["timestamp"]),
                    cfg=TripleBarrierConfig(
                        horizon_bars=int(horizon_bars),
                        use_atr=True, atr_period=14,
                        tp_mult=float(tp_mult), sl_mult=float(sl_mult),
                        percent_mode=False, side="long"
                    )
                )
                y = tb["label"].replace({-1: np.nan})
                data = pd.concat([feats, y.rename("label")], axis=1).dropna()
                X = data.drop(columns=["label"])
                y_clean = data["label"].astype(int)

            with st.spinner("TimeSeriesSplit + kalibracja…"):
                proba_oof, info = time_series_fit_predict_proba(X, y_clean, n_splits=5)

            st.session_state["ml_info"] = info
            st.session_state["ml_proba"] = pd.DataFrame({
                "timestamp": df.loc[data.index, "timestamp"].values,
                "proba": proba_oof
            }, index=data.index).reset_index().rename(columns={"index":"row"})
            st.session_state["ml_oof_labels"] = pd.Series(y_clean.values, index=data.index)

            st.success(f"Trening OK. AUC mean={info.auc_mean:.3f} (±{info.auc_std:.3f}), Brier={info.brier:.4f}")

            # PR/Expectancy – materiał
            mask = ~np.isnan(proba_oof)
            p = proba_oof[mask]
            yv = y_clean.iloc[mask].to_numpy()

            pr_df = precision_recall_table(yv, p)
            exp_df = expectancy_table(yv, p, tp_mult=float(tp_mult), sl_mult=float(sl_mult), cost_R=float(cost_R))

            # sugerowane progi
            st.session_state["thr_suggested"] = suggest_threshold_by_f1(yv, p)
            st.session_state["exp_thr_suggested"] = suggest_threshold_by_expectancy(
                yv, p, tp_mult=float(tp_mult), sl_mult=float(sl_mult), cost_R=float(cost_R)
            )

            st.subheader("Precision–Recall vs. próg")
            fig = plt.figure(figsize=(7, 4))
            plt.plot(pr_df["thr"], pr_df["precision"], label="precision")
            plt.plot(pr_df["thr"], pr_df["recall"], label="recall")
            plt.plot(pr_df["thr"], pr_df["f1"], label="F1")
            if st.session_state["thr_suggested"] is not None:
                plt.axvline(st.session_state["thr_suggested"], linestyle="--", label=f"F1 max={st.session_state['thr_suggested']:.2f}")
            plt.xlabel("Próg p(win)"); plt.ylabel("Wartość"); plt.legend()
            st.pyplot(fig)

            st.subheader("Expectancy [R] vs. próg")
            fig2 = plt.figure(figsize=(7, 4))
            plt.plot(exp_df["thr"], exp_df["expectancy_R"], label="Expectancy (R)")
            if st.session_state["exp_thr_suggested"] is not None:
                plt.axvline(st.session_state["exp_thr_suggested"], linestyle="--",
                            label=f"E[R] max={st.session_state['exp_thr_suggested']:.2f}")
            plt.xlabel("Próg p(win)"); plt.ylabel("E[R] na trade")
            plt.legend()
            st.pyplot(fig2)

            st.info(
                f"Sugerowany próg F1: **{st.session_state['thr_suggested']:.2f}**  |  "
                f"Sugerowany próg Expectancy: **{st.session_state['exp_thr_suggested']:.2f}** "
                f"(koszt={float(cost_R):.2f} R)"
            )

            st.subheader("OOF p(win) w czasie")
            tmp = st.session_state["ml_proba"].copy()
            st.line_chart(tmp.set_index("timestamp")["proba"])

# ---------- TAB: Trafność OOF + PR + Expectancy ----------
with tabs[3]:
    st.subheader("Trafność predykcji (OOF) i Expectancy")
    if st.session_state["ml_info"] is None or st.session_state["ml_proba"] is None or st.session_state["ml_oof_labels"] is None:
        st.info("Najpierw wytrenuj model w zakładce „ML (training)”.")
    else:
        thr_default = float(st.session_state.get("exp_thr_suggested") or st.session_state.get("thr_suggested") or decision_thr)
        thr = st.slider("Próg p(win) do oceny", 0.50, 0.90, thr_default, 0.01)
        proba_df = st.session_state["ml_proba"].dropna(subset=["proba"]).copy()
        proba_df.set_index("row", inplace=True)
        y_series = st.session_state["ml_oof_labels"]
        common_idx = proba_df.index.intersection(y_series.index)
        p = proba_df.loc[common_idx, "proba"].values
        y_true = y_series.loc[common_idx].values.astype(int)

        mthr = threshold_metrics(y_true, p, thr)

        # Expectancy dla wybranego progu
        from models.ml import expectancy_from_precision
        exp_R = expectancy_from_precision(mthr["precision"], tp_mult=float(tp_mult), sl_mult=float(sl_mult), cost_R=float(cost_R))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Predykcje (p≥thr)", mthr["predicted_positives"])
        c2.metric("Trafione (TP)", mthr["TP"])
        c3.metric("Hit-rate (precision)", f"{mthr['precision']*100:.1f}%")
        c4.metric("Recall", f"{mthr['recall']*100:.1f}%")
        c5.metric("Expectancy [R]", f"{exp_R:.3f}")

        st.caption("Macierz pomyłek")
        cm_df = pd.DataFrame(
            [[mthr["TP"], mthr["FP"]],
             [mthr["FN"], mthr["TN"]]],
            index=["Pred=1/Rzecz=1 (TP)","Pred=0/Rzecz=1 (FN)"],
            columns=["Pred=1/Rzecz=1/0","Pred=0/Rzecz=0/1"]
        )
        st.dataframe(cm_df)

        st.caption("Szczegóły")
        st.table(pd.DataFrame({
            "predicted_positives":[mthr["predicted_positives"]],
            "precision":[f"{mthr['precision']*100:.2f}%"],
            "recall":[f"{mthr['recall']*100:.2f}%"],
            "f1":[f"{mthr['f1']*100:.2f}%"],
            "accuracy":[f"{mthr['accuracy']*100:.2f}%"],
            "expectancy_R":[f"{exp_R:.3f}"],
        }))

# ---------- TAB: Model Save/Load ----------
with tabs[4]:
    st.subheader("Zapis/odczyt modelu ML")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.session_state["ml_info"] is None:
            st.info("Wytrenuj model, aby zapisać artefakty.")
        else:
            if st.button("💾 Zapisz wytrenowany model + meta"):
                info = st.session_state["ml_info"]
                meta = {
                    "features": info.features,
                    "auc_mean": info.auc_mean,
                    "auc_std": info.auc_std,
                    "brier": info.brier,
                    "symbol": symbol,
                    "interval": interval_label,
                    "horizon_bars": int(horizon_bars),
                    "tp_mult": float(tp_mult),
                    "sl_mult": float(sl_mult),
                    "cost_R": float(cost_R),
                }
                model_path, meta_path = save_model_artifacts(info.model, meta, out_dir="models", base_name="lr_winprob")
                st.success(f"Zapisano:\n- {model_path}\n- {meta_path}")

    with col_b:
        model_file = st.file_uploader("Wczytaj .joblib", type=["joblib"])
        meta_file = st.file_uploader("Wczytaj .json (opcjonalnie)", type=["json"])
        if st.button("📂 Wczytaj model"):
            if model_file is None:
                st.warning("Wybierz plik .joblib.")
            else:
                tmp_model = os.path.join("models", "_tmp_upload.joblib")
                os.makedirs("models", exist_ok=True)
                with open(tmp_model, "wb") as f:
                    f.write(model_file.getbuffer())
                tmp_meta = None
                if meta_file is not None:
                    tmp_meta = os.path.join("models", "_tmp_upload.json")
                    with open(tmp_meta, "wb") as f:
                        f.write(meta_file.getbuffer())
                model, meta = load_model_artifacts(tmp_model, tmp_meta)
                st.session_state["loaded_model"] = model
                st.session_state["loaded_meta"] = meta
                st.success("Model wczytany.")
                if meta:
                    st.json(meta)

# ---------- TAB: ML predictions (live) ----------
with tabs[5]:
    st.subheader("Sygnały ML (próg decyzji)")
    model_to_use = None
    if st.session_state["loaded_model"] is not None:
        model_to_use = st.session_state["loaded_model"]
        st.caption("Używam modelu z pliku.")
    elif st.session_state["ml_info"] is not None:
        model_to_use = st.session_state["ml_info"].model
        st.caption("Używam modelu z treningu w sesji.")
    else:
        st.info("Brak modelu — wczytaj lub wytrenuj w poprzednich zakładkach.")

    if model_to_use is not None:
        if df.empty:
            st.info("Pobierz dane.")
        else:
            feats_all = make_features(df).dropna()
            idx_all = feats_all.index
            p_all = model_to_use.predict_proba(feats_all)[:, 1]
            pred = pd.DataFrame({"timestamp": df.loc[idx_all, "timestamp"].values, "proba": p_all}, index=idx_all)

            thr_live = st.slider(
                "Próg p(win) dla generacji sygnałów",
                0.50, 0.90,
                float(st.session_state.get("exp_thr_suggested") or st.session_state.get("thr_suggested") or decision_thr),
                0.01, key="thr_live"
            )
            hits = pred[pred["proba"] >= thr_live]
            st.write(f"Liczba sygnałów: {len(hits)}")

            if hits.empty:
                st.info("Brak sygnałów powyżej progu.")
            else:
                atr14 = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
                sig = pd.DataFrame({
                    "idx": hits.index.astype(int),
                    "side": "long",
                    "tp": (df["close"].iloc[hits.index] + float(tp_mult) * atr14.iloc[hits.index]).values,
                    "sl": (df["close"].iloc[hits.index] - float(sl_mult) * atr14.iloc[hits.index]).values,
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
                if tdf.empty:
                    st.info("Brak transakcji po egzekucji.")
                else:
                    tdf["success"] = tdf["net_pnl"] > 0
                    st.dataframe(tdf)

                    m = metrics(trades)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Trades", m["trades"])
                    c2.metric("Winrate", f"{m['winrate']*100:.1f}%")
                    pf = m["profit_factor"]; c3.metric("PF", f"{pf:.2f}" if np.isfinite(pf) else "∞")
                    c4.metric("MaxDD", f"{m['max_dd']:.2f}")

                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    csv_buf = io.StringIO(); tdf.to_csv(csv_buf, index=False)
                    st.download_button("⬇️ trades_ml.csv", csv_buf.getvalue(), file_name=f"trades_ml_{symbol}_{ts}.csv", mime="text/csv")
