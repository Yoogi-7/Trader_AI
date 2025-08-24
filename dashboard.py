# dashboard.py
# Trader AI — Meta-labeling + Purged TimeSeries CV (embargo) + celowany winrate

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
    suggest_threshold_for_precision,
    expectancy_table,
    suggest_threshold_by_expectancy,
    expectancy_from_precision,
    build_meta_frame,
    meta_time_series_fit_predict_proba,
    combined_metrics_for_thresholds,
    suggest_meta_threshold_for_precision,
)

st.set_page_config(
    page_title="Trader AI — Meta-labeling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== Helpers ==============

def resample_to(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if df.empty or not rule:
        return df.copy()
    # 'T' deprecated → używamy 'min'
    norm_rule = rule.replace("T", "min") if isinstance(rule, str) else rule
    g = df.set_index("timestamp")
    out = pd.DataFrame({
        "open": g["open"].resample(norm_rule).first(),
        "high": g["high"].resample(norm_rule).max(),
        "low": g["low"].resample(norm_rule).min(),
        "close": g["close"].resample(norm_rule).last(),
        "volume": g["volume"].resample(norm_rule).sum(),
    }).dropna().reset_index()
    return out

def regime_mask(df: pd.DataFrame, bullish_only: bool, min_atr_pct: float) -> pd.Series:
    m = pd.Series(True, index=df.index)
    close = df["close"].astype(float)
    if bullish_only:
        sma200 = close.rolling(200, min_periods=200).mean()
        m &= close > sma200
    if min_atr_pct > 0:
        atr = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
        atr_pct = atr / (close.replace(0, np.nan)).abs()
        m &= atr_pct >= (min_atr_pct / 100.0)
    return m.fillna(False)

def fetch_with_cache(symbol: str, fetch_interval: str, start_dt: datetime, end_dt: datetime, resample_rule: str | None) -> pd.DataFrame:
    full_cache = ensure_range_cached(symbol, fetch_interval, start_dt, end_dt)
    df = slice_range(full_cache, start_dt, end_dt)
    df = resample_to(df, resample_rule)
    return df

# ============== Sidebar ==============

st.sidebar.header("⚙️ Dane (cache)")
symbol = st.sidebar.text_input("Symbol (Binance)", "BTCUSDT")
interval_label = st.sidebar.selectbox("Interwał", ["1m","5m","10m (agregacja)","15m","30m","1h"], index=2)
days = st.sidebar.slider("Zakres (dni wstecz)", 1, 365, 90, 1)

# '10T' → '10min'
if interval_label == "10m (agregacja)":
    fetch_interval, resample_rule = "1m", "10min"
else:
    fetch_interval, resample_rule = interval_label, None

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=days)

c1, c2 = st.sidebar.columns(2)
fetch_btn = c1.button("📥 Pobierz/odśwież")
clear_btn = c2.button("🗑️ Wyczyść cache")
if clear_btn:
    ok, path = clear_cache(symbol, fetch_interval)
    st.sidebar.success(f"Usunięto cache: {path}" if ok else "Brak pliku cache")

# Reżim (opcjonalnie)
st.sidebar.header("📈 Filtr reżimu")
bullish_only = st.sidebar.checkbox("Tylko trend wzrostowy (close>SMA200)", True)
min_atr_pct = st.sidebar.number_input("Min ATR% (zmienność, %)", 0.0, 10.0, 0.5, 0.1)

# CV
st.sidebar.header("🧪 Walidacja (CV)")
n_splits = st.sidebar.slider("n_splits (CV)", 3, 10, 5, 1)
embargo = st.sidebar.slider("Embargo (bary)", 0, 200, 30, 5)

# Egzekucja
st.sidebar.header("🛠 Egzekucja")
fee_bp = st.sidebar.number_input("Prowizja (bp/strona)", 0.0, 50.0, 1.0, 0.1)
slip_ticks = st.sidebar.number_input("Slippage (ticki)", 0, 20, 1, 1)
tick_size = st.sidebar.number_input("Tick size", 0.0001, 100.0, 0.1, step=0.0001, format="%.4f")
latency = st.sidebar.number_input("Latency (bary)", 0, 5, 1, 1)
capital_ref = st.sidebar.number_input("Kapitał ref. [$]", 10.0, 100000.0, 100.0, 10.0)
risk_pct = st.sidebar.number_input("Ryzyko na trade [%]", 0.1, 10.0, 1.0, 0.1) / 100.0

# Cele WINRATE
st.sidebar.header("🎯 Cele WINRATE")
target_winrate = st.sidebar.slider("Docelowy WINRATE (precision)", 0.55, 0.99, 0.72, 0.01)
min_signals = st.sidebar.number_input("Min. sygnałów (OOF)", 10, 10000, 100, 10)
cost_R = st.sidebar.number_input("Koszt na trade [R] (fee+slip+latency)", 0.0, 1.0, 0.00, 0.01)

# Meta-labeling
st.sidebar.header("🧠 Meta-labeling")
enable_meta = st.sidebar.checkbox("Włącz meta-filter (Base→Meta)", True)

train_btn = st.sidebar.button("🤖 Trenuj (Base + opcjonalnie Meta)")

# ============== State ==============
if "data_df" not in st.session_state: st.session_state["data_df"] = pd.DataFrame()
if "ml_info" not in st.session_state: st.session_state["ml_info"] = None
if "ml_proba" not in st.session_state: st.session_state["ml_proba"] = None
if "ml_labels" not in st.session_state: st.session_state["ml_labels"] = None

if "meta_info" not in st.session_state: st.session_state["meta_info"] = None
if "meta_proba" not in st.session_state: st.session_state["meta_proba"] = None

if "thr_base" not in st.session_state: st.session_state["thr_base"] = None
if "thr_meta" not in st.session_state: st.session_state["thr_meta"] = None

st.title("🧠 Trader AI — Meta-labeling (winrate booster)")

# ============== Fetch ==============
if fetch_btn:
    with st.spinner(f"Cache + inkrementalne pobranie ({symbol}, {fetch_interval})…"):
        df = fetch_with_cache(symbol, fetch_interval, start_dt, end_dt, resample_rule)
        st.session_state["data_df"] = df
        st.success(f"Dane gotowe: {len(df)} świec.")

df = st.session_state["data_df"]

tabs = st.tabs([
    "🧱 Dane",
    "🤖 Trening (Base + Meta)",
    "📊 OOF: Base vs Base+Meta",
    "🔔 Live sygnały (gating)",
])

# ============== TAB: Dane ==============
with tabs[0]:
    st.subheader("Podgląd danych")
    if df.empty:
        st.info("Kliknij „Pobierz/odśwież”.")
    else:
        st.dataframe(df.head(200))
        rm = regime_mask(df, bullish_only, min_atr_pct)
        st.caption(f"Reżim TRUE dla {int(rm.sum())}/{len(rm)} świec ({rm.mean()*100:.1f}%).")

# ============== TAB: Trening ==============
with tabs[1]:
    if train_btn:
        if df.empty:
            st.warning("Najpierw pobierz dane.")
        else:
            with st.spinner("Buduję cechy i triple-barrier etykiety…"):
                feats = make_features(df)
                tb = triple_barrier_labels(
                    df.assign(timestamp=df["timestamp"]),
                    cfg=TripleBarrierConfig(
                        horizon_bars=60, use_atr=True, atr_period=14,
                        tp_mult=2.0, sl_mult=1.0, percent_mode=False, side="long"
                    )
                )
                y = tb["label"].replace({-1: np.nan})
                data = pd.concat([feats, y.rename("label")], axis=1).dropna()

                reg_m = regime_mask(df, bullish_only, min_atr_pct).reindex(data.index).fillna(False)
                data = data.loc[reg_m]

                X = data.drop(columns=["label"])
                y_clean = data["label"].astype(int)

            with st.spinner("CV (purged + embargo) — model BASE…"):
                p_base_oof, base_info = time_series_fit_predict_proba(
                    X, y_clean, n_splits=int(n_splits), embargo=int(embargo)
                )

            st.session_state["ml_info"] = base_info
            st.session_state["ml_proba"] = pd.Series(p_base_oof, index=X.index)
            st.session_state["ml_labels"] = y_clean

            mask = ~np.isnan(p_base_oof)
            if mask.any():
                yv = y_clean.iloc[mask].to_numpy()
                p_base = p_base_oof[mask]

                thr_base_prec = suggest_threshold_for_precision(yv, p_base, float(target_winrate), int(min_signals))
                thr_base_f1 = suggest_threshold_by_f1(yv, p_base)
                thr_base_exp = suggest_threshold_by_expectancy(yv, p_base, 2.0, 1.0, float(cost_R))

                st.session_state["thr_base"] = thr_base_prec if thr_base_prec is not None else thr_base_f1
            else:
                st.session_state["thr_base"] = 0.7
                yv = np.array([])
                p_base = np.array([])

            st.success(
                "BASE gotowy | "
                f"AUC={base_info.auc_mean:.3f}±{base_info.auc_std:.3f}, Brier={base_info.brier:.4f} | "
                f"thr_base={st.session_state['thr_base']:.2f}"
            )

            # META
            if enable_meta:
                with st.spinner("CV — META (na OOF base)…"):
                    X_meta, y_meta = build_meta_frame(X, y_clean, p_base_oof)
                    if X_meta.empty:
                        st.warning("Brak danych do meta-modelu (OOF base puste).")
                        st.session_state["meta_info"] = None
                        st.session_state["meta_proba"] = None
                    else:
                        p_meta_oof, meta_info = meta_time_series_fit_predict_proba(
                            X_meta, y_meta, n_splits=int(n_splits), embargo=int(embargo)
                        )
                        st.session_state["meta_info"] = meta_info
                        meta_proba_series = pd.Series(np.nan, index=X.index)
                        meta_proba_series.loc[X_meta.index] = p_meta_oof
                        st.session_state["meta_proba"] = meta_proba_series

                        thr_base_use = float(st.session_state["thr_base"])
                        idx_common = X_meta.index
                        y_c = y_clean.loc[idx_common].to_numpy()
                        p_b = st.session_state["ml_proba"].loc[idx_common].to_numpy()
                        p_m = meta_proba_series.loc[idx_common].to_numpy()

                        thr_meta = suggest_meta_threshold_for_precision(
                            y_true=y_c,
                            p_base=p_b,
                            p_meta=p_m,
                            thr_base=thr_base_use,
                            target_precision=float(target_winrate),
                            min_signals=int(min_signals),
                        )
                        st.session_state["thr_meta"] = thr_meta

                        st.success(
                            "META gotowa | "
                            f"AUC={meta_info.auc_mean:.3f}±{meta_info.auc_std:.3f}, Brier={meta_info.brier:.4f} | "
                            f"thr_meta={thr_meta if thr_meta is not None else 'brak'}"
                        )

            # PR-curve BASE
            if mask.any():
                st.subheader("BASE: Precision/Recall/F1 vs próg")
                pr_df = precision_recall_table(yv, p_base, steps=301)
                fig = plt.figure(figsize=(7, 4))
                plt.plot(pr_df["thr"], pr_df["precision"], label="precision")
                plt.plot(pr_df["thr"], pr_df["recall"], label="recall")
                plt.plot(pr_df["thr"], pr_df["f1"], label="F1")
                if st.session_state["thr_base"] is not None:
                    plt.axvline(float(st.session_state["thr_base"]), linestyle="--", label=f"thr_base={float(st.session_state['thr_base']):.2f}")
                plt.xlabel("Próg p_base"); plt.ylabel("Wartość"); plt.legend()
                st.pyplot(fig)

# ============== TAB: OOF porównanie ==============
with tabs[2]:
    st.subheader("OOF: Base vs Base+Meta")
    if st.session_state["ml_info"] is None or st.session_state["ml_proba"] is None:
        st.info("Wytrenuj model w poprzedniej zakładce.")
    else:
        y_series = st.session_state["ml_labels"]
        p_base_series = st.session_state["ml_proba"]
        thr_base_use = float(st.session_state.get("thr_base") or 0.7)

        mask = ~np.isnan(p_base_series)
        idx = p_base_series.index[mask]
        yv = y_series.loc[idx].to_numpy()
        p_base = p_base_series.loc[idx].to_numpy()

        m_base = threshold_metrics(yv, p_base, thr_base_use)

        if enable_meta and st.session_state["meta_proba"] is not None and st.session_state["thr_meta"] is not None:
            p_meta = st.session_state["meta_proba"].loc[idx].to_numpy()
            thr_meta_use = float(st.session_state["thr_meta"])
            m_combo = combined_metrics_for_thresholds(yv, p_base, p_meta, thr_base_use, thr_meta_use)
        else:
            m_combo = {"predicted_positives":0,"TP":0,"FP":0,"FN":int((yv==1).sum()),
                       "TN":int((yv==0).sum()),"precision":0.0,"recall":0.0,"f1":0.0,"accuracy":0.0}

        c1, c2, c3 = st.columns(3)
        c1.metric("BASE — sygnały", m_base["predicted_positives"])
        c2.metric("BASE — WINRATE", f"{m_base['precision']*100:.1f}%")
        c3.metric("BASE — Recall", f"{m_base['recall']*100:.1f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("BASE+META — sygnały", m_combo["predicted_positives"])
        c5.metric("BASE+META — WINRATE", f"{m_combo['precision']*100:.1f}%")
        c6.metric("BASE+META — Recall", f"{m_combo['recall']*100:.1f}%")

        st.caption("Szczegóły (BASE)")
        st.table(pd.DataFrame({k:[v] for k,v in m_base.items()}))

        st.caption("Szczegóły (BASE+META)")
        st.table(pd.DataFrame({k:[v] for k,v in m_combo.items()}))

# ============== TAB: Live sygnały ==============
with tabs[3]:
    st.subheader("Live: generacja sygnałów (gating Base + Meta + reżim)")
    if df.empty:
        st.info("Pobierz dane.")
    elif st.session_state["ml_info"] is None:
        st.info("Wytrenuj model.")
    else:
        base_model = st.session_state["ml_info"].model
        meta_model = st.session_state["meta_info"].model if (enable_meta and st.session_state["meta_info"] is not None) else None

        thr_base_live = float(st.session_state.get("thr_base") or 0.7)
        thr_meta_live = float(st.session_state.get("thr_meta") or 0.7)

        feats_all = make_features(df).dropna()
        idx_all = feats_all.index

        p_base_full = base_model.predict_proba(feats_all)[:, 1]
        pred_base = pd.Series(p_base_full, index=idx_all, name="p_base")

        if meta_model is not None:
            X_meta_live = pd.DataFrame({"p_base": pred_base})
            p_meta_full = meta_model.predict_proba(X_meta_live)[:, 1]
            pred_meta = pd.Series(p_meta_full, index=idx_all, name="p_meta")
        else:
            pred_meta = pd.Series(np.ones(len(pred_base)), index=idx_all, name="p_meta")

        reg_m = regime_mask(df, bullish_only, min_atr_pct)
        take = (pred_base >= thr_base_live) & (pred_meta >= thr_meta_live) & reg_m.reindex(idx_all).fillna(False)
        hits_idx = idx_all[take]

        st.write(f"Sygnały (po gatingu): **{int(take.sum())}**")
        hits = pd.DataFrame({
            "timestamp": df.loc[hits_idx, "timestamp"].values,
            "p_base": pred_base.loc[hits_idx].values,
            "p_meta": pred_meta.loc[hits_idx].values,
        }, index=hits_idx).sort_index()
        st.dataframe(hits.tail(100))

        if len(hits_idx) > 0:
            atr14 = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
            sig = pd.DataFrame({
                "idx": hits.index.astype(int),
                "side": "long",
                "tp": (df["close"].iloc[hits.index] + 2.0 * atr14.iloc[hits.index]).values,
                "sl": (df["close"].iloc[hits.index] - 1.0 * atr14.iloc[hits.index]).values,
                "horizon_bars": 60,
            }).reset_index(drop=True)

            exec_cfg = ExecConfig(
                latency_bar=int(latency), fee_bp=float(fee_bp),
                slippage_ticks=int(slip_ticks), tick_size=float(tick_size),
                contract_value=1.0, use_trailing=False, time_stop_bars=60
            )
            with st.spinner("Symuluję egzekucję…"):
                trades = backtest_trades(df, sig, exec_cfg, capital_ref=float(capital_ref), risk_pct=float(risk_pct))
            if trades:
                m = metrics(trades)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trades", m["trades"])
                c2.metric("Winrate", f"{m['winrate']*100:.1f}%")
                pf = m["profit_factor"]; c3.metric("PF", f"{pf:.2f}" if np.isfinite(pf) else "∞")
                c4.metric("MaxDD", f"{m['max_dd']:.2f}")
