"""
Shared scanning pipeline for TRADER_AI.
- Optionally performs incremental ingest from Bitget,
- Resamples 1m to signal TFs and higher TFs,
- Applies HTF trend filter (EMA-200 + slope),
- Detects setups (breakout + fallback momentum),
- Computes levels and EV filters,
- Returns a list of signal dicts.
All texts/comments are in English.
"""
from __future__ import annotations

from typing import List, Dict, Any, Iterable
import pandas as pd

from app.config import POLICY
from app.data.ingest_bitget import incremental_update
from app.data.resample import resample_to_tf
from app.data.store import path_raw, path_tf
from app.rules.trend import trend_status
from app.rules.detectors import detect_candidates
from app.rules.levels import propose_levels
from app.rules.ev import (
    Costs,
    position_notional,
    trade_costs,
    net_profit_at_tp,
    expected_value,
    passes_min_profit,
)

TF_ORDER = ["10m", "15m", "30m", "1h", "2h", "4h"]

def next_two_higher_tfs(tf: str):
    i = TF_ORDER.index(tf)
    h1 = TF_ORDER[min(i + 1, len(TF_ORDER) - 1)]
    h2 = TF_ORDER[min(i + 2, len(TF_ORDER) - 1)]
    return h1, h2

def scan_symbols(
    symbols: Iterable[str],
    equity: float = 5_000.0,
    risk_profile: str = "medium",
    signal_tfs: Iterable[str] = ("10m", "15m", "30m"),
    run_ingest: bool = True,
) -> List[Dict[str, Any]]:
    """
    Main scanning entrypoint used by both CLI/bootstrap and API.
    Returns a flat list of signal dicts for all given symbols and TFs.
    """
    results: List[Dict[str, Any]] = []

    det_defaults = POLICY["detector_defaults"]
    htf_ema_len = POLICY["htf_confirmations"]["ema_len"]
    prof = POLICY["risk_profiles"][risk_profile]

    for symbol in symbols:
        if run_ingest:
            added = incremental_update(symbol)
            print(f"[INGEST] {symbol} +{added} rows")

        p_1m = path_raw(symbol)
        df_1m = pd.read_parquet(p_1m)

        for tf in signal_tfs:
            df_tf = resample_to_tf(df_1m, tf)
            h1, h2 = next_two_higher_tfs(tf)
            df_h1 = resample_to_tf(df_1m, h1)
            df_h2 = resample_to_tf(df_1m, h2)

            path_tf(symbol, tf).parent.mkdir(parents=True, exist_ok=True)
            df_tf.to_parquet(path_tf(symbol, tf))
            df_h1.to_parquet(path_tf(symbol, h1))
            df_h2.to_parquet(path_tf(symbol, h2))

            t1 = trend_status(df_h1, ema_len=htf_ema_len)
            t2 = trend_status(df_h2, ema_len=htf_ema_len)
            if t1 == "flat" or t2 == "flat" or t1 != t2:
                continue

            tf_cfg = POLICY["tf_params"][tf]
            lookback = tf_cfg.get("lookback", det_defaults["breakout_lookback"])
            cands = detect_candidates(
                df_tf,
                lookback=lookback,
                use_fallback=det_defaults["use_fallback"],
                fallback_ma_len=det_defaults["fallback_ma_len"],
                fallback_squeeze_len=det_defaults["fallback_squeeze_len"],
                fallback_squeeze_threshold=det_defaults["fallback_squeeze_threshold"],
            )
            if not cands:
                continue

            c = cands[-1]

            if (t1 == "up" and c["side"] == "SHORT") or (t1 == "down" and c["side"] == "LONG"):
                continue

            entry, sl, tp1, tp2 = propose_levels(
                c["side"], c["swing_low"], c["swing_high"], c["atr"],
                k_atr_sl=tf_cfg["k_atr_sl"]
            )

            notional = position_notional(
                equity=equity,
                risk=prof["risk_per_trade"],
                entry=entry,
                sl=sl,
                lmax=prof["max_leverage"],
            )
            costs = Costs(
                taker_bps_in=POLICY["execution"]["taker_fee_bps_default"],
                taker_bps_out=POLICY["execution"]["taker_fee_bps_default"],
                slippage_pct=POLICY["execution"]["avg_slippage_pct"],
            )
            fee, slip = trade_costs(notional, costs)

            p_hit = 0.60  # placeholder, will be replaced by ML
            rr = abs(tp1 - entry) / abs(entry - sl)
            rr_min = tf_cfg["rr_min"]
            min_per_100 = tf_cfg["min_net_per_100"]
            net_tp = net_profit_at_tp(entry, tp1, notional, fee, slip)
            ev = expected_value(p_hit, entry, tp1, sl, notional, fee, slip)

            ok = (rr >= rr_min) and passes_min_profit(net_tp, notional, min_per_100) and (ev > 0)

            results.append({
                "symbol": symbol,
                "tf": tf, "htf1": t1, "htf2": t2,
                "ts": str(c["i"]), "side": c["side"],
                "entry": round(entry, 2), "sl": round(sl, 2),
                "tp1": round(tp1, 2), "tp2": round(tp2, 2),
                "rr": round(rr, 3), "p_hit": p_hit,
                "notional": round(notional, 2),
                "fee": round(fee, 2), "slip": round(slip, 2),
                "net_tp": round(net_tp, 2), "ev": round(ev, 2),
                "ok": ok
            })

    return results
