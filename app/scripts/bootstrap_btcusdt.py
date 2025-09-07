"""
Bootstrap script for BTCUSDT on Bitget:
1) Incrementally fetch 1m OHLCV and store as Parquet,
2) Resample to 10m/15m/30m,
3) HTF trend filter (two higher TFs must agree),
4) Detect setups (breakout + fallback momentum) and compute levels + EV,
5) Print candidate signals to stdout.
"""
import pandas as pd

from app.config import POLICY
from app.data.store import path_raw, path_tf
from app.data.ingest_bitget import incremental_update
from app.data.resample import resample_to_tf
from app.rules.detectors import detect_candidates
from app.rules.levels import propose_levels
from app.rules.trend import trend_status
from app.rules.ev import Costs, position_notional, trade_costs, net_profit_at_tp, expected_value, passes_min_profit

SYMBOL = "BTCUSDT"
EQUITY = 5_000.0      # demo equity for sizing
PROFILE = "medium"    # risk profile used for sizing

HTF_EMA_LEN = POLICY["htf_confirmations"]["ema_len"]

# Timeframe order for "two levels up"
TF_ORDER = ["10m", "15m", "30m", "1h", "2h", "4h"]
def next_two_higher_tfs(tf: str):
    i = TF_ORDER.index(tf)
    h1 = TF_ORDER[min(i + 1, len(TF_ORDER) - 1)]
    h2 = TF_ORDER[min(i + 2, len(TF_ORDER) - 1)]
    return h1, h2

def main() -> None:
    # 1) Ingest 1m
    added = incremental_update(SYMBOL)
    print(f"[INGEST] Added rows: {added}")

    # 2) Load 1m
    p_1m = path_raw(SYMBOL)
    df_1m = pd.read_parquet(p_1m)

    signals = []
    det_defaults = POLICY["detector_defaults"]
    prof = POLICY["risk_profiles"][PROFILE]

    for tf in ["10m", "15m", "30m"]:
        # Resample signal TF + its two higher TFs
        df_tf = resample_to_tf(df_1m, tf)
        h1, h2 = next_two_higher_tfs(tf)
        df_h1 = resample_to_tf(df_1m, h1)
        df_h2 = resample_to_tf(df_1m, h2)

        # Persist resampled data
        path_tf(SYMBOL, tf).parent.mkdir(parents=True, exist_ok=True)
        df_tf.to_parquet(path_tf(SYMBOL, tf))
        df_h1.to_parquet(path_tf(SYMBOL, h1))
        df_h2.to_parquet(path_tf(SYMBOL, h2))

        # 3) HTF trend check
        t1 = trend_status(df_h1, ema_len=HTF_EMA_LEN)
        t2 = trend_status(df_h2, ema_len=HTF_EMA_LEN)
        if t1 == "flat" or t2 == "flat" or t1 != t2:
            # skip if higher timeframes are not aligned
            continue

        # 4) Detect candidates on signal TF
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

        # Evaluate last candidate
        c = cands[-1]
        # Enforce direction to match HTFs: if HTFs are 'up', prefer LONG; if 'down', prefer SHORT
        if t1 == "up" and c["side"] == "SHORT":
            # flip only if it makes sense? for MVP, just skip mismatched side
            pass_side_filter = False
        elif t1 == "down" and c["side"] == "LONG":
            pass_side_filter = False
        else:
            pass_side_filter = True
        if not pass_side_filter:
            continue

        entry, sl, tp1, tp2 = propose_levels(
            c["side"], c["swing_low"], c["swing_high"], c["atr"],
            k_atr_sl=tf_cfg["k_atr_sl"]
        )

        notional = position_notional(
            equity=EQUITY,
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

        # MVP probability placeholder; will be replaced with a model later
        p_hit = 0.60
        rr = abs(tp1 - entry) / abs(entry - sl)
        rr_min = tf_cfg["rr_min"]
        min_per_100 = tf_cfg["min_net_per_100"]

        net_tp = net_profit_at_tp(entry, tp1, notional, fee, slip)
        ev = expected_value(p_hit, entry, tp1, sl, notional, fee, slip)

        ok = (rr >= rr_min) and passes_min_profit(net_tp, notional, min_per_100) and (ev > 0)

        signals.append({
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

    print("\n=== SIGNALS (BTCUSDT) ===")
    if not signals:
        print("No candidates passed HTF filter in the latest window.")
    else:
        for s in signals:
            print(s)

if __name__ == "__main__":
    main()
