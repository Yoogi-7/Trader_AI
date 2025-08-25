from __future__ import annotations
import pandas as pd
import numpy as np

def _safe_pct(x):
    try:
        return float(x)
    except:
        return np.nan

def metrics_from_signals(df: pd.DataFrame) -> dict:
    """
    Oczekuje df z kolumnami:
    status ∈ {'TP','SL','TP1_TRAIL','EXPIRED',...}, pnl_usd, pnl_pct
    """
    d = {}
    if df.empty:
        return {
            "trades": 0, "closed": 0, "wins": 0, "winrate": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "expectancy_pct": 0.0, "pnl_usd": 0.0,
            "max_dd_usd": 0.0, "sharpe": 0.0
        }

    df = df.copy()
    df["pnl_pct"] = df["pnl_pct"].apply(_safe_pct)
    df["pnl_usd"] = df["pnl_usd"].apply(_safe_pct)

    closed_mask = df["status"].isin(["TP","SL","TP1_TRAIL"])
    wins_mask = df["status"].isin(["TP","TP1_TRAIL"])
    losses_mask = df["status"].isin(["SL"])

    trades = len(df)
    closed = int(closed_mask.sum())
    wins = int(wins_mask.sum())
    winrate = (wins/closed*100.0) if closed else 0.0

    avg_win = df.loc[wins_mask, "pnl_pct"].mean() if wins else 0.0
    avg_loss = df.loc[losses_mask, "pnl_pct"].mean() if losses_mask.any() else 0.0

    # Expectancy (tylko na closed)
    exp = 0.0
    if closed:
        p_win = wins/closed
        p_loss = 1.0 - p_win
        avg_win_usd = df.loc[wins_mask, "pnl_usd"].mean() if wins else 0.0
        avg_loss_usd = df.loc[losses_mask, "pnl_usd"].mean() if losses_mask.any() else 0.0
        # w % na trade
        exp_pct = p_win * (avg_win if not np.isnan(avg_win) else 0.0) + \
                  p_loss * (avg_loss if not np.isnan(avg_loss) else 0.0)
        exp = exp_pct

    pnl_usd_total = df["pnl_usd"].sum(skipna=True)

    # Equity curve & DD (po kolei wg czasu)
    dfe = df.sort_values("closed_ts_ms", na_position="last").copy()
    dfe["pnl_usd_fill"] = dfe["pnl_usd"].fillna(0.0)
    dfe["equity"] = dfe["pnl_usd_fill"].cumsum()
    dfe["highwater"] = dfe["equity"].cummax()
    dfe["dd"] = dfe["equity"] - dfe["highwater"]
    max_dd_usd = float(dfe["dd"].min()) if not dfe.empty else 0.0

    # pseudo-Sharpe na dziennych PnL (jeśli masz timestampy dzienne; jeśli nie – na kolejnych transakcjach)
    # Tutaj Sharpe na serii transakcji (nie dziennych) jako przybliżenie
    r = dfe["pnl_usd_fill"]
    sharpe = 0.0
    if len(r) > 2 and r.std(ddof=1) > 0:
        sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))

    return {
        "trades": trades,
        "closed": closed,
        "wins": wins,
        "winrate": winrate,
        "avg_win_pct": float(avg_win) if avg_win==avg_win else 0.0,
        "avg_loss_pct": float(avg_loss) if avg_loss==avg_loss else 0.0,
        "expectancy_pct": float(exp) if exp==exp else 0.0,
        "pnl_usd": float(pnl_usd_total),
        "max_dd_usd": float(max_dd_usd),
        "sharpe": sharpe,
    }

def equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["idx","equity","dd","highwater"])
    dfe = df.sort_values("closed_ts_ms", na_position="last").copy()
    dfe["pnl_usd_fill"] = dfe["pnl_usd"].fillna(0.0)
    dfe["equity"] = dfe["pnl_usd_fill"].cumsum()
    dfe["highwater"] = dfe["equity"].cummax()
    dfe["dd"] = dfe["equity"] - dfe["highwater"]
    dfe["idx"] = range(len(dfe))
    return dfe[["idx","equity","dd","highwater","ts_ms","closed_ts_ms","symbol","timeframe","status"]]
