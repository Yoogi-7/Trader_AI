from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict

from core.execution import ExecConfig, backtest_trades

@dataclass
class WFConfig:
    n_splits: int = 5
    min_train_bars: int = 5000
    step_bars: int = 1000
    use_walk_forward: bool = True

def generate_signals_from_proba(
    proba: np.ndarray,
    df: pd.DataFrame,
    threshold: float = 0.55,
    side: str = "long",
    tp: float | None = None,
    sl: float | None = None,
    horizon_bars: int = 60,
) -> pd.DataFrame:
    idxs = np.where(proba >= threshold)[0]
    if len(idxs) == 0:
        return pd.DataFrame(columns=["idx","side","tp","sl","horizon_bars"])
    sig = pd.DataFrame({
        "idx": idxs,
        "side": side,
        "tp": tp if tp is not None else np.nan,
        "sl": sl if sl is not None else np.nan,
        "horizon_bars": horizon_bars
    })
    sig = sig[sig["idx"] < len(df) - 2].reset_index(drop=True)
    return sig

def equity_curve(trades: List[Dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["timestamp_exit","equity"])
    t = pd.DataFrame(trades).sort_values("timestamp_exit")
    t["equity"] = t["net_pnl"].cumsum()
    return t[["timestamp_exit", "equity"]]

def metrics(trades: List[Dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "winrate": 0.0,
            "avg_R": 0.0,
            "profit_factor": 0.0,
            "max_dd": 0.0,
            "ulcer": 0.0,
            "mar": 0.0,
            "expectancy": 0.0,
        }
    t = pd.DataFrame(trades)
    wins = t.loc[t["net_pnl"] > 0, "net_pnl"].sum()
    losses = -t.loc[t["net_pnl"] < 0, "net_pnl"].sum()
    profit_factor = wins / losses if losses > 0 else np.inf
    winrate = (t["net_pnl"] > 0).mean()

    R = t["net_pnl"] / 1.0
    avg_R = R.mean()
    expectancy = t["net_pnl"].mean()

    eq = t["net_pnl"].cumsum()
    running_max = eq.cummax()
    dd = eq - running_max
    max_dd = dd.min()
    ulcer = np.sqrt((dd.pow(2)).mean())
    total = eq.iloc[-1]
    mar = (total / abs(max_dd)) if max_dd < 0 else np.inf

    return {
        "trades": int(len(t)),
        "winrate": float(winrate),
        "avg_R": float(avg_R),
        "profit_factor": float(profit_factor),
        "max_dd": float(max_dd),
        "ulcer": float(ulcer),
        "mar": float(mar),
        "expectancy": float(expectancy),
    }

def walk_forward_backtest(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, slice], pd.DataFrame],
    exec_cfg: ExecConfig,
    wf_cfg: WFConfig,
    capital_ref: float = 100.0,
    risk_pct: float = 0.01,
) -> Tuple[List[Dict], pd.DataFrame]:
    n = len(df)
    start = wf_cfg.min_train_bars
    all_trades: List[Dict] = []
    rows = []

    while start + wf_cfg.step_bars < n:
        te_slice = slice(start, start + wf_cfg.step_bars)
        sig = signal_fn(df, te_slice)
        if len(sig) > 0:
            trades = backtest_trades(df, sig, exec_cfg, capital_ref=capital_ref, risk_pct=risk_pct)
            all_trades.extend(trades)
            m = metrics(trades)
        else:
            m = metrics([])

        rows.append({
            "train_end_idx": start,
            "test_start_idx": start,
            "test_end_idx": start + wf_cfg.step_bars,
            **m
        })
        start += wf_cfg.step_bars

    return all_trades, pd.DataFrame(rows)
