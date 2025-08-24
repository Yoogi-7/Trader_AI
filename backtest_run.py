from __future__ import annotations

import sys
print("=== Python environment ===")
print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
print("============================================================")

try:
    import pandas as pd  # noqa: E402
except ModuleNotFoundError as e:
    raise SystemExit(
        "Brak pandas w aktywnym interpreterze.\n"
        "Użyj: .\\.venv\\Scripts\\Activate.ps1 ; potem: python backtest_run.py ..."
    ) from e

import argparse  # noqa: E402
import os  # noqa: E402
import numpy as np  # noqa: E402
from datetime import datetime  # noqa: E402

from core.execution import ExecConfig  # noqa: E402
from core.backtest import WFConfig, walk_forward_backtest, equity_curve, metrics  # noqa: E402

def _load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise SystemExit(
                "Nie mogę odczytać Parquet. Zainstaluj pyarrow: python -m pip install pyarrow\n"
                f"Błąd: {e}"
            )
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Nieznany format: {ext}")
    needed = {"timestamp","open","high","low","close","volume"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Brak kolumn: {needed - set(df.columns)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def demo_signal_fn(df: pd.DataFrame, te_slice: slice) -> pd.DataFrame:
    slc = df.iloc[te_slice]
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

    tp = (close.loc[idxs] + 2.0 * atr.loc[idxs]).values
    sl = (close.loc[idxs] - 1.0 * atr.loc[idxs]).values
    sig = pd.DataFrame({
        "idx": idxs,
        "side": "long",
        "tp": tp,
        "sl": sl,
        "horizon_bars": 60
    })
    sig = sig[sig["idx"] < te_slice.stop - 1].reset_index(drop=True)
    return sig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--fee_bp", type=float, default=1.0)
    ap.add_argument("--slip_ticks", type=int, default=1)
    ap.add_argument("--tick_size", type=float, default=0.1)
    ap.add_argument("--latency", type=int, default=1)
    ap.add_argument("--capital", type=float, default=100.0)
    ap.add_argument("--risk_pct", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = _load_any(args.data)

    exec_cfg = ExecConfig(
        latency_bar=args.latency,
        fee_bp=args.fee_bp,
        slippage_ticks=args.slip_ticks,
        tick_size=args.tick_size,
        contract_value=1.0,
        use_trailing=False,
        time_stop_bars=60,
    )
    wf_cfg = WFConfig(n_splits=5, min_train_bars=5000, step_bars=1000, use_walk_forward=True)

    trades, wf_table = walk_forward_backtest(
        df=df,
        signal_fn=demo_signal_fn,
        exec_cfg=exec_cfg,
        wf_cfg=wf_cfg,
        capital_ref=args.capital,
        risk_pct=args.risk_pct,
    )

    ec = equity_curve(trades)
    m = metrics(trades)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    wf_path = os.path.join(args.out, f"wf_{timestamp}.csv")
    eq_path = os.path.join(args.out, f"equity_{timestamp}.csv")
    tr_path = os.path.join(args.out, f"trades_{timestamp}.csv")
    wf_table.to_csv(wf_path, index=False)
    ec.to_csv(eq_path, index=False)
    pd.DataFrame(trades).to_csv(tr_path, index=False)

    print("==== SUMMARY ====")
    for k, v in m.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
    print(f"Saved: {wf_path}, {eq_path}, {tr_path}")

if __name__ == "__main__":
    main()
