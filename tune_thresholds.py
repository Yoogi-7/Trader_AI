"""
Auto-strojenie progów per para/TF pod target winrate, z maksymalnym coverage.
Użycie CLI:
    python tune_thresholds.py           # tryb z configu (domyślnie EV)
    python tune_thresholds.py --mode p  # próg p(win)
"""

import argparse
import os
import sqlite3
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(cfg: dict):
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

def _fees_total_pct(cfg: dict) -> float:
    fees_cfg = cfg["signals"]["fees"]
    mode = cfg["models"]["meta"]["gating"].get("costs_mode", "auto")
    if mode == "maker":
        f = float(fees_cfg["maker_pct"])
    elif mode == "taker":
        f = float(fees_cfg["taker_pct"])
    else:
        f = float(fees_cfg["maker_pct"]) if fees_cfg.get("assume_maker", False) else float(fees_cfg["taker_pct"])
    return 2.0 * f

def _net_move_pct(direction: str, entry: float, level: float) -> float:
    if direction == "long":
        return (level / entry - 1.0) * 100.0
    else:
        return (entry / level - 1.0) * 100.0

def _ev_pct_row(row, fees_total: float, slip_pct: float) -> float:
    p = float(row["ml_p"])
    entry = float(row["entry"]); tp1 = float(row["tp1"]); sl = float(row["sl"])
    g_win = _net_move_pct(row["direction"], entry, tp1)
    g_loss = _net_move_pct(row["direction"], entry, sl)
    g_win_net = g_win - fees_total - slip_pct
    g_loss_net = g_loss - fees_total - slip_pct
    return p * g_win_net + (1.0 - p) * g_loss_net

def _closed_mask(df: pd.DataFrame) -> pd.Series:
    return df["status"].isin(["TP","SL","TP1_TRAIL"])

def _win_mask(df: pd.DataFrame) -> pd.Series:
    return df["status"].isin(["TP","TP1_TRAIL"])

def _best_threshold_for_pair(df: pd.DataFrame, mode: str, target_win: float) -> tuple[float | None, float, int]:
    dfc = df[_closed_mask(df)].dropna(subset=["ml_p"]).copy()
    if dfc.empty:
        return None, 0.0, 0

    if mode == "ev":
        candidates = np.round(np.arange(-1.0, 3.01, 0.1), 2)
        key = "ev_pct"
    else:
        candidates = np.round(np.arange(0.50, 0.91, 0.01), 2)
        key = "ml_p"

    best = None; best_cov = -1.0; best_win = 0.0; best_n = 0
    total = len(dfc)
    for thr in candidates:
        sub = dfc[dfc[key] >= thr]
        n = len(sub)
        if n == 0:
            continue
        wins = int(_win_mask(sub).sum())
        winrate = wins / n * 100.0
        coverage = n / total * 100.0
        if winrate >= target_win and (coverage > best_cov or (coverage == best_cov and winrate > best_win)):
            best = float(thr); best_cov = coverage; best_win = winrate; best_n = n

    if best is None:
        for thr in reversed(candidates):
            sub = dfc[dfc[key] >= thr]; n = len(sub)
            if n == 0:
                continue
            wins = int(_win_mask(sub).sum()); winrate = wins / n * 100.0
            coverage = n / total * 100.0
            if coverage >= 5.0:
                best = float(thr); best_cov = coverage; best_win = winrate; best_n = n
                break

    if best is None:
        return None, 0.0, 0
    return best, best_win, best_n

def _run(mode: str | None = None, window_days: int | None = None):
    cfg = load_config()
    db_path = cfg["app"]["db_path"]
    gating = cfg["models"]["meta"]["gating"]
    target_win = float(gating.get("target_winrate_pct", 60.0))
    min_closed = int(gating.get("min_closed_samples", 60))
    window_days = int(window_days or gating.get("window_days", 120))
    mode = mode or ("ev" if gating.get("use_ev", False) else "p")

    with sqlite3.connect(db_path, timeout=60) as conn:
        q = """
        SELECT ts_ms, closed_ts_ms, symbol, timeframe, status, direction, entry, sl, tp1, ml_p
        FROM signals
        ORDER BY ts_ms ASC
        """
        df = pd.read_sql_query(q, conn)
    if df.empty:
        print("Brak sygnałów w bazie."); return

    if window_days > 0:
        cutoff = int((datetime.now(tz=timezone.utc) - timedelta(days=window_days)).timestamp() * 1000)
        df = df[df["ts_ms"] >= cutoff]

    if mode == "ev":
        fees_total = _fees_total_pct(cfg)
        slip = float(cfg["signals"]["slippage_pct"])
        df = df.dropna(subset=["ml_p"])
        if df.empty:
            print("Brak ml_p do obliczenia EV."); return
        df["ev_pct"] = df.apply(lambda r: _ev_pct_row(r, fees_total, slip), axis=1)

    per = {}
    for (sym, tf), grp in df.groupby(["symbol","timeframe"]):
        closed_n = int(_closed_mask(grp).sum())
        if closed_n < min_closed:
            print(f"[SKIP] {sym} {tf}: zamkniętych {closed_n}<{min_closed}")
            continue
        thr, win, n = _best_threshold_for_pair(grp, mode, target_win)
        if thr is None:
            print(f"[NOOPT] {sym} {tf}")
            continue
        if sym not in per:
            per[sym] = {}
        if mode == "ev":
            per[sym][tf] = {"min_ev_pct": float(round(thr, 2))}
            print(f"[SET] {sym} {tf}: min_ev_pct={thr:.2f}% (win≈{win:.1f}%, n={n})")
        else:
            per[sym][tf] = {"threshold": float(round(thr, 2))}
            print(f"[SET] {sym} {tf}: threshold={thr:.2f} (win≈{win:.1f}%, n={n})")

    if not per:
        print("Brak rekomendacji progów."); return

    # zapis do config.yaml (scalanie)
    if "models" not in cfg: cfg["models"] = {}
    if "meta" not in cfg["models"]: cfg["models"]["meta"] = {}
    if "gating" not in cfg["models"]["meta"]: cfg["models"]["meta"]["gating"] = {}
    exist = cfg["models"]["meta"]["gating"].get("per_pair_tf", {}) or {}
    for sym, mp in per.items():
        if sym not in exist:
            exist[sym] = {}
        exist[sym].update(mp)
    cfg["models"]["meta"]["gating"]["per_pair_tf"] = exist

    # backup i zapis
    backup = "config.yaml.bak"
    if not os.path.exists(backup):
        try:
            import shutil; shutil.copyfile("config.yaml", backup)
            print(f"Backup zapisany: {backup}")
        except Exception:
            pass

    save_config(cfg)
    print("Zapisano progi do config.yaml.")

def tune_thresholds_programmatic(mode: str | None = None, window_days: int | None = None):
    _run(mode=mode, window_days=window_days)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ev","p"], default=None)
    parser.add_argument("--window-days", type=int, default=None)
    args = parser.parse_args()
    _run(mode=args.mode, window_days=args.window_days)

if __name__ == "__main__":
    main()
