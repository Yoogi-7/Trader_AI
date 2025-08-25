from __future__ import annotations
import sqlite3
from typing import Optional
import pandas as pd
from core.indicators import atr as atr_series

def timeframe_to_ms(tf: str) -> int:
    m = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000
    }
    if tf not in m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return m[tf]

def _read_signals_pending(conn: sqlite3.Connection, exchange: str) -> pd.DataFrame:
    q = """
    SELECT id, ts_ms, exchange, symbol, timeframe, direction, entry, sl, tp1, tp2, status, tp1_hit
    FROM signals
    WHERE exchange=? AND status='PENDING'
    ORDER BY ts_ms ASC
    """
    return pd.read_sql_query(q, conn, params=(exchange,))

def _read_candles_since(conn: sqlite3.Connection, exchange: str, symbol: str, timeframe: str, ts_from: int, limit: int = 5000) -> pd.DataFrame:
    q = """
    SELECT ts_ms, open, high, low, close, volume
    FROM ohlcv
    WHERE exchange=? AND symbol=? AND timeframe=? AND ts_ms>=?
    ORDER BY ts_ms ASC
    LIMIT ?
    """
    return pd.read_sql_query(q, conn, params=(exchange, symbol, timeframe, ts_from, limit))

def _first_touch_index(df: pd.DataFrame, price: float) -> Optional[int]:
    # zwraca indeks pierwszej świecy, której high/low obejmuje price
    for i, row in df.iterrows():
        if row["low"] <= price <= row["high"]:
            return i
    return None

def _pnl_pct(entry: float, exit_price: float, direction: str) -> float:
    if direction == "long":
        return (exit_price / entry - 1.0) * 100.0
    else:
        return (entry / exit_price - 1.0) * 100.0

def _decide_on_bar(row, direction: str, lvl_a: float, lvl_b: float, prefer: str) -> Optional[str]:
    """
    Sprawdza, czy na świecy padł poziom A lub B. Zwraca "A" lub "B" przy konflikcie
    zgodnie z preferencją (np. prefer='B' dla konserwatywnego wyboru stopa).
    """
    hit_a = row["high"] >= lvl_a if direction == "long" else row["low"] <= lvl_a
    hit_b = row["low"]  <= lvl_b if direction == "long" else row["high"] >= lvl_b
    if hit_a and hit_b:
        return prefer
    if hit_a:
        return "A"
    if hit_b:
        return "B"
    return None

def simulate_and_update(conn: sqlite3.Connection, cfg: dict):
    exchange = cfg["exchange"]["id"]
    validity = int(cfg["signals"]["validity_candles"])
    bar_policy = cfg["signals"].get("bar_policy", "conservative")
    prefer_stop = "B"  # w _decide_on_bar: A=TP, B=SL/trail
    if bar_policy != "conservative":
        prefer_stop = "A"

    usd_per_trade = float(cfg["signals"]["evaluation"]["usd_per_trade"])
    tp1_fraction = float(cfg["execution"]["tp1_fraction"])
    move_be = bool(cfg["execution"]["move_sl_to_be_on_tp1"])
    trail_use = bool(cfg["execution"]["trail"]["use"])
    trail_method = cfg["execution"]["trail"]["method"]
    trail_atr_period = int(cfg["execution"]["trail"]["atr_period"])
    trail_atr_mult = float(cfg["execution"]["trail"]["atr_mult"])

    sigs = _read_signals_pending(conn, exchange)
    if sigs.empty:
        return

    for _, s in sigs.iterrows():
        symbol = s["symbol"]; tf = s["timeframe"]; direction = s["direction"]
        entry = float(s["entry"]); sl_init = float(s["sl"])
        tp1 = float(s["tp1"]); tp2 = float(s["tp2"])
        ts_from = int(s["ts_ms"])

        df = _read_candles_since(conn, exchange, symbol, tf, ts_from)
        if df.empty or len(df) < 2:
            continue

        # ATR do traila (liczymy na całym df; można przyciąć po entry)
        df_calc = df.copy()
        df_calc["atr"] = atr_series(df_calc, period=trail_atr_period)

        # 1) Czekamy na wejście (od kolejnej świecy)
        entry_idx = _first_touch_index(df.iloc[1:], entry)
        if entry_idx is None or entry_idx > validity:
            # EXPIRED
            conn.execute(
                "UPDATE signals SET status='EXPIRED', opened_ts_ms=NULL, closed_ts_ms=?, exit_price=NULL, pnl_usd=0, pnl_pct=0, tp1_hit=0, exit_reason='EXPIRED' WHERE id=?",
                (int(df["ts_ms"].iloc[min(validity, len(df)-1)]), int(s["id"]))
            )
            continue

        opened_ts = int(df["ts_ms"].iloc[entry_idx])

        # 2) Po wejściu: najpierw faza do TP1/SL
        sub = df.iloc[entry_idx:]  # włącznie z entry-bar
        sl = sl_init
        tp1_done = False
        exit_reason = None
        part_pnl_pct = 0.0  # PnL z części TP1

        for i, row in sub.iterrows():
            # sprawdź TP1 vs SL
            dec = _decide_on_bar(row, direction, lvl_a=tp1, lvl_b=sl, prefer=prefer_stop)
            if dec is None:
                # nic, idziemy dalej
                continue

            if dec == "B":
                # SL przed TP1
                final_pct = _pnl_pct(entry, sl, direction)
                pnl_usd = usd_per_trade * (final_pct / 100.0)
                conn.execute("""
                    UPDATE signals
                    SET status='SL', opened_ts_ms=?, closed_ts_ms=?, exit_price=?, pnl_usd=?, pnl_pct=?, tp1_hit=0, exit_reason='SL'
                    WHERE id=?
                """, (opened_ts, int(row["ts_ms"]), float(sl), float(pnl_usd), float(final_pct), int(s["id"])))
                exit_reason = "SL"
                break

            # TP1 padł — częściowa realizacja
            tp1_done = True
            tp1_pct = _pnl_pct(entry, tp1, direction)
            part_pnl_pct = tp1_fraction * tp1_pct
            # po TP1: BE oraz trailing start (opcjonalnie)
            if move_be:
                sl = entry

            # 3) Faza po TP1: TP2 vs TRAIL/BE
            # trailing stop startowy:
            trail_level = None
            if trail_use and trail_method == "atr":
                atr_here = float(df_calc.loc[df_calc["ts_ms"] == row["ts_ms"], "atr"].values[0])
                if direction == "long":
                    trail_level = row["high"] - trail_atr_mult * atr_here
                else:
                    trail_level = row["low"] + trail_atr_mult * atr_here
                # nie pozwól trailowi być "gorszym" niż BE/SL
                trail_level = max(trail_level, sl) if direction == "long" else min(trail_level, sl)

            # przechodzimy do kolejnych świec po TP1 (włącznie z tą samą już obsłużoną)
            sub2 = df.iloc[i:]  # od świecy TP1
            closed = False
            for j, row2 in sub2.iterrows():
                # aktualizuj trailing
                if trail_use and trail_method == "atr":
                    atr_val = float(df_calc.loc[df_calc["ts_ms"] == row2["ts_ms"], "atr"].values[0])
                    if direction == "long":
                        candidate = row2["high"] - trail_atr_mult * atr_val
                        if trail_level is None:
                            trail_level = candidate
                        else:
                            trail_level = max(trail_level, candidate)
                        # BE jako minimum
                        trail_level = max(trail_level, sl)
                    else:
                        candidate = row2["low"] + trail_atr_mult * atr_val
                        if trail_level is None:
                            trail_level = candidate
                        else:
                            trail_level = min(trail_level, candidate)
                        trail_level = min(trail_level, sl)

                # zdecyduj: TP2 (A) vs TRAIL/BE (B)
                level_b = trail_level if trail_use else sl
                dec2 = _decide_on_bar(row2, direction, lvl_a=tp2, lvl_b=level_b, prefer=prefer_stop)

                if dec2 is None:
                    continue

                if dec2 == "B":
                    # trafiony trailing/BE
                    exit_px = level_b
                    rest_pct = (1.0 - tp1_fraction) * _pnl_pct(entry, exit_px, direction)
                    final_pct = part_pnl_pct + rest_pct
                    pnl_usd = usd_per_trade * (final_pct / 100.0)
                    conn.execute("""
                        UPDATE signals
                        SET status='TP1_TRAIL', opened_ts_ms=?, closed_ts_ms=?, exit_price=?, pnl_usd=?, pnl_pct=?, tp1_hit=1, exit_reason='TP1_TRAIL'
                        WHERE id=?
                    """, (opened_ts, int(row2["ts_ms"]), float(exit_px), float(pnl_usd), float(final_pct), int(s["id"])))
                    exit_reason = "TP1_TRAIL"
                    closed = True
                    break
                else:
                    # TP2 trafiony
                    rest_pct = (1.0 - tp1_fraction) * _pnl_pct(entry, tp2, direction)
                    final_pct = part_pnl_pct + rest_pct
                    pnl_usd = usd_per_trade * (final_pct / 100.0)
                    conn.execute("""
                        UPDATE signals
                        SET status='TP', opened_ts_ms=?, closed_ts_ms=?, exit_price=?, pnl_usd=?, pnl_pct=?, tp1_hit=1, exit_reason='TP'
                        WHERE id=?
                    """, (opened_ts, int(row2["ts_ms"]), float(tp2), float(pnl_usd), float(final_pct), int(s["id"])))
                    exit_reason = "TP"
                    closed = True
                    break

            if not closed:
                # nie trafił TP2 ani trail w oknie ważności ⇒ EXPIRED (po TP1)
                last_i = min(i + validity, len(df) - 1)
                # traktujemy jak zamknięcie reszty po cenie close ostatniej świecy
                exit_px = float(df["close"].iloc[last_i])
                rest_pct = (1.0 - tp1_fraction) * _pnl_pct(entry, exit_px, direction)
                final_pct = part_pnl_pct + rest_pct
                pnl_usd = usd_per_trade * (final_pct / 100.0)
                conn.execute("""
                    UPDATE signals
                    SET status='EXPIRED', opened_ts_ms=?, closed_ts_ms=?, exit_price=?, pnl_usd=?, pnl_pct=?, tp1_hit=1, exit_reason='EXPIRED'
                    WHERE id=?
                """, (opened_ts, int(df["ts_ms"].iloc[last_i]), float(exit_px), float(pnl_usd), float(final_pct), int(s["id"])))
                exit_reason = "EXPIRED"
            break  # zakończ pętlę po sub (mamy TP1 już obsłużone)

        # jeżeli pętla zakończyła się bez żadnego break i bez exit_reason → nic nie wydarzyło się w oknie
        if exit_reason is None:
            last_i = min(entry_idx + validity, len(df) - 1)
            conn.execute("""
                UPDATE signals
                SET status='EXPIRED', opened_ts_ms=?, closed_ts_ms=?, exit_price=NULL, pnl_usd=0, pnl_pct=0, tp1_hit=0, exit_reason='EXPIRED'
                WHERE id=?
            """, (opened_ts, int(df["ts_ms"].iloc[last_i]), int(s["id"])))
