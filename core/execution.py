# core/execution.py
"""
Egzekucja transakcji ze wsparciem:
- latency_bar: wejście na otwarciu następnej świecy po sygnale,
- slippage_ticks * tick_size: korekta fill-a na wejściu i wyjściu,
- fee_bp: prowizja w basis points (1 bp = 0.01%) za stronę,
- Priorytet: SL -> TP przy zdarzeniach na tej samej świecy,
- trailing stop (opcjonalny, na bazie ATR lub stałej odległości),
- time_stop (horyzont bars).

Wejście:
- df OHLCV z ['timestamp','open','high','low','close','volume'].
- sygnały jako DataFrame z indeksami/kolumną 'idx' wskazującą bar sygnału oraz:
  ['side','tp','sl','horizon_bars'] — jeśli brak, można użyć wartości domyślnych z configu.
- tick_size — minimalny krok ceny (np. 0.1$ dla BTC w niektórych marketach),
- contract_value — wartość 1 kontraktu na 1$ ruchu (dla spot = 1.0).

Zwraca:
- trades: lista słowników z pełnym dziennikiem transakcji.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ExecConfig:
    latency_bar: int = 1
    fee_bp: float = 1.0            # 1 bp = 0.01% -> 1.0 = 0.01% (per side)
    slippage_ticks: int = 1
    tick_size: float = 0.1
    contract_value: float = 1.0    # $ PnL per 1 qty per $ ruchu
    use_trailing: bool = False
    trailing_atr_mult: float = 2.0
    atr_period: int = 14
    time_stop_bars: int | None = None  # jeśli None, użyj z sygnału (horizon_bars)

def _apply_slippage(price: float, side: str, slippage_ticks: int, tick_size: float, is_exit: bool = False) -> float:
    slip = slippage_ticks * tick_size
    if side == "long":
        return price + slip if not is_exit else price - slip
    else:  # short
        return price - slip if not is_exit else price + slip

def _fee_amount(price: float, qty: float, fee_bp: float) -> float:
    # fee_bp = 1.0 -> 0.01% -> 0.0001
    return price * qty * (fee_bp * 0.0001)

def position_size(entry: float, sl: float, capital: float = 100.0, risk_pct: float = 0.01, contract_value: float = 1.0) -> float:
    risk_amt = capital * risk_pct
    stop_dist = abs(entry - sl)
    if stop_dist <= 0:
        return 0.0
    # ile qty daje ryzyko ~ risk_amt przy ruchu do SL
    qty = (risk_amt / stop_dist) / contract_value
    return max(0.0, qty)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(period, min_periods=1).mean()

def backtest_trades(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: ExecConfig,
    capital_ref: float = 100.0,
    risk_pct: float = 0.01,
) -> list[dict]:
    df = df.reset_index(drop=True).copy()
    atr = _atr(df, cfg.atr_period) if cfg.use_trailing else None
    trades: list[dict] = []

    for _, s in signals.iterrows():
        i = int(s["idx"])  # index świecy sygnału
        side = str(s.get("side", "long")).lower()
        tp = float(s["tp"]) if "tp" in s else np.nan
        sl = float(s["sl"]) if "sl" in s else np.nan
        horizon = int(s["horizon_bars"]) if "horizon_bars" in s else (cfg.time_stop_bars or 60)

        # Wejście na kolejnej świecy (latency)
        entry_bar = i + cfg.latency_bar
        if entry_bar >= len(df):
            continue

        raw_entry = float(df.loc[entry_bar, "open"])
        entry = _apply_slippage(raw_entry, side, cfg.slippage_ticks, cfg.tick_size, is_exit=False)
        qty = position_size(entry, sl, capital=capital_ref, risk_pct=risk_pct, contract_value=cfg.contract_value)
        if qty == 0.0 or math.isinf(qty) or math.isnan(qty):
            continue

        # symulacja bar po barze do horyzontu (włącznie)
        j_end = min(entry_bar + horizon, len(df) - 1)
        exit_idx = None
        exit_reason = "horizon"
        exit_price = float(df.loc[j_end, "close"])  # domyślnie time-stop po close
        trail_sl = sl

        for j in range(entry_bar, j_end + 1):
            high = float(df.loc[j, "high"])
            low = float(df.loc[j, "low"])

            # trailing stop aktualizacja na końcu poprzedniej świecy (konserwatywnie)
            if cfg.use_trailing and j > entry_bar:
                if side == "long":
                    new_sl = float(df.loc[j - 1, "close"]) - float(atr.iloc[j - 1]) * cfg.trailing_atr_mult
                    trail_sl = max(trail_sl, new_sl)
                else:
                    new_sl = float(df.loc[j - 1, "close"]) + float(atr.iloc[j - 1]) * cfg.trailing_atr_mult
                    trail_sl = min(trail_sl, new_sl)

            # kolejność zdarzeń: SL -> TP
            if side == "long":
                # sprawdź SL (trail lub stały)
                curr_sl = min(trail_sl, sl) if cfg.use_trailing else sl
                if low <= curr_sl:
                    exit_idx = j
                    exit_reason = "sl"
                    exit_raw = curr_sl
                    break
                if not np.isnan(tp) and high >= tp:
                    exit_idx = j
                    exit_reason = "tp"
                    exit_raw = tp
                    break
            else:  # short
                curr_sl = max(trail_sl, sl) if cfg.use_trailing else sl
                if high >= curr_sl:
                    exit_idx = j
                    exit_reason = "sl"
                    exit_raw = curr_sl
                    break
                if not np.isnan(tp) and low <= tp:
                    exit_idx = j
                    exit_reason = "tp"
                    exit_raw = tp
                    break

        if exit_idx is not None:
            exit_fill = _apply_slippage(exit_raw, side, cfg.slippage_ticks, cfg.tick_size, is_exit=True)
        else:
            exit_idx = j_end
            exit_reason = "horizon"
            exit_raw = float(df.loc[exit_idx, "close"])
            exit_fill = _apply_slippage(exit_raw, side, cfg.slippage_ticks, cfg.tick_size, is_exit=True)

        # PnL brutto
        price_move = (exit_fill - entry) if side == "long" else (entry - exit_fill)
        gross_pnl = price_move * qty * cfg.contract_value

        # opłaty — wejście i wyjście
        fee_in = _fee_amount(entry, qty, cfg.fee_bp)
        fee_out = _fee_amount(exit_fill, qty, cfg.fee_bp)
        net_pnl = gross_pnl - fee_in - fee_out

        trades.append({
            "signal_idx": i,
            "entry_idx": entry_bar,
            "exit_idx": exit_idx,
            "timestamp_entry": df.loc[entry_bar, "timestamp"],
            "timestamp_exit": df.loc[exit_idx, "timestamp"],
            "side": side,
            "qty": qty,
            "entry": entry,
            "exit": exit_fill,
            "tp": tp,
            "sl": sl,
            "exit_reason": exit_reason,
            "gross_pnl": gross_pnl,
            "fee_in": fee_in,
            "fee_out": fee_out,
            "net_pnl": net_pnl,
            "bars_held": int(exit_idx - entry_bar),
        })

    return trades
