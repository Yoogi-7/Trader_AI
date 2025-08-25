from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Fees:
    taker_pct: float = 0.06  # w %
    maker_pct: float = 0.02  # w %
    assume_maker: bool = False

def pct_move(a: float, b: float) -> float:
    return abs(b - a) / a * 100.0

def roundtrip_fees_pct(fees: Fees) -> float:
    f = fees.maker_pct if fees.assume_maker else fees.taker_pct
    return 2.0 * f

def tp_net_pct(entry: float, target: float, fees: Fees, slippage_pct: float) -> float:
    gross = pct_move(entry, target)
    net = gross - roundtrip_fees_pct(fees) - slippage_pct
    return net

def sizing_and_leverage(equity: float, risk_pct: float, entry: float, sl: float,
                        max_leverage: float, liquidation_buffer: float) -> tuple[float, float]:
    """
    Zwraca (position_notional, leverage)
    Ryzyko $ = (risk_pct% * equity). Poziom SL wyraża stratę % ruchu ceny.
    position_notional tak, by strata przy SL ~= risk_amount (z buforem).
    """
    sl_pct = pct_move(entry, sl)  # w %
    risk_amount = (risk_pct / 100.0) * equity
    denom = (sl_pct / 100.0) * max(liquidation_buffer, 1.0)
    if denom <= 0:
        return 0.0, 0.0
    position_notional = risk_amount / denom
    leverage = min(max_leverage, position_notional / equity) if equity > 0 else 0.0
    return position_notional, leverage
