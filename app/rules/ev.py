"""
Risk sizing and Expected Value (EV) utilities.
- Position sizing uses risk-per-trade and SL distance.
- EV includes both taker fees (in+out) and slippage.
"""
from dataclasses import dataclass

@dataclass
class Costs:
    taker_bps_in: float
    taker_bps_out: float
    slippage_pct: float

def position_notional(equity: float, risk: float, entry: float, sl: float, lmax: float) -> float:
    d_sl = abs(entry - sl) / entry
    if d_sl <= 0:
        return 0.0
    notional = (equity * risk) / d_sl
    return min(notional, equity * lmax)

def trade_costs(notional: float, c: Costs) -> tuple[float, float]:
    fee = notional * ((c.taker_bps_in + c.taker_bps_out) / 10000.0)
    slip = notional * c.slippage_pct
    return fee, slip

def net_profit_at_tp(entry: float, tp: float, notional: float, fee: float, slip: float) -> float:
    gross = notional * (abs(tp - entry) / entry)
    return gross - fee - slip

def passes_min_profit(net_profit: float, notional: float, min_per_100: float) -> bool:
    # e.g. min_per_100=2.0 means ≥2$ per each $100 notional => ≥2%
    return net_profit >= (min_per_100 / 100.0) * notional

def expected_value(p_hit: float, entry: float, tp: float, sl: float, notional: float, fee: float, slip: float) -> float:
    gain = notional * (abs(tp - entry) / entry) - fee - slip
    loss = notional * (abs(entry - sl) / entry) + fee + slip
    return p_hit * gain - (1.0 - p_hit) * loss
