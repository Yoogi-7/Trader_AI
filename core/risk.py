from __future__ import annotations
import math
from typing import Dict

def _edge_bps(entry: float, tp: float) -> float:
    # przewaga ceny w bps względem entry
    return (tp - entry) / entry * 1e4 if tp >= entry else (entry - tp) / entry * 1e4

def _risk_per_unit(entry: float, sl: float) -> float:
    return abs(entry - sl)

def _fees_for_roundtrip(notional: float, taker_bps: float, maker_bps: float, prefer_maker: bool) -> float:
    bps = (maker_bps if prefer_maker else taker_bps)
    # entry + exit
    return notional * (bps / 1e4) * 2.0

def _slippage_cost(notional: float, slippage_bps: float) -> float:
    # entry + exit
    return notional * (slippage_bps / 1e4) * 2.0

def size_and_leverage(entry: float, sl: float, equity_usdt: float, risk_pct: float,
                      max_leverage: float, fees_cfg: Dict | None = None, exec_cfg: Dict | None = None) -> Dict:
    """
    Zwraca rozmiar pozycji i dźwignię, uwzględniając:
      - ryzyko R (equity * risk_pct),
      - minimalny notional i krok ilości,
      - opłaty taker/maker i poślizg,
      - bufor marginesu (margin_buffer),
      - filtr edge vs fees (min_edge_bps).
    """
    fees_cfg = fees_cfg or {}
    exec_cfg = exec_cfg or {}
    taker_bps = float(fees_cfg.get('taker_bps', 6.0))
    maker_bps = float(fees_cfg.get('maker_bps', 2.0))
    slippage_bps = float(fees_cfg.get('slippage_bps', 1.5))
    min_notional = float(fees_cfg.get('min_notional_usdt', 5.0))
    qty_step = float(fees_cfg.get('min_qty_step', 0.0001))
    prefer_maker = bool(exec_cfg.get('prefer_maker', False))
    margin_buffer = float(exec_cfg.get('margin_buffer', 0.9)) if 'margin_buffer' in exec_cfg else 0.9

    risk_usdt = equity_usdt * risk_pct
    per_unit_risk = _risk_per_unit(entry, sl)
    if per_unit_risk <= 0:
        return {'qty': 0.0, 'leverage': 1.0, 'notional': 0.0, 'risk_usdt': 0.0,
                'fees_est': 0.0, 'slippage_est': 0.0, 'edge_bps': 0.0, 'fee_ratio': None}

    # ilość po ryzyku (bez dźwigni; dźwignia tylko zmienia wymagany depozyt, nie ryzyko)
    qty = risk_usdt / per_unit_risk
    # zaokrąglenie do kroku
    qty = math.floor(qty / qty_step) * qty_step
    qty = max(qty, qty_step)

    notional = qty * entry
    if notional < min_notional:
        # spróbuj podnieść ilość do minimalnego notionala, ale ryzyko wzrośnie – klient może to chcieć wiedzieć
        qty = max(qty, math.ceil(min_notional / entry / qty_step) * qty_step)
        notional = qty * entry
        # przeliczenie ryzyka dla informacji
        risk_usdt = per_unit_risk * qty

    # dźwignia = tak, by depozyt <= equity * margin_buffer
    # depozyt = notional / lev -> lev = notional / (equity * margin_buffer)
    min_lev = notional / max(1e-9, equity_usdt * margin_buffer)
    lev = max(1.0, min(min_lev, max_leverage))

    # estymacja kosztów
    fees_est = _fees_for_roundtrip(notional, taker_bps, maker_bps, prefer_maker)
    slipp_est = _slippage_cost(notional, slippage_bps)
    # edge (bez znaku)
    # Dashboard podaje TP, ale ta funkcja nie – edge_bps liczony będzie wyżej; tutaj zostaw 0 i niech caller wypełni jeśli ma TP.
    return {
        'qty': float(qty),
        'leverage': float(lev),
        'notional': float(notional),
        'risk_usdt': float(per_unit_risk * qty),
        'fees_est': float(fees_est),
        'slippage_est': float(slipp_est),
        'edge_bps': 0.0,
        'fee_ratio': None
    }

def fee_guard_ok(entry: float, tp: float, fees_cfg: Dict, exec_cfg: Dict) -> Dict:
    taker_bps = float(fees_cfg.get('taker_bps', 6.0))
    slippage_bps = float(fees_cfg.get('slippage_bps', 1.5))
    prefer_maker = bool(exec_cfg.get('prefer_maker', False))
    maker_bps = float(fees_cfg.get('maker_bps', 2.0))
    # minimalny edge jeśli nie zdefiniowano w configu: 2x opłata + 2x slippage + 5 bps bufor
    auto_min_edge = 2 * (maker_bps if prefer_maker else taker_bps) + 2 * slippage_bps + 5.0
    min_edge_cfg = exec_cfg.get('min_edge_bps', None)
    min_edge = float(min_edge_cfg) if min_edge_cfg is not None else float(auto_min_edge)
    edge = _edge_bps(entry, tp)
    return {
        'edge_bps': edge,
        'min_edge_bps': min_edge,
        'ok': edge >= min_edge
    }
