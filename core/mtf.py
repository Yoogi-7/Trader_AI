from __future__ import annotations
import pandas as pd
from typing import Dict, List
from .indicators import add_indicators

def _align_htf(htf_df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if htf_df.empty:
        return None
    idx = htf_df.index.searchsorted(ts, side='right') - 1
    if idx < 0:
        return None
    return htf_df.iloc[idx]

def _check_long(row: pd.Series, prev_row: pd.Series | None, rules: Dict) -> bool:
    ok = True
    if rules.get('price_above_ema200'):
        ok &= row['close'] > row['ema200']
    if rules.get('ema200_slope_gt0'):
        if prev_row is None: return False
        ok &= (row['ema200'] - prev_row['ema200']) > 0
    if rules.get('macd_hist_positive'):
        ok &= row.get('macd_hist', 0) > 0
    if 'rsi_min' in rules:
        ok &= row.get('rsi', 0) >= float(rules['rsi_min'])
    return bool(ok)

def _check_short(row: pd.Series, prev_row: pd.Series | None, rules: Dict) -> bool:
    ok = True
    if rules.get('price_below_ema200'):
        ok &= row['close'] < row['ema200']
    if rules.get('ema200_slope_lt0'):
        if prev_row is None: return False
        ok &= (row['ema200'] - prev_row['ema200']) < 0
    if rules.get('macd_hist_negative'):
        ok &= row.get('macd_hist', 0) < 0
    if 'rsi_max' in rules:
        ok &= row.get('rsi', 100) <= float(rules['rsi_max'])
    return bool(ok)

def build_htf_context(symbol: str, exchange: str, tfs: List[str], limit: int, fetch_fn) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        df = fetch_fn(symbol, tf, exchange, limit)
        out[tf] = add_indicators(df) if not df.empty else df
    return out

def confirm_signal(ts: pd.Timestamp, direction: str, htf_ctx: Dict[str, pd.DataFrame], rules_cfg: Dict) -> bool:
    for tf, hdf in htf_ctx.items():
        if hdf is None or hdf.empty:
            return False
        row = _align_htf(hdf, ts)
        if row is None:
            return False
        prev_row = None
        idx = hdf.index.get_loc(row.name)
        if isinstance(idx, int) and idx > 0:
            prev_row = hdf.iloc[idx-1]
        if direction == 'long':
            if not _check_long(row, prev_row, rules_cfg.get('long', {})):
                return False
        else:
            if not _check_short(row, prev_row, rules_cfg.get('short', {})):
                return False
    return True
