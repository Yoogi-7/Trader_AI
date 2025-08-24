from __future__ import annotations
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from .indicators import add_indicators
from .mtf import confirm_signal

# -------------------------------
# Ulepszony generator sygnałów
# -------------------------------
# Założenia poprawiające jakość:
# 1) Filtr trendu: cena > EMA200 (long) / < EMA200 (short) + nachylenie EMA200 zgodne z kierunkiem.
# 2) Reżim zmienności: ATR w percentylu [atr_pct_low, atr_pct_high].
# 3) Filtr wolumenu: z-score wolumenu >= vol_z_min.
# 4) Momentum guard: MACD histogram w kierunku sygnału lub RSI w strefie sprzyjającej.
# 5) Minimalny edge: RR >= rr_min_keep do TP wyliczanego z docelowego rr_target i SL = ATR * atr_sl_mult.
# 6) Opcjonalne MTF (potwierdzenia HTF).
# 7) Opcjonalny ML-score z progiem min_prob.
#
# Wynik: lista dictów {time, direction, entry, sl, tp, rr, prob, filters}

def _rr(entry: float, sl: float, tp: float, direction: str) -> float:
    if direction == 'long':
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    return reward / risk if risk > 0 else 0.0

def _atr_percentile(atr_series: pd.Series, window: int = 200) -> pd.Series:
    roll = atr_series.rolling(window=window, min_periods=10)
    # percentyl z rankingu ruchomego (ostatnia próbka)
    return roll.apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) else np.nan,
        raw=False
    )

def _engulfing(df: pd.DataFrame, min_body_atr: float = 0.25) -> pd.Series:
    body = (df['close'] - df['open']).abs()
    prev_body = (df['close'].shift(1) - df['open'].shift(1)).abs()
    atr = df['atr'].replace(0, np.nan)

    bull = (df['close'] > df['open']) & (df['close'] >= df['open'].shift(1)) & (df['open'] <= df['close'].shift(1))
    bear = (df['close'] < df['open']) & (df['close'] <= df['open'].shift(1)) & (df['open'] >= df['close'].shift(1))
    strong = (body >= min_body_atr * atr) & (body > prev_body)

    return pd.Series(np.select([bull & strong, bear & strong], [1, -1], default=0), index=df.index)

def _pinbar(df: pd.DataFrame, k: float = 2.0, min_body_atr: float = 0.1) -> pd.Series:
    body = (df['close'] - df['open']).abs()
    lower_wick = df['open'].combine(df['close'], min) - df['low']
    upper_wick = df['high'] - df['open'].combine(df['close'], max)
    atr = df['atr'].replace(0, np.nan)

    bull = (lower_wick > k*body) & (lower_wick > upper_wick) & (body >= min_body_atr*atr)
    bear = (upper_wick > k*body) & (upper_wick > lower_wick) & (body >= min_body_atr*atr)

    return pd.Series(np.select([bull, bear], [1, -1], default=0), index=df.index)

def _trend_ok(df: pd.DataFrame, direction: str) -> pd.Series:
    ema_up = (df['close'] > df['ema200']) & (df['ema200'].diff() > 0)
    ema_dn = (df['close'] < df['ema200']) & (df['ema200'].diff() < 0)
    return ema_up if direction == 'long' else ema_dn

def _momentum_ok(df: pd.DataFrame, direction: str) -> pd.Series:
    if direction == 'long':
        return (df.get('macd_hist', 0) >= 0) | (df.get('rsi', 50) >= 45)
    else:
        return (df.get('macd_hist', 0) <= 0) | (df.get('rsi', 50) <= 55)

def _build_levels(row, rr_target: float, atr_mult_sl: float):
    price = float(row['close'])
    atr = float(row.get('atr', price * 0.003))
    if rr_target <= 0:
        rr_target = 2.0
    if atr_mult_sl <= 0:
        atr_mult_sl = 1.0

    sl_long = price - atr_mult_sl * atr
    sl_short = price + atr_mult_sl * atr
    tp_long = price + rr_target * (price - sl_long)
    tp_short = price - rr_target * (sl_short - price)
    return sl_long, tp_long, sl_short, tp_short

def generate_signals(df: pd.DataFrame, cfg: Dict, htf_ctx: Optional[Dict] = None) -> List[Dict]:
    """
    Zwraca listę sygnałów: dict z kluczami:
      time, direction, entry, sl, tp, rr, prob (opcjonalnie), filters{...}
    """
    if df is None or df.empty:
        return []

    sig_cfg = cfg.get('signals', {})
    mtf_cfg = cfg.get('mtf', {})
    ml_cfg  = cfg.get('ml',  {})

    rr_target   = float(sig_cfg.get('rr_target', 2.0))
    atr_sl_mult = float(sig_cfg.get('atr_sl_mult', 1.0))
    atr_p_low   = float(sig_cfg.get('atr_pct_low', 0.20))
    atr_p_high  = float(sig_cfg.get('atr_pct_high', 0.90))
    vol_z_min   = float(sig_cfg.get('vol_z_min', -0.2))
    use_engulf  = bool(sig_cfg.get('use_engulfing', True))
    use_pin     = bool(sig_cfg.get('use_pinbar', True))
    rr_min_keep = float(sig_cfg.get('rr_min_keep', 1.2))
    min_gap_bps = float(sig_cfg.get('min_gap_bps', 0.0))

    use_mtf = bool(mtf_cfg.get('enabled', False))
    rules_cfg = mtf_cfg.get('rules', {})

    use_ml   = bool(ml_cfg.get('enabled', False))
    ml_p_min = float(ml_cfg.get('min_prob', 0.55))

    # Normalizacja nagłówków i wskaźniki
    df = add_indicators(
        df.rename(columns={c: c.lower() for c in df.columns}),
        ema21=21, ema50=50, ema200=200, rsi_len=14, macd=(12, 26, 9), atr_len=14
    )
    df['atr_pct'] = _atr_percentile(df['atr']).clip(0, 1)

    # Sygnał bazowy = suma (engulfing + pinbar)
    sig_raw = pd.Series(0, index=df.index)
    if use_engulf:
        sig_raw = sig_raw.add(_engulfing(df), fill_value=0)
    if use_pin:
        sig_raw = sig_raw.add(_pinbar(df), fill_value=0)

    signals: List[Dict] = []

    for ts in df.index[1:]:
        row = df.loc[ts]
        base_sig = int(sig_raw.loc[ts])
        if base_sig == 0:
            continue

        direction = 'long' if base_sig > 0 else 'short'

        # Trend + momentum
        if not bool(_trend_ok(df.loc[:ts], direction).iloc[-1]):
            continue
        if not bool(_momentum_ok(df.loc[:ts], direction).iloc[-1]):
            continue

        # Reżim zmienności
        atr_pct = float(row['atr_pct'])
        if not (atr_p_low <= atr_pct <= atr_p_high):
            continue

        # Wolumen (z-score)
        if float(row.get('vol_z', 0.0)) < vol_z_min:
            continue

        # Minimalny dystans do EMA200 w bps (edge)
        ema200 = float(row.get('ema200', np.nan))
        if np.isfinite(ema200) and min_gap_bps > 0:
            gap_bps = abs((row['close'] - ema200) / ema200) * 1e4
            if gap_bps < min_gap_bps:
                continue

        # MTF (opcjonalnie)
        mtf_ok = True
        if use_mtf and htf_ctx is not None:
            try:
                mtf_ok = bool(confirm_signal(htf_ctx, ts, direction, rules_cfg))
            except Exception:
                mtf_ok = True
        if not mtf_ok:
            continue

        # Poziomy i RR
        sl_long, tp_long, sl_short, tp_short = _build_levels(row, rr_target, atr_sl_mult)
        if direction == 'long':
            entry, sl, tp = float(row['close']), float(sl_long), float(tp_long)
        else:
            entry, sl, tp = float(row['close']), float(sl_short), float(tp_short)

        rr = _rr(entry, sl, tp, direction)
        if rr < rr_min_keep or rr <= 0 or not np.isfinite(rr):
            continue

        # ML-score (opcjonalnie) — lokalny import, żeby uniknąć pętli importów
        prob = None
        if use_ml:
            try:
                from .ml import infer_proba  # lokalnie, brak cyklicznego importu
                prob = float(infer_proba(cfg, df, ts, direction, mtf_ok))
                if prob < ml_p_min:
                    continue
            except Exception:
                prob = None

        signals.append({
            'time': ts,
            'direction': direction,
            'entry': float(entry),
            'sl': float(sl),
            'tp': float(tp),
            'rr': float(rr),
            'prob': prob,
            'filters': {
                'trend': True,
                'momentum': True,
                'atr_pct': atr_pct,
                'vol_z': float(row.get('vol_z', 0.0)),
                'mtf': bool(mtf_ok),
            }
        })

    return signals
