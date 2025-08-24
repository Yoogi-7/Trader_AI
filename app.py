from __future__ import annotations
import argparse, json, os, time, random
import pandas as pd
import yaml
from rich import print
from rich.table import Table

from core.data import fetch_ohlcv_10m, fetch_ohlcv
from core.indicators import add_indicators
from core.signals import generate_signals
from core.backtest import simulate, summarize
from core.risk import size_and_leverage
from core.utils import wilson_interval
from core.mtf import build_htf_context
from core.ml import infer_proba

def _dbg(msg: str, cfg: dict):
    if cfg.get('debug', {}).get('enabled', False):
        print(msg)

def _debug_counts(symbol: str, df: pd.DataFrame, cfg: dict):
    if not cfg.get('debug', {}).get('enabled', False):
        return
    try:
        ema_up  = (df['ema21'].shift(1) < df['ema50'].shift(1)) & (df['ema21'] > df['ema50'])
        ema_dn  = (df['ema21'].shift(1) > df['ema50'].shift(1)) & (df['ema21'] < df['ema50'])
        macd_up = (df['macd_hist'].shift(1) < 0) & (df['macd_hist'] > 0)
        macd_dn = (df['macd_hist'].shift(1) > 0) & (df['macd_hist'] < 0)
        rsi_up  = (df['rsi'].shift(1) < 35) & (df['rsi'] > 35)
        rsi_dn  = (df['rsi'].shift(1) > 65) & (df['rsi'] < 65)
        brk_lb  = int(cfg.get('signals', {}).get('breakout_lookback', 30))
        hh = df['high'].rolling(brk_lb, min_periods=1).max()
        ll = df['low'].rolling(brk_lb, min_periods=1).min()
        brk_up = (df['close'] > hh.shift(1))
        brk_dn = (df['close'] < ll.shift(1))
        cnts = {
            'candles': len(df),
            'ema_up': int(ema_up.sum()),
            'ema_dn': int(ema_dn.sum()),
            'macd_up': int(macd_up.sum()),
            'macd_dn': int(macd_dn.sum()),
            'rsi_up': int(rsi_up.sum()),
            'rsi_dn': int(rsi_dn.sum()),
            'brk_up': int(brk_up.sum()),
            'brk_dn': int(brk_dn.sum()),
        }
        print(f"[cyan]{symbol} DEBUG[/cyan]: {cnts}")
    except Exception as e:
        print(f"[yellow]DEBUG error for {symbol}: {e}[/yellow]")

def run_symbol(symbol: str, cfg: dict) -> dict:
    df_raw = fetch_ohlcv_10m(symbol, cfg['exchange'], cfg['limit_10m'])
    _dbg(f"[magenta]{symbol} RAW candles: {len(df_raw)}[/magenta]", cfg)

    df = add_indicators(df_raw)
    _debug_counts(symbol, df, cfg)

    htf_ctx = {}
    mtf_cfg = cfg.get('mtf', {})
    if mtf_cfg.get('enabled', False) and mtf_cfg.get('timeframes'):
        htf_ctx = build_htf_context(symbol, cfg['exchange'], mtf_cfg['timeframes'], 600, fetch_ohlcv)

    sigs = generate_signals(df, cfg, htf_ctx if htf_ctx else None)
    if not sigs:
        return {'symbol': symbol, 'timeframe': cfg['timeframe'], 'latest_signal': None, 'probability': None}

    results = simulate(df, sigs, cfg['backtest'])
    stats = summarize(results)

    trials = stats['wins'] + stats['losses']
    p_hat, lo, hi = wilson_interval(stats['wins'], trials) if trials > 0 else (0.0, 0.0, 0.0)

    latest = sigs[-1]
    risk_cfg = cfg['risk']
    sizing = size_and_leverage(
        latest['entry'], latest['sl'],
        risk_cfg['equity_usdt'], risk_cfg['risk_pct'],
        risk_cfg['max_leverage']
    )

    ml_p = None
    if cfg.get('ml', {}).get('enabled', True) and latest:
        mtf_ok = True
        if htf_ctx:
            from core.mtf import confirm_signal
            mtf_ok = confirm_signal(pd.Timestamp(latest['time']), latest['direction'], htf_ctx, mtf_cfg.get('rules', {}))
        ml_p = infer_proba(cfg, df, pd.Timestamp(latest['time']), latest['direction'], bool(mtf_ok))

    prob = {'p_win': p_hat, 'ci95_lo': lo, 'ci95_hi': hi, 'ml_p_win': ml_p}

    return {
        'symbol': symbol,
        'timeframe': cfg['timeframe'],
        'latest_signal': latest,
        'probability': prob,
        'backtest': stats,
        'sizing': sizing,
    }

def save_reports(rows: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    flat = []
    for r in rows:
        base = {'symbol': r['symbol'], 'timeframe': r['timeframe']}
        if r['probability']:
            base.update({
                'p_win': r['probability']['p_win'],
                'p_lo': r['probability']['ci95_lo'],
                'p_hi': r['probability']['ci95_hi'],
                'ml_p_win': r['probability']['ml_p_win']
            })
        if r['latest_signal']:
            sig = r['latest_signal']
            base.update({
                'signal_time': sig['time'],
                'direction': sig['direction'],
                'entry': sig['entry'],
                'sl': sig['sl'],
                'tp': sig['tp']
            })
        if r.get('sizing'):
            base.update({
                'qty': r['sizing']['qty'],
                'leverage': r['sizing']['leverage'],
                'notional': r['sizing']['notional'],
                'risk_usdt': r['sizing']['risk_usdt']
            })
        flat.append(base)
    pd.DataFrame(flat).to_csv(os.path.join(out_dir, 'report.csv'), index=False)

def print_table(rows: list):
    t = Table(title='10m Screener + MTF + ML (Windows local)')
    t.add_column('Symbol'); t.add_column('Kier.'); t.add_column('Czas')
    t.add_column('Entry'); t.add_column('SL'); t.add_column('TP')
    t.add_column('p(win) hist'); t.add_column('p(win) ML'); t.add_column('Lev'); t.add_column('Qty')
    for r in rows:
        sym = r['symbol']
        if not r['latest_signal']:
            t.add_row(sym, '-', '-', '-', '-', '-', '-', '-', '-', '-')
            continue
        sig = r['latest_signal']
        prob = r['probability'] or {}
        ph = float(prob.get('p_win') or 0.0)
        pm = float(prob.get('ml_p_win') or 0.0)
        t.add_row(sym, sig['direction'], sig['time'].replace('T',' ')[:19],
                  f"{sig['entry']:.6f}", f"{sig['sl']:.6f}", f"{sig['tp']:.6f}",
                  f"{ph*100:.1f}%", f"{pm*100:.1f}%", f"{r['sizing']['leverage']:.1f}", f"{r['sizing']['qty']:.6f}")
    print(t)

def scan_once(cfg: dict) -> list:
    rows = []
    for sym in cfg['symbols']:
        try:
            rows.append(run_symbol(sym, cfg))
        except Exception as e:
            print(f"[red]Błąd dla {sym}: {e}[/red]")   # FIX: było print(f(...))
    return rows

def with_lock(lockfile: str):
    import os, time
    class _Lock:
        def __init__(self, path): self.path = path; self.fd = None
        def __enter__(self):
            for _ in range(10):
                try:
                    self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    os.write(self.fd, str(os.getpid()).encode())
                    return self
                except FileExistsError:
                    time.sleep(0.5)
            raise RuntimeError("Nie mogę zdobyć locka (inny proces w trakcie).")
        def __exit__(self, exc_type, exc, tb):
            try:
                if self.fd: os.close(self.fd)
                if os.path.exists(self.path): os.remove(self.path)
            except Exception:
                pass
    return _Lock(lockfile)

def main_once(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    rows = scan_once(cfg)
    print_table(rows)
    save_reports(rows, cfg['report']['out_dir'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--loop', action='store_true', help='Włącz pętlę skanu wg ustawień scheduler.* z config.yaml')
    args = ap.parse_args()

    if not args.loop:
        main_once(args.config)
    else:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        interval = max(1, int(cfg.get('scheduler', {}).get('interval_minutes', 3))) * 60
        jitter = int(cfg.get('scheduler', {}).get('jitter_seconds', 0))
        lockfile = cfg.get('scheduler', {}).get('lockfile', '.scan.lock')
        print(f"[green]Pętla skanu aktywna co {interval//60} min (+/- {jitter}s). Ctrl+C aby przerwać.[/green]")
        try:
            while True:
                try:
                    with with_lock(lockfile):
                        rows = scan_once(cfg)
                        print_table(rows)
                        save_reports(rows, cfg['report']['out_dir'])
                except RuntimeError as e:
                    print(f"[yellow]{e} – pomijam ten cykl.[/yellow]")
                sleep_s = interval + (random.randint(-jitter, jitter) if jitter > 0 else 0)
                time.sleep(max(1, sleep_s))
        except KeyboardInterrupt:
            print("[red]Przerwano pętlę.[/red]")
