from __future__ import annotations
import pandas as pd
import ccxt
from rich import print
import time

def _make_exchange(exchange: str):
    ex = getattr(ccxt, exchange)({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    ex.load_markets()
    return ex

_RULE = {
    '1m': '1min', '3m': '3min', '5m': '5min', '10m': '10min', '15m': '15min', '30m': '30min',
    '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H', '1d': '1D'
}

def _resample_1m(df1: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _RULE.get(tf)
    if rule is None:
        raise ValueError(f"Nieobsługiwany TF do resamplingu: {tf}")
    out = pd.DataFrame()
    out['open']   = df1['open'].resample(rule).first()
    out['high']   = df1['high'].resample(rule).max()
    out['low']    = df1['low'].resample(rule).min()
    out['close']  = df1['close'].resample(rule).last()
    out['volume'] = df1['volume'].resample(rule).sum()
    return out.dropna()

def _candidate_symbols(ex: ccxt.Exchange, symbol: str) -> list[str]:
    cands = [symbol]
    base = symbol
    if ':USDT' in symbol:
        base = symbol.replace(':USDT', '')
        cands.append(base)
    else:
        cands.append(symbol + ':USDT')
    existing = set(ex.markets.keys())
    cands = [s for s in cands if s in existing]
    if not cands:
        prefix = base
        cands = [m for m in existing if m.startswith(prefix.split(':')[0])]
    seen, uniq = set(), []
    for s in cands:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

def _fetch_native(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        return pd.DataFrame(columns=['open','high','low','close','volume'])
    df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.set_index('ts').sort_index()

def _fetch_1m_once(ex: ccxt.Exchange, symbol: str, since_ms: int, step_limit: int = 1000):
    try:
        return ex.fetch_ohlcv(symbol, timeframe='1m', since=since_ms, limit=step_limit)
    except Exception:
        time.sleep(0.8)
        return ex.fetch_ohlcv(symbol, timeframe='1m', since=since_ms, limit=step_limit)

def _fetch_1m_resample_paged(ex: ccxt.Exchange, symbol: str, timeframe: str, limit_10m: int) -> pd.DataFrame:
    minutes_needed = int(limit_10m * 10 * 1.1)
    t_to = pd.Timestamp.utcnow().tz_convert('UTC')   # FIX
    t_from = t_to - pd.Timedelta(minutes=minutes_needed)
    since_ms = int(t_from.timestamp() * 1000)
    to_ms = int(t_to.timestamp() * 1000)

    all_rows = []
    cur = since_ms
    last_ts = None
    for _ in range(200):
        rows = _fetch_1m_once(ex, symbol, cur, step_limit=1000)
        if not rows:
            break
        if last_ts is not None and rows and rows[0][0] == last_ts:
            rows = rows[1:]
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        cur = last_ts + 60_000
        if cur >= to_ms:
            break
        time.sleep(ex.rateLimit / 1000.0)

    if not all_rows:
        return pd.DataFrame(columns=['open','high','low','close','volume'])

    df1 = pd.DataFrame(all_rows, columns=['ts','open','high','low','close','volume'])
    df1['ts'] = pd.to_datetime(df1['ts'], unit='ms', utc=True)
    df1 = df1.set_index('ts').sort_index()
    df10 = _resample_1m(df1, timeframe).tail(limit_10m)
    return df10

def fetch_ohlcv(symbol: str, timeframe: str, exchange: str = "binance", limit: int = 600) -> pd.DataFrame:
    ex = _make_exchange(exchange)
    cands = _candidate_symbols(ex, symbol)
    if not cands:
        print(f"[yellow]{exchange}: symbol '{symbol}' nie został znaleziony[/yellow]")
        return pd.DataFrame(columns=['open','high','low','close','volume'])

    native_supported = timeframe in (getattr(ex, 'timeframes', {}) or {})

    last_err = None
    for sym_try in cands:
        try:
            if native_supported:
                df = _fetch_native(ex, sym_try, timeframe, limit)
                if len(df):
                    if sym_try != symbol:
                        print(f"[yellow]Używam '{sym_try}' zamiast '{symbol}'[/yellow]")
                    return df
            df = _fetch_1m_resample_paged(ex, sym_try, timeframe, limit)
            if len(df):
                if sym_try != symbol:
                    print(f"[yellow]Używam '{sym_try}' zamiast '{symbol}' (paged 1m→{timeframe})[/yellow]")
                return df
        except Exception as e:
            last_err = e
            continue

    if last_err:
        print(f"[red]{exchange}: fetch OHLCV nieudany dla {symbol}. Ostatni błąd: {last_err}[/red]")
    else:
        print(f"[yellow]{exchange}: brak danych OHLCV dla {symbol}[/yellow]")
    return pd.DataFrame(columns=['open','high','low','close','volume'])

def fetch_ohlcv_10m(symbol: str, exchange: str = "binance", limit_10m: int = 600) -> pd.DataFrame:
    return fetch_ohlcv(symbol, '10m', exchange, limit_10m)
