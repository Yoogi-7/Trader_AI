# core/cache.py
from __future__ import annotations
import os
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime, timezone

CACHE_DIR = "cache"  # dodaj do .gitignore

def _safe_fname(symbol: str, interval: str) -> str:
    sym = "".join(c for c in symbol.upper() if c.isalnum())
    itv = interval.lower().replace(" ", "").replace("/", "_")
    return f"{sym}__{itv}.parquet"

def cache_path(symbol: str, interval: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, _safe_fname(symbol, interval))

def load_cached(symbol: str, interval: str) -> pd.DataFrame:
    path = cache_path(symbol, interval)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.read_parquet(path)
    # standaryzacja
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

def save_cached(symbol: str, interval: str, df: pd.DataFrame) -> None:
    path = cache_path(symbol, interval)
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp","open","high","low","close","volume"]]
    df.to_parquet(path, index=False)

def _fetch_binance(symbol: str, interval_code: str, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
    """Pobiera świeczki z Binance w [start_ms, end_ms)."""
    from binance.client import Client
    client = Client()  # publiczne end-pointy
    start_str = start_ms
    end_str = end_ms if end_ms is not None else None
    klines = client.get_historical_klines(symbol, interval_code, start_str, end_str)
    if not klines:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["timestamp","open","high","low","close","volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.astype({"open":"float","high":"float","low":"float","close":"float","volume":"float"})
    return df.sort_values("timestamp").reset_index(drop=True)

def ensure_range_cached(
    symbol: str,
    interval_code: str,
    need_start: datetime,
    need_end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Upewnia się, że cache pokrywa wymagany zakres [need_start, need_end].
    Braki dociąga z Binance i zapisuje do pliku cache. Zwraca CAŁY cache (po aktualizacji).
    """
    need_start = need_start.astimezone(timezone.utc)
    if need_end is None:
        need_end = datetime.now(timezone.utc)
    else:
        need_end = need_end.astimezone(timezone.utc)

    cached = load_cached(symbol, interval_code)
    if cached.empty:
        # brak cache – pobierz cały potrzebny zakres
        df_new = _fetch_binance(symbol, interval_code, int(need_start.timestamp()*1000), int(need_end.timestamp()*1000))
        # deduplikacja na wszelki wypadek
        if not df_new.empty:
            df_new = df_new.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        save_cached(symbol, interval_code, df_new)
        return df_new

    # mamy jakiś cache – sprawdź luki
    have_start = cached["timestamp"].iloc[0]
    have_end = cached["timestamp"].iloc[-1]

    pieces = [cached]

    # jeśli potrzebujemy wcześniejszych danych niż najstarsze w cache – dolej z lewej
    if need_start < have_start:
        df_left = _fetch_binance(symbol, interval_code, int(need_start.timestamp()*1000), int(have_start.timestamp()*1000))
        if not df_left.empty:
            pieces.insert(0, df_left)

    # jeśli potrzebujemy nowszych niż to, co mamy – dolej z prawej
    if need_end > have_end:
        # start od (have_end + 1 ms), żeby nie dublować ostatniej świecy
        df_right = _fetch_binance(symbol, interval_code, int(have_end.timestamp()*1000)+1, int(need_end.timestamp()*1000))
        if not df_right.empty:
            pieces.append(df_right)

    merged = pd.concat(pieces, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    save_cached(symbol, interval_code, merged)
    return merged

def slice_range(df: pd.DataFrame, start_dt: datetime, end_dt: Optional[datetime]) -> pd.DataFrame:
    if df.empty:
        return df
    start_dt = start_dt.astimezone(timezone.utc)
    if end_dt is not None:
        end_dt = end_dt.astimezone(timezone.utc)
        m = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
    else:
        m = (df["timestamp"] >= start_dt)
    return df.loc[m].copy().reset_index(drop=True)

def clear_cache(symbol: str, interval_code: str) -> Tuple[bool, str]:
    """Usuwa plik cache dla (symbol, interval)."""
    path = cache_path(symbol, interval_code)
    if os.path.exists(path):
        os.remove(path)
        return True, path
    return False, path
