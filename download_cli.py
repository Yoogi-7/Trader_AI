# download_cli.py
# Uniwersalny downloader danych z Binance z parametrami CLI.
# Przykłady:
#   python download_cli.py --symbol BTCUSDT --interval 1m --days 7
#   python download_cli.py --symbol ETHUSDT --interval 10m --days 30
#   python download_cli.py --symbol BTCUSDT --interval 1h --start "2024-01-01" --end "2024-02-01"

from __future__ import annotations

import os
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd

SUPPORTED_NATIVE = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"}

def _fetch_binance(symbol: str, interval_code: str, start_str: str, end_str: str | None) -> pd.DataFrame:
    try:
        from binance.client import Client
    except Exception as e:
        raise SystemExit("Brak pakietu 'python-binance'. Zainstaluj: python -m pip install python-binance") from e

    client = Client()  # publiczne
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

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    g = df.set_index("timestamp")
    out = pd.DataFrame()
    out["open"] = g["open"].resample(rule).first()
    out["high"] = g["high"].resample(rule).max()
    out["low"] = g["low"].resample(rule).min()
    out["close"] = g["close"].resample(rule).last()
    out["volume"] = g["volume"].resample(rule).sum()
    return out.dropna().reset_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="np. BTCUSDT")
    ap.add_argument("--interval", required=True, help="np. 1m,5m,10m,15m,30m,1h,...")
    ap.add_argument("--days", type=int, default=None, help="Ile dni wstecz (alternatywa dla --start/--end)")
    ap.add_argument("--start", type=str, default=None, help='YYYY-MM-DD lub fraza Binance, np. "1 Jan, 2024"')
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (opcjonalnie)")
    ap.add_argument("--outdir", type=str, default="data", help="Katalog wyjściowy")
    args = ap.parse_args()

    # Ustal zakres czasu
    if args.days is not None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=args.days)
        start_str = start_dt.strftime("%d %b, %Y %H:%M:%S")
        end_str = None
    else:
        start_str = args.start or "30 days ago UTC"
        end_str = args.end

    interval = args.interval.lower()
    needs_resample = False
    fetch_interval = interval
    rule = None

    # obsługa niestandardowych interwałów jak 10m
    if interval not in SUPPORTED_NATIVE:
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            if minutes % 1 != 0 or minutes <= 0:
                raise SystemExit(f"Nieobsługiwany interwał: {interval}")
            # pobieramy 1m i składamy
            fetch_interval = "1m"
            rule = f"{minutes}T"
            needs_resample = True
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            fetch_interval = "1m"
            rule = f"{hours}H"
            needs_resample = True
        else:
            raise SystemExit(f"Nieobsługiwany interwał: {interval}")
    else:
        # natywny — bez resamplingu
        rule = None

    print(f"Pobieram {args.symbol} {fetch_interval} od {start_str}…")
    df = _fetch_binance(args.symbol, fetch_interval, start_str, end_str)
    if df.empty:
        raise SystemExit("Brak danych z API dla podanych parametrów.")

    if needs_resample and rule:
        print(f"Agreguję do {interval} (pandas rule={rule})…")
        df = _resample(df, rule)

    os.makedirs(args.outdir, exist_ok=True)
    fname = f"{args.symbol}_{interval}.parquet"
    out_path = os.path.join(args.outdir, fname)
    df.to_parquet(out_path, index=False)
    print(f"Zapisano {len(df)} świec do {out_path}")

if __name__ == "__main__":
    main()
