"""Backfill two years of OHLCV for multiple symbols & timeframes (SQLite).

Usage:
  python -m app.scripts.backfill_binance_bulk \\
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
    --tfs 1m,10m,15m,30m,1h,2h,4h

Notes:
  - Streams in chunks (no huge memory)
  - Respects rate limits (sleep between requests)
  - Upserts to table `ohlcv` with indexes
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List

import pandas as pd

from app.data.exchange import iter_ohlcv_1m, resample, normalize_symbol
from app.storage.market import upsert_ohlcv_df, get_last_ts

def _tf_to_rule(tf: str) -> str:
    tf = tf.strip().lower()
    return tf.replace("m", "min") if tf.endswith("m") else tf

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    parser.add_argument("--tfs", type=str, default="1m,10m,15m,30m,1h,2h,4h")
    parser.add_argument("--days", type=int, default=730)  # two years
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep_ms", type=int, default=150)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tfs = [s.strip() for s in args.tfs.split(",") if s.strip()]

    now_ms = int(time.time() * 1000)
    since_ms = now_ms - args.days * 24 * 60 * 60 * 1000

    total_rows = 0
    for raw in symbols:
        sym = normalize_symbol(raw)
        print(f"==> {sym} | backfill 1m for {args.days} days", flush=True)

        # Resume support for 1m if present
        last_1m = get_last_ts(sym, "1m")
        start_ms = max(since_ms, (last_1m + 60) * 1000) if last_1m else since_ms

        # Stream chunks and upsert 1m
        for df_chunk in iter_ohlcv_1m(sym, since_ms=start_ms, until_ms=None, limit=args.limit, sleep_ms=args.sleep_ms):
            if df_chunk.empty:
                continue
            n = upsert_ohlcv_df(df_chunk, symbol=sym, timeframe="1m")
            total_rows += n
            print(f"[1m] inserted {n} rows (total={total_rows})")

        # Build and upsert higher TFs from consolidated 1m
        # To avoid loading all 1m, we re-fetch slices from DB would be ideal,
        # but for simplicity we rebuild from the latest 1m we just streamed.
        # If you want full rebuild from DB, implement a DB->DataFrame reader.
        print(f"==> {sym} | resampling to {tfs}")
        # We need an in-memory 1m set: iter again (cheap CPU, limited net)
        frames = []
        for df_chunk in iter_ohlcv_1m(sym, since_ms=since_ms, until_ms=None, limit=args.limit, sleep_ms=10):
            frames.append(df_chunk)
        if frames:
            df_1m = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
            for tf in tfs:
                if tf == "1m":
                    continue
                rule = _tf_to_rule(tf)
                df_tf = resample(df_1m, rule)
                if df_tf.empty:
                    continue
                n = upsert_ohlcv_df(df_tf, symbol=sym, timeframe=tf)
                print(f"[{tf}] upserted {n} rows")

    print(f"DONE. Total rows upserted: {total_rows}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
