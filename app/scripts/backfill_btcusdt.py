"""
One-off backfill for BTCUSDT to build enough history for dataset/training.
Adjust DAYS if you want more/less.
"""
from app.data.ingest_bitget import backfill_days
from app.data.store import path_raw

DAYS = 365  # try 365 first; you can increase if needed

if __name__ == "__main__":
    n = backfill_days("BTCUSDT", days=DAYS)
    print(f"Backfilled {n} rows into {path_raw('BTCUSDT')}")
