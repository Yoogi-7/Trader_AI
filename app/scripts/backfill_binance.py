"""One-off backfill script using Binance and saving to DB."""
from app.api.routes_ingest import ingest_binance

if __name__ == "__main__":
    print(ingest_binance(symbol="BTCUSDT", days=365, timeframe="1h", persist=True))
