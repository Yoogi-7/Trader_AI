import os
import pandas as pd
from binance.client import Client

client = Client()  # public, no key needed

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE  # pobieramy 1m i zbijemy do 10m
start = "7 days ago UTC"   # <-- tylko tydzień wstecz
end = None                 # do teraz

print(f"Pobieram dane {symbol} {interval} od {start}...")

klines = client.get_historical_klines(symbol, interval, start, end)

df = pd.DataFrame(klines, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
])

df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.astype({
    "open": "float",
    "high": "float",
    "low": "float",
    "close": "float",
    "volume": "float",
})

# agregacja do 10 minut
df = df.set_index("timestamp")
df_10m = pd.DataFrame()
df_10m["open"] = df["open"].resample("10T").first()
df_10m["high"] = df["high"].resample("10T").max()
df_10m["low"] = df["low"].resample("10T").min()
df_10m["close"] = df["close"].resample("10T").last()
df_10m["volume"] = df["volume"].resample("10T").sum()
df_10m = df_10m.dropna().reset_index()

os.makedirs("data", exist_ok=True)
out_path = "data/BTCUSDT_10m.parquet"
df_10m.to_parquet(out_path, index=False)
print(f"Zapisano {len(df_10m)} świec do {out_path}")
