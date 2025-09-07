# TRADER_AI — MVP (BTCUSDT @ Bitget)

Local MVP for scanning BTCUSDT perpetual (Bitget) on TF ≥ 10m:
- 1m ingest via CCXT → Parquet
- Resample to 10m/15m/30m (and higher for HTF)
- HTF trend filter (EMA-200 + slope)
- Simple detectors (breakout + fallback momentum)
- EV/risk sizing placeholders (to be refined with ML later)

## Quickstart (WSL Ubuntu 24 + Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m app.data.ingest_bitget
python -m app.scripts.bootstrap_btcusdt
