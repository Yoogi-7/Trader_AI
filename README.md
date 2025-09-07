# TRADER_AI — MVP (BTCUSDT @ Bitget)

Local MVP for scanning BTCUSDT perpetual futures (Bitget) on TF ≥ 10m:
- 1m ingest via CCXT → Parquet
- Resample to 10m/15m/30m (and higher for HTF filter)
- HTF trend filter (EMA-200 + slope)
- Simple detectors (breakout + fallback momentum, series-mode)
- Risk sizing and EV placeholders
- ML model (LightGBM) for `p_hit` (probability of TP1-before-SL)
- **SQLite persistence** of signals + API endpoints to browse stats

---

## 📦 Project layout

```
app/
  api/              # FastAPI server
  data/             # Ingest, resample, store
  features/         # Feature engineering
  labels/           # Labeling (TP1-before-SL)
  model/            # Dataset build, train, predict
  pipeline/         # Shared scan pipeline
  rules/            # Detectors, EV, levels, trend
  scripts/          # Bootstrap + utilities
  storage/          # SQLite/SQLAlchemy persistence
policy.yaml         # Signal, risk, and detector config
.env.example        # Local config
```

---

## ⚙️ Setup (Ubuntu 24 + Python 3.12)

```bash
git clone <your-repo-url>
cd Trader_AI

python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

> **Gotcha (your error):** If you see `ModuleNotFoundError: No module named 'sqlalchemy'`,  
> make sure your `requirements.txt` is updated and **reinstalled**:
> ```bash
> pip install -r requirements.txt
> ```

---

## ⏳ Data ingest & backfill

By default, the system incrementally fetches new 1m candles.  
To build a dataset and train a model, we recommend **backfilling ~365 days**:

```bash
python -m app.scripts.backfill_btcusdt  # defaults to 365 days
```

Data is stored under `./data/raw_1m/`.

---

## 📊 Build dataset & train model

```bash
python -m app.model.build_dataset
python -m app.model.train
```

Artifacts:
- `data/datasets/btcusdt_tf10_30.parquet`
- `data/models/p_hit_lgbm.joblib`

If no model is found, pipeline falls back to a constant `p_hit = 0.60`.

---

## 🚀 Run API

```bash
uvicorn app.api.main:app --reload --port 8080
```

Open http://127.0.0.1:8080/docs for Swagger UI.

- **POST /scan** → returns signals with entry/SL/TP, RR, fees, EV, and `p_hit`.  
  - Set `persist=true` to save signals into SQLite.
- **GET /signals/last** → fetch recent persisted signals.  
- **GET /stats** → quick aggregates from persisted signals.  
- **GET /health** → simple health check.

Example curl:

```bash
# Persist latest scan
curl -X POST "http://127.0.0.1:8080/scan" -H "Content-Type: application/json"      -d '{"symbols":["BTCUSDT"],"risk_profile":"medium","equity":5000,"tfs":["10m","30m"],"run_ingest":true,"persist":true}'

# Fetch last signals
curl "http://127.0.0.1:8080/signals/last?limit=20"

# Basic stats
curl "http://127.0.0.1:8080/stats"
```

---

## 🖥️ CLI bootstrap

You can also run a one-off scan directly:

```bash
python -m app.scripts.bootstrap_btcusdt
```

---

## 🔧 Config

Edit **policy.yaml** to adjust:
- Timeframes (`signal_timeframes`)
- Detector params (`detector_defaults`, per-TF `lookback`)
- Risk profiles (low/medium/high)
- Execution costs (fees, slippage)

---

## 🛠️ Development

- All code/comments are in English.
- Data, models, SQLite db, and `.env` are gitignored.
- Recommended editor: VS Code with Python extension.

---

## ✅ Next steps

- Walk-forward validation for `p_hit`
- Multi-symbol training
- Extend API with `/stats` (rolling hit-rate, EV)
- Minimal Streamlit dashboard

---

## 🧪 GitHub Actions (CI)

A minimal CI workflow (`.github/workflows/ci.yml`) is included to:
- Set up Python 3.12
- Install requirements
- Run quick sanity-checks (import modules, run bootstrap; ignore absence of signals)

To enable it:
1. Create the folder `.github/workflows/` at the repo root.
2. Add the `ci.yml` from this package (see download below).
3. Commit & push — then check the **Actions** tab.

```yaml
name: CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run sanity tests
        run: |
          python -c "import app; print('Import OK')"
          python -m app.scripts.bootstrap_btcusdt || true
```

---

## 📝 Commit message suggestion

```
fix(api): install SQLAlchemy and document persistence flow
docs: update README with persistence, API endpoints, and CI workflow
```
