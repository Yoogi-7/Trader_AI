# Trader\_AI

An automated, AI-assisted crypto trading signal engine with a simple user-facing workflow. It continuously ingests market data, detects opportunities using technical logic (Fibonacci/AVWAP/ATR), filters them with a meta-model to target higher win-rate, and can notify users on Telegram. Everything runs on a local SQLite database and a scheduler.

> **Goal:** deliver high-quality, actionable signals (entry, TP, SL, leverage, position size) while keeping the UI simple. The AI tunes thresholds, calibrates probabilities, and adapts per pair, timeframe, and market regime.

---

## Key Features

* **Data Pipeline**

  * Historical and incremental OHLCV download via `ccxt` (default: Bitget).
  * Resampling from 1m to higher timeframes in-database.
  * SQLite with WAL mode for robust concurrent access.
* **Signal Engine**

  * Technical setup: **Fibonacci** retracements/extensions, **AVWAP** (anchored VWAP), **ATR** for SL/volatility.
  * AI-generated trade plan: **entry, SL, TP1/TP2, leverage, position size (risk %)**.
  * **Minimum net TP** constraint (fees + slippage aware).
* **Meta-Model (Precision Filter)**

  * XGBoost classifier → `p(win)` for each candidate signal.
  * Optional **probability calibration** (Isotonic regression) for reliable probabilities.
  * **EV-gating**: only pass signals with positive expected value (or use `p(win)` threshold).
  * Per **pair/TF** overrides and per **regime** overrides (trend/range, vol low/normal/high).
* **Automation**

  * Weekly **re-train**, **calibrate**, and **tune thresholds** automatically.
  * Retro-scanner to backfill historical signals and simulate executions.
* **Dashboard**

  * Streamlit UI: recent candles, latest signals, equity curve, metrics, and coverage vs win-rate.
* **Notifications**

  * Telegram push on **new signals** and on **closed trades** (TP/SL/trailing exits).

---

## Project Structure (high level)

```
Trader_AI/
├─ config.yaml                # Main configuration
├─ requirements.txt           # Python dependencies
├─ scheduler_run.py           # Orchestrates periodic jobs
├─ download_data.py           # Backfill + incremental OHLCV (via ccxt)
├─ retro_scan.py              # Historical signal seeding and simulation
├─ scan_signals.py            # Real-time signal scanner + gating
├─ core/
│  ├─ schema.py               # SQLite DDL & migrations
│  ├─ resample.py             # 1m → higher TF resampling
│  ├─ signals.py              # Signal generation & DB insert
│  ├─ execution.py            # Trade simulation & status updates
│  ├─ indicators.py, fibo.py, avwap.py, risk.py
│  ├─ features.py             # Feature engineering for ML
│  ├─ ml.py                   # Model/calibrator loading, inference
│  ├─ performance.py          # Metrics & equity
│  ├─ regime.py               # Trend/range + volatility detection
│  └─ notify.py               # Telegram integration
├─ train_meta.py              # Train the meta-model
├─ calibrate_meta.py          # Calibrate probabilities
├─ tune_thresholds.py         # Auto-tune per-pair/TF thresholds
├─ maintenance.py             # Weekly automation wrappers
├─ dashboard.py               # Streamlit app (UI)
├─ analyze_results.py         # Offline metrics & equity CSV
└─ db_inspect.py              # Quick DB counters
```

---

## Requirements

* **Python**: 3.10–3.11 recommended
* `pip install -r requirements.txt`
* Exchange connectivity via **ccxt** (default market: Bitget; configure in `config.yaml`).

---

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
2. **Configure** your settings in `config.yaml` (see [Configuration](#configuration)).
3. **Initialize DB schema**

   ```bash
   python init_db.py
   ```
4. **Backfill/Incremental data** (optional backfill; scheduler will keep it fresh):

   ```bash
   python download_data.py   # one-off backfill (respects config)
   ```
5. **Run the scheduler** (continuous fetching, resampling, scanning, executing):

   ```bash
   python scheduler_run.py
   ```
6. **Open the dashboard** in another terminal:

   ```bash
   streamlit run dashboard.py
   ```

> The DB is SQLite in WAL mode. File size grows mostly in `ohlcv.db-wal` until periodic checkpoints merge it into `ohlcv.db`.

---

## Configuration

Everything is controlled via `config.yaml`. Key sections:

### `app`

* `data_dir`, `db_path`, `timezone` (used for scheduling and display).

### `exchange`

* `id`: e.g. `bitget` (must be supported by ccxt).
* `rate_limit_ms`: throttle for API calls.

### `symbols`, `timeframes`

* Trading universe and timeframes to scan.

### `data`

* `base_timeframe`: usually `1m`.
* `derived_timeframes`: resampled targets (e.g. `5m`, `15m`, `1h`).
* `resample.lookback_minutes`: 1m window used when resampling.

### `backfill` & `incremental`

* `start_date`: historical start (e.g. `2020-01-01`).
* `batch_limit`, `overlap_candles`, `run_seconds`: batching and cadence.

### `signals` (strategy-level parameters)

* `lookback_candles`, `atr_period`, `atr_mult_sl`.
* `fibo.*` (pivot window, number of anchors considered), `avwap.use`.
* **Constraints**: `min_tp_net_pct` (ensure TP1 covers costs), `fees`, `slippage_pct`, `validity_candles`.
* `evaluation.usd_per_trade`: nominal stake for PnL accounting.

### `risk`

* Mode-based risk per trade, max leverage, liquidation buffer.

### `execution`

* TP1 fraction, BE-on-TP1, and ATR-based trailing settings.

### `models.meta`

* `enabled`: enable the precision filter.
* `model_path`: where the trained classifier is saved.
* `calibration.enabled` + `path`: turn on Isotonic calibration.
* **Gating** (`models.meta.gating`):

  * `use_ev`: if true, pass only signals with non-negative **expected value** (EV) instead of pure `p(win)` threshold.
  * `min_ev_pct`: global EV minimum; can be overridden per pair/TF and by regime.
  * `threshold`: global `p(win)` fallback when `use_ev=false`.
  * `costs_mode`: `auto`/`maker`/`taker` (cost assumption in EV calc).
  * `target_winrate_pct`, `window_days`, `min_closed_samples`: auto-tuning controls.
  * `per_pair_tf`: overrides per symbol & timeframe (filled by tuner).
  * `per_regime`: overrides per **trend/range** and **vol low/normal/high**.

### `auto`

* Weekly jobs (UTC) for **training**, **calibration**, and **threshold tuning**. The scheduler picks them up automatically.

### `notify.telegram`

* Set `enabled: true`, provide `bot_token` and `chat_id` to receive push messages for new signals and closed trades.

---

## Running Components

### Scheduler (`scheduler_run.py`)

Jobs (all UTC):

* `incremental_sync`: periodic OHLCV updates via ccxt.
* `resample`: build derived timeframes from 1m.
* `scan_signals`: detect opportunities; apply meta-model gating (EV or `p(win)`); insert signals into DB.
* `execute_signals`: simulate TP/SL/TP1→BE/trailing; finalize trades.
* `db_checkpoint`: WAL checkpoint every 30 minutes; daily VACUUM.
* `notify_closed`: send Telegram messages for trades closed in the last minute.
* Weekly: `auto_train`, `auto_calibrate`, `auto_tune` if enabled.

### Dashboard (`dashboard.py`)

* **Signals & Candles**: latest OHLCV and most recent signals.
* **Results**: equity curve, core metrics, coverage vs win-rate table, histogram of `ml_p`.

---

## Machine Learning

### Training (`train_meta.py`)

* Builds a dataset from historical **closed** signals (`TP`, `SL`, `TP1_TRAIL`).
* Time-aware cross-validation (purged walk-forward) and final fit.
* Saves bundle `{model, features}` to `models/meta_xgb.pkl`.

### Calibration (`calibrate_meta.py`)

* Uses a holdout split to fit Isotonic regression.
* Saves `models/meta_cal.pkl`. When enabled, inferences use calibrated probabilities.

### Auto-Tuning (`tune_thresholds.py`)

* Searches thresholds per **symbol/TF** to meet `target_winrate_pct` with maximum coverage within the recent window (`window_days`).
* Works in two modes:

  * **EV mode**: find `min_ev_pct` per pair/TF.
  * **p(win) mode**: find `threshold` per pair/TF.
* Writes overrides to `models.meta.gating.per_pair_tf` in `config.yaml`.

> The scanner can also combine **per-regime** overrides (trend/range, vol low/high), allowing adaptive gating.

---

## Historical Seeding (Retro-Scan)

Use `retro_scan.py` to apply the current strategy logic over historical candles and simulate outcomes:

```bash
python retro_scan.py --start 2021-01-01 --end 2021-12-31 --stride 3 --limit-per-pair 300
```

* `--stride`: test every N-th bar to speed up.
* Respects the same gating (EV or `p(win)`) as live.
* Immediately runs simulation to produce closed trades.

---

## Notifications (Telegram)

1. Create a bot with @BotFather and get `BOT_TOKEN`.
2. Get your `CHAT_ID` (e.g., by messaging the bot and using a helper or a simple script).
3. In `config.yaml`:

   ```yaml
   notify:
     telegram:
       enabled: true
       bot_token: "YOUR_BOT_TOKEN"
       chat_id: "YOUR_CHAT_ID"
   ```
4. Start the scheduler; you’ll receive new-signal and trade-closure alerts.

---

## Data Storage

* **SQLite** at `data/ohlcv.db` with WAL journaling:

  * tables: `ohlcv`, `sync_checkpoint`, `signals` (+ indices and evolving columns).
  * WAL file `ohlcv.db-wal` grows between checkpoints.
* **Checkpoint/Vacuum** are handled automatically by the scheduler.

Quick inspection:

```bash
python db_inspect.py
```

---

## Troubleshooting

* **DB size not increasing**: WAL mode stores recent writes in `ohlcv.db-wal`. Scheduler periodically checkpoints and vacuums.
* **Duplicate column errors**: schema migrations are idempotent; ensure you’re on latest `core/schema.py` and restart.
* **No signals**: ensure `download_data.py` or the scheduler is fetching 1m data and that resampling is running; check `signals.min_tp_net_pct` is not too strict.
* **Telegram not sending**: verify `enabled`, `bot_token`, `chat_id`; check internet access and bot permissions.

---

## Roadmap

* Funding/OI/Basis as ML features (if data feed available).
* Per-symbol/TF small models (or thresholds) with automatic drift detection.
* Cooldown & max concurrent position guardrails.
* Weekly HTML/Telegram performance summary.

---

## Disclaimer

This repository is for research and educational purposes only. It is **not** financial advice. Crypto derivatives and leverage trading carry significant risk. Use at your own responsibility.
