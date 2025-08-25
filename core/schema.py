# core/schema.py

DDL_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    exchange   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    ts_ms      INTEGER NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe, ts_ms)
);
"""

DDL_CHECKPOINT = """
CREATE TABLE IF NOT EXISTS sync_checkpoint (
    exchange   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    last_ts_ms INTEGER NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe)
);
"""

# Podstawowa definicja tabeli sygnałów (minimalny szkielet)
DDL_SIGNALS_BASE = """
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,
  exchange TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  direction TEXT NOT NULL,
  entry REAL NOT NULL,
  sl REAL NOT NULL,
  tp1 REAL NOT NULL,
  tp2 REAL NOT NULL,
  leverage REAL NOT NULL,
  risk_pct REAL NOT NULL,
  position_notional REAL NOT NULL,
  confidence REAL NOT NULL,
  rationale TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'PENDING'
);
CREATE INDEX IF NOT EXISTS idx_signals_recent ON signals(exchange, symbol, timeframe, ts_ms DESC);
"""

# Kolumny dodawane ewolucyjnie (ALTER TABLE) – zgodne z tym, co używa dashboard/egzekucja
SIGNALS_MISSING_COLS = [
    ("opened_ts_ms", "INTEGER"),
    ("closed_ts_ms", "INTEGER"),
    ("exit_price",   "REAL"),
    ("pnl_usd",      "REAL"),
    ("pnl_pct",      "REAL"),
    ("tp1_hit",      "INTEGER"),
    ("exit_reason",  "TEXT"),
]

def ensure_base_schema(conn):
    """Utwórz podstawowe tabele, jeśli brak."""
    conn.execute(DDL_OHLCV)
    conn.execute(DDL_CHECKPOINT)
    for stmt in DDL_SIGNALS_BASE.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")

def migrate_signals_schema(conn):
    """Dodaj brakujące kolumny do signals (idempotentnie)."""
    cur = conn.execute("PRAGMA table_info(signals);")
    cols = {row[1] for row in cur.fetchall()}
    for col, coltype in SIGNALS_MISSING_COLS:
        if col not in cols:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {coltype};")
