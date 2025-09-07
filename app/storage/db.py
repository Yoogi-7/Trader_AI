"""SQLite persistence layer for signals (backward-compatible + minimal).

Provides:
  • SignalDTO (dataclass) – app-level data transfer object
  • SQLAlchemy ORM model `Signal`
  • ENGINE (global Engine) + init_db()/ensure_db()
  • insert_signals()/fetch_last_signals()/basic_stats()/rolling_stats()
  • execute()/fetch_all() helpers

Default DB URL: sqlite:///./data/signals.db
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import json
import os
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_DB_URL = "sqlite:///./data/signals.db"

# -------------------------------------------------------------------------
# Dataclass DTO
# -------------------------------------------------------------------------

@dataclass
class SignalDTO:
    ts: int
    symbol: str
    timeframe: str
    direction: str
    entry: float
    sl: float
    tp: float
    rr: Optional[float] = None
    ev: Optional[float] = None
    fee: Optional[float] = None
    slippage: Optional[float] = None
    p_hit: Optional[float] = None
    detector: Optional[str] = None
    trend: Optional[str] = None
    model: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# -------------------------------------------------------------------------
# ORM model
# -------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass

class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[int]
    symbol: Mapped[str]
    timeframe: Mapped[str]
    direction: Mapped[str]
    entry: Mapped[float]
    sl: Mapped[float]
    tp: Mapped[float]
    rr: Mapped[Optional[float]]
    ev: Mapped[Optional[float]]
    fee: Mapped[Optional[float]]
    slippage: Mapped[Optional[float]]
    p_hit: Mapped[Optional[float]]
    detector: Mapped[Optional[str]]
    trend: Mapped[Optional[str]]
    model: Mapped[Optional[str]]
    meta: Mapped[Optional[str]]
    created_at: Mapped[Optional[int]]

# -------------------------------------------------------------------------
# Engine + bootstrap
# -------------------------------------------------------------------------

_ENGINE: Optional[Engine] = None

def get_engine(db_url: str = DEFAULT_DB_URL) -> Engine:
    global _ENGINE
    if _ENGINE is None or str(_ENGINE.url) != db_url:
        _ENGINE = create_engine(db_url, future=True)
    return _ENGINE

ENGINE: Engine = get_engine()

def init_db() -> None:
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(ENGINE)

def ensure_db() -> None:
    init_db()

# -------------------------------------------------------------------------
# Low-level helpers
# -------------------------------------------------------------------------

def execute(sql: str, params: Dict[str, Any] | None = None) -> None:
    init_db()
    with ENGINE.begin() as conn:
        conn.execute(text(sql), params or {})

def fetch_all(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    init_db()
    with ENGINE.connect() as conn:
        res = conn.execute(text(sql), params or {})
        return [dict(r._mapping) for r in res]

# -------------------------------------------------------------------------
# High-level API
# -------------------------------------------------------------------------

def insert_signals(signals: Iterable[SignalDTO]) -> int:
    init_db()
    rows = []
    for s in signals:
        d = asdict(s)
        d["meta"] = json.dumps(d.get("meta") or {})
        rows.append(d)
    if not rows:
        return 0
    sql = text(
        """
        INSERT INTO signals
        (ts, symbol, timeframe, direction, entry, sl, tp, rr, ev, fee, slippage, p_hit,
         detector, trend, model, meta, created_at)
        VALUES
        (:ts, :symbol, :timeframe, :direction, :entry, :sl, :tp, :rr, :ev, :fee, :slippage, :p_hit,
         :detector, :trend, :model, :meta, strftime('%s','now'))
        """
    )
    with ENGINE.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def fetch_last_signals(
    limit: int = 50,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> List[Dict[str, Any]]:
    init_db()
    clauses = ["1=1"]
    params: Dict[str, Any] = {"limit": int(limit)}
    if symbol:
        clauses.append("symbol = :symbol")
        params["symbol"] = symbol
    if timeframe:
        clauses.append("timeframe = :timeframe")
        params["timeframe"] = timeframe

    sql = f"""
    SELECT id, ts, symbol, timeframe, direction, entry, sl, tp, rr, ev, fee, slippage, p_hit,
           detector, trend, model, meta, created_at
    FROM signals
    WHERE {' AND '.join(clauses)}
    ORDER BY ts DESC, id DESC
    LIMIT :limit
    """
    rows = fetch_all(sql, params)
    for r in rows:
        if isinstance(r.get("meta"), str):
            try:
                r["meta"] = json.loads(r["meta"])
            except Exception:
                r["meta"] = {}
    return rows

def basic_stats(days: int = 30) -> Dict[str, Any]:
    init_db()
    now = int(time.time())
    since = now - days * 86400

    overall_sql = text(
        """
        SELECT
            COUNT(*) AS cnt,
            AVG(COALESCE(p_hit, 0)) AS avg_p_hit,
            AVG(COALESCE(rr, 0)) AS avg_rr,
            AVG(COALESCE(ev, 0)) AS avg_ev
        FROM signals
        WHERE ts >= :since
        """
    )
    by_dir_sql = text(
        """
        SELECT direction, COUNT(*) AS cnt
        FROM signals
        WHERE ts >= :since
        GROUP BY direction
        ORDER BY cnt DESC
        """
    )
    by_symbol_sql = text(
        """
        SELECT symbol, COUNT(*) AS cnt
        FROM signals
        WHERE ts >= :since
        GROUP BY symbol
        ORDER BY cnt DESC
        LIMIT 10
        """
    )
    with ENGINE.connect() as conn:
        overall = dict(conn.execute(overall_sql, {"since": since}).mappings().first() or {})
        by_dir = [dict(r) for r in conn.execute(by_dir_sql, {"since": since})]
        by_symbol = [dict(r) for r in conn.execute(by_symbol_sql, {"since": since})]
    return {
        "window_days": days,
        "since_ts": since,
        "now_ts": now,
        "overall": overall,
        "by_direction": by_dir,
        "top_symbols": by_symbol,
    }

def rolling_stats(
    days: int = 30,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return per-day aggregates over the last N days, optionally filtered."""
    init_db()
    now = int(time.time())
    since = now - days * 86400

    clauses = ["ts >= :since"]
    params: Dict[str, Any] = {"since": since}
    if symbol:
        clauses.append("symbol = :symbol")
        params["symbol"] = symbol
    if timeframe:
        clauses.append("timeframe = :timeframe")
        params["timeframe"] = timeframe

    sql = text(
        f"""
        SELECT
            date(ts, 'unixepoch') AS day,
            COUNT(*) AS cnt,
            AVG(COALESCE(p_hit, 0)) AS avg_p_hit,
            AVG(COALESCE(rr, 0)) AS avg_rr,
            AVG(COALESCE(ev, 0)) AS avg_ev
        FROM signals
        WHERE {' AND '.join(clauses)}
        GROUP BY day
        ORDER BY day ASC
        """
    )
    with ENGINE.connect() as conn:
        res = conn.execute(sql, params)
        return [dict(r) for r in res]
