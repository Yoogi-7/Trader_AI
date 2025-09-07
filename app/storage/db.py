"""Tiny SQLAlchemy helper for a single-file SQLite setup.

The goal is to keep DB wiring minimal and readable. ORM models live elsewhere;
this module only exposes:
  - ``get_engine``: returns a process-wide Engine (lazy singleton)
  - ``execute``: run a write statement inside a transaction
  - ``fetch_all``: run a read statement and return list of dicts

Default DB location: ``sqlite:///./data/signals.db`` (relative to CWD).
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DB_URL = "sqlite:///./data/signals.db"

_engine: Optional[Engine] = None  # lazy singleton


def get_engine(db_url: str = DEFAULT_DB_URL) -> Engine:
    """Return a cached SQLAlchemy Engine for the given URL.

    The first call creates the engine; subsequent calls return the same object
    as long as the ``db_url`` does not change.
    """
    global _engine
    if _engine is None or str(_engine.url) != db_url:
        _engine = create_engine(db_url, future=True)
    return _engine


def execute(sql: str, params: Dict[str, Any] | None = None) -> None:
    """Execute a write statement inside a transaction."""
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), params or {})


def fetch_all(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Execute a read statement and return rows as a list of dicts."""
    eng = get_engine()
    with eng.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(row._mapping) for row in result]
