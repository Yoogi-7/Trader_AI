"""
Export helpers for persisted signals.
- Export to CSV or Parquet from SQLite using SQLAlchemy session.
- Returns the absolute path of the created file.
All texts/comments are in English.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.storage.db import ENGINE, Signal, init_db
from app.config import DATA_DIR

EXPORT_DIR = DATA_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_signals(format: str = "csv", limit: Optional[int] = None) -> Path:
    """Export signals to CSV or Parquet. Returns created file path."""
    fmt = (format or "csv").lower()
    if fmt not in {"csv", "parquet"}:
        raise ValueError("format must be 'csv' or 'parquet'")

    init_db()
    with Session(ENGINE) as s:
        q = s.query(Signal).order_by(Signal.id.desc())
        if isinstance(limit, int) and limit > 0:
            q = q.limit(limit)
        rows = q.all()

    if not rows:
        # create an empty file with headers
        df = pd.DataFrame(columns=[
            "id","ts","symbol","tf","side","htf1","htf2",
            "entry","sl","tp1","tp2","rr","p_hit","notional",
            "fee","slip","net_tp","ev","ok"
        ])
    else:
        df = pd.DataFrame([{
            "id": r.id,
            "ts": r.ts,
            "symbol": r.symbol,
            "tf": r.tf,
            "side": r.side,
            "htf1": r.htf1,
            "htf2": r.htf2,
            "entry": r.entry,
            "sl": r.sl,
            "tp1": r.tp1,
            "tp2": r.tp2,
            "rr": r.rr,
            "p_hit": r.p_hit,
            "notional": r.notional,
            "fee": r.fee,
            "slip": r.slip,
            "net_tp": r.net_tp,
            "ev": r.ev,
            "ok": bool(r.ok),
        } for r in rows])

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"signals_{ts}.{fmt}"

    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path
