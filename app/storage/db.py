"""
SQLite storage for TRADER_AI signals using SQLAlchemy 2.0.
- DB file lives under DATA_DIR / "trader_ai.sqlite"
- Provides ORM model and simple helpers to insert/query signals
All texts/comments are in English.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from app.config import DATA_DIR

DB_PATH = DATA_DIR / "trader_ai.sqlite"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)

class Base(DeclarativeBase):
    pass

class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Timestamps as ISO strings to keep it simple; could use REAL/INTEGER epoch too
    ts: Mapped[str] = mapped_column()
    symbol: Mapped[str] = mapped_column()
    tf: Mapped[str] = mapped_column()
    side: Mapped[str] = mapped_column()
    htf1: Mapped[str] = mapped_column()
    htf2: Mapped[str] = mapped_column()

    entry: Mapped[float] = mapped_column()
    sl: Mapped[float] = mapped_column()
    tp1: Mapped[float] = mapped_column()
    tp2: Mapped[float] = mapped_column()
    rr: Mapped[float] = mapped_column()
    p_hit: Mapped[float] = mapped_column()
    notional: Mapped[float] = mapped_column()
    fee: Mapped[float] = mapped_column()
    slip: Mapped[float] = mapped_column()
    net_tp: Mapped[float] = mapped_column()
    ev: Mapped[float] = mapped_column()
    ok: Mapped[int] = mapped_column()  # store as 0/1

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(ENGINE)

@dataclass
class SignalDTO:
    ts: str
    symbol: str
    tf: str
    side: str
    htf1: str
    htf2: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    p_hit: float
    notional: float
    fee: float
    slip: float
    net_tp: float
    ev: float
    ok: bool

def insert_signals(items: Iterable[SignalDTO]) -> int:
    init_db()
    count = 0
    with Session(ENGINE) as s:
        for it in items:
            row = Signal(
                ts=it.ts,
                symbol=it.symbol,
                tf=it.tf,
                side=it.side,
                htf1=it.htf1,
                htf2=it.htf2,
                entry=it.entry,
                sl=it.sl,
                tp1=it.tp1,
                tp2=it.tp2,
                rr=it.rr,
                p_hit=it.p_hit,
                notional=it.notional,
                fee=it.fee,
                slip=it.slip,
                net_tp=it.net_tp,
                ev=it.ev,
                ok=1 if it.ok else 0,
            )
            s.add(row)
            count += 1
        s.commit()
    return count

def fetch_last_signals(limit: int = 50) -> List[dict]:
    init_db()
    with Session(ENGINE) as s:
        q = s.query(Signal).order_by(Signal.id.desc()).limit(limit)
        rows = q.all()
        out = []
        for r in rows:
            out.append({
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
            })
        return out

def basic_stats() -> dict:
    """
    Compute simple descriptive stats from persisted signals.
    Note: this is not PnL backtest; just aggregates on proposed signals.
    """
    init_db()
    with Session(ENGINE) as s:
        # total count
        total = s.query(Signal).count()
        if total == 0:
            return {
                "total": 0,
                "avg_p_hit": None,
                "avg_ev": None,
                "ok_rate": None,
                "by_tf": {},
                "by_side": {},
            }
        # averages
        avg_p_hit = float(s.execute(text("SELECT AVG(p_hit) FROM signals")).scalar() or 0.0)
        avg_ev = float(s.execute(text("SELECT AVG(ev) FROM signals")).scalar() or 0.0)
        ok_rate = float(s.execute(text("SELECT AVG(ok) FROM signals")).scalar() or 0.0)

        # group by tf
        by_tf_rows = s.execute(text("SELECT tf, COUNT(*) c FROM signals GROUP BY tf")).all()
        by_tf = {row[0]: int(row[1]) for row in by_tf_rows}

        # group by side
        by_side_rows = s.execute(text("SELECT side, COUNT(*) c FROM signals GROUP BY side")).all()
        by_side = {row[0]: int(row[1]) for row in by_side_rows}

        return {
            "total": int(total),
            "avg_p_hit": avg_p_hit,
            "avg_ev": avg_ev,
            "ok_rate": ok_rate,
            "by_tf": by_tf,
            "by_side": by_side,
        }
