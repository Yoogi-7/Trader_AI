"""Health endpoints with DB details."""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter
from sqlalchemy import text

from app.storage.db import ENGINE, init_db

router = APIRouter(tags=["health"])


@router.get("/health/details")
def health_details() -> Dict[str, Any]:
    """Return extended health info: DB connectivity and basic counts."""
    init_db()
    payload: Dict[str, Any] = {"status": "ok", "ts": int(time.time())}
    try:
        with ENGINE.connect() as conn:
            # signals table exists?
            exists = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
            ).fetchone()
            payload["db"] = {"connected": True, "signals_table": bool(exists)}

            # counts (safe even if empty table)
            if exists:
                counts = conn.execute(
                    text("SELECT COUNT(*) AS cnt FROM signals")
                ).mappings().first()
                payload["signals"] = {"total": int(counts["cnt"]) if counts else 0}
            else:
                payload["signals"] = {"total": 0}
    except Exception as e:
        payload["db"] = {"connected": False, "error": str(e)}
    return payload
