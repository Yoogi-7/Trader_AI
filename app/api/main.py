"""FastAPI application entrypoint.

- Defines the FastAPI `app` first.
- Adds CORS (optional).
- Exposes `/` and `/health`.
- Includes project routers (stats, health, alerts, ingest, market, scan, signals, export).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="TRADER_AI API", version="0.2.3")

# -----------------------------------------------------------------------------
# CORS (optional)
# -----------------------------------------------------------------------------
try:
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore

    allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    # CORS is optional; ignore if middleware not available
    pass


# -----------------------------------------------------------------------------
# Root & Health
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "TRADER_AI",
        "version": "0.2.3",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "ts": int(time.time()),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Lightweight health with DB ping and `signals` table check."""
    payload: Dict[str, Any] = {"status": "ok", "ts": int(time.time())}
    try:
        from app.storage.db import ENGINE, init_db  # lazy import to avoid heavy startup

        init_db()
        with ENGINE.connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
            ).fetchone()
        payload["db"] = {"connected": True, "signals_table": bool(row)}
    except Exception as e:  # pragma: no cover
        payload["db"] = {"connected": False, "error": str(e)}
    return payload


# -----------------------------------------------------------------------------
# Router includes (explicit + safe fallback)
# -----------------------------------------------------------------------------
def include_router_safe(module_path: str, attr: str = "router") -> None:
    """Try to import and include a router; ignore if missing/invalid."""
    try:
        mod = __import__(module_path, fromlist=[attr])
        router = getattr(mod, attr)
        app.include_router(router)
    except Exception:
        # silently skip: missing module or invalid router
        pass


# --- Critical routers (with visible error for stats) --------------------------
try:
    from app.api.routes_stats import router as stats_router  # type: ignore

    app.include_router(stats_router)
except Exception as e:  # pragma: no cover
    _stats_router_error = str(e)

    @app.get("/_router_stats_error")
    def _router_stats_error():
        return JSONResponse(
            {"error": f"routes_stats not loaded: {_stats_router_error}"},
            status_code=500,
        )


# --- Optional project routers (included if present) ---------------------------
include_router_safe("app.api.routes_health")        # /health/details
include_router_safe("app.api.routes_alerts")        # /alerts/*
include_router_safe("app.api.routes_ingest")        # /ingest/binance
include_router_safe("app.api.routes_ingest_bulk")   # /ingest/binance/bulk
include_router_safe("app.api.routes_market")        # /market/coverage, /ingest/binance/resume
include_router_safe("app.api.routes_scan")          # /scan
include_router_safe("app.api.routes_signals")       # /signals/*
include_router_safe("app.api.routes_export")        # /export
