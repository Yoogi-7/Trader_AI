"""FastAPI application entrypoint (fixed error handler).

- Defines app first
- Includes routers
- Health endpoint
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="TRADER_AI API", version="0.2.1")

# Optional CORS
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
    pass


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "TRADER_AI",
        "version": "0.2.1",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "ts": int(time.time()),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "ok", "ts": int(time.time())}
    try:
        from app.storage.db import ENGINE, init_db  # lazy import
        init_db()
        with ENGINE.connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
            ).fetchone()
        payload["db"] = {"connected": True, "signals_table": bool(row)}
    except Exception as e:  # pragma: no cover
        payload["db"] = {"connected": False, "error": str(e)}
    return payload


# Routers
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

for mod_name, attr in [
    ("app.api.routes_health", "router"),
    ("app.api.routes_alerts", "router"),
    ("app.api.routes_ingest", "router"),
    ("app.api.routes_ingest_bulk", "router"),
    ("app.api.routes_market", "router"),
    ("app.api.routes_scan", "router"),
    ("app.api.routes_signals", "router"),
    ("app.api.routes_export", "router"),
]:
    try:
        mod = __import__(mod_name, fromlist=[attr])
        router = getattr(mod, attr)
        app.include_router(router)
    except Exception:
        pass
