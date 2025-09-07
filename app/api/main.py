"""FastAPI application entrypoint.

This file defines the FastAPI `app` first, then includes routers.
It also provides a robust `/health` endpoint that checks DB connectivity.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes_health import router as health_router
from app.api.routes_alerts import router as alerts_router

# Create app FIRST
app = FastAPI(title="TRADER_AI API", version="0.1.0")

# Optional: CORS (enable if you hit browser CORS in Streamlit/UI)
try:
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

# ---- Health endpoint with DB check --------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "ok", "ts": int(time.time())}
    try:
        from app.storage.db import ENGINE, init_db  # type: ignore
        init_db()
        with ENGINE.connect() as conn:
            row = conn.execute(
                # lightweight check; table may be empty
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
            ).fetchone()
        payload["db"] = {"connected": True, "signals_table": bool(row)}
    except Exception as e:  # pragma: no cover
        payload["db"] = {"connected": False, "error": str(e)}
    return payload

# ---- Router includes (guarded) ------------------------------------------
# Stats rolling router
try:
    from app.api.routes_stats import router as stats_router  # type: ignore
    app.include_router(stats_router)
except Exception as e:  # pragma: no cover
    # Expose why router didn't load (visible in /health via logs)
    @app.get("/_router_stats_error")
    def _router_stats_error():
        return JSONResponse({"error": f"routes_stats not loaded: {e}"}, status_code=500)

# Existing project routers (optional, included if present)
# e.g., routes_scan, routes_signals, routes_export
for mod_name, attr in [
    ("app.api.routes_scan", "router"),
    ("app.api.routes_signals", "router"),
    ("app.api.routes_export", "router"),
]:
    try:
        mod = __import__(mod_name, fromlist=[attr])
        router = getattr(mod, attr)
        app.include_router(router)
    except Exception:
        # silently skip if module not found or invalid
        pass
