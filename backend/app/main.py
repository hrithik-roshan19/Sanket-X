from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from uuid import uuid4
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers import alerts, ensemble, ingest, metrics, model_status, regions, replay, upload, ws
from app.config import settings
from app.db.base import init_db
from app.ml import registry
from app.realtime.broadcaster import manager
from app.main_metrics import record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("forecastguard")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    manager.bind_loop(asyncio.get_running_loop())

    try:
        from app.utils.geo import get_resolver
        get_resolver()
        log.info("geo index ready")
    except Exception:  # noqa: BLE001
        log.exception("geo index failed to build; region resolution will degrade")

    log.info("current model run: %s", registry.current_run_id() or "none (no model trained yet)")

    if registry.current_run_id() and settings.warm_caches_on_startup:
        def _warm_replay() -> None:
            try:
                from app.services import replay_service
                n = len(replay_service.list_cycles())
                log.info("guided-replay cache warm: %d cycles ranked", n)
                from app.services import ensemble_service
                ensemble_service.get_divergence()
                log.info("ensemble-divergence cache warm")
            except Exception:  # noqa: BLE001
                log.exception("guided-replay warm failed (endpoint still works, just cold)")

        threading.Thread(target=_warm_replay, name="replay-warm", daemon=True).start()
    elif registry.current_run_id():
        log.info("startup cache warm disabled; replay and ensemble will be cold on first call")

    from app.live.scheduler import scheduler
    scheduler.start()
    try:
        yield
    finally:
        scheduler.stop()


app = FastAPI(
    title="Sanket",
    version="0.7.0",
    lifespan=lifespan,
    description=(
        "Medium-range (Day 1-10) forecast-bust detection for Indian regions. "
        "SIH 2026 PS 26079. Every number served here traces to a real upload and a real "
        "trained model; when neither exists the API says so explicitly."
    ),
)


@app.middleware("http")
async def observability_middleware(request, call_next):
    started = time.perf_counter()
    rid = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = rid
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        record(request.url.path, 500, elapsed)
        raise
    elapsed = (time.perf_counter() - started) * 1000
    record(request.url.path, response.status_code, elapsed)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(regions.router)
app.include_router(alerts.router)
app.include_router(model_status.router)
app.include_router(ingest.router)
app.include_router(replay.router)
app.include_router(ensemble.router)
app.include_router(ws.router)
app.include_router(metrics.router)


def _build_commit() -> str:
    return os.getenv("RENDER_GIT_COMMIT") or os.getenv("BUILD_COMMIT") or ""


def _rss_mb() -> "float | None":
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


# HEAD as well as GET: uptime monitors send HEAD, and @app.get alone answers it with 405.
@app.api_route("/api/health", methods=["GET", "HEAD"], tags=["health"])
def health() -> dict:
    return {
        "status": "ok",
        "model_trained": registry.current_run_id() is not None,
        "current_run_id": registry.current_run_id(),
        "websocket_clients": manager.connection_count,
        "commit": _build_commit(),
        "memory_mb": _rss_mb(),
    }


_STATIC_DIR = Path(__file__).resolve().parent / "static"

if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith(("api/", "ws")):
            raise HTTPException(status_code=404, detail=f"No such endpoint: /{full_path}")

        candidate = (_STATIC_DIR / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(_STATIC_DIR):
            return FileResponse(candidate)
        return FileResponse(_STATIC_DIR / "index.html")

    log.info("serving built frontend from %s", _STATIC_DIR)
else:
    log.info("no built frontend at %s; API only (use `npm run dev` for the UI)", _STATIC_DIR)
