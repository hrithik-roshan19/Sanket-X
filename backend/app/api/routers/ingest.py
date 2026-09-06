from __future__ import annotations

import threading
from datetime import date
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.config import settings
from app.api.security import require_admin_key
from app.live import orchestrator
from app.live.scheduler import scheduler

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


@router.get("/status")
def ingest_status() -> dict:
    return {**orchestrator.feed_status(), "scheduler": scheduler.status()}


@router.get("/runs")
def ingest_runs(limit: int = Query(25, ge=1, le=200)) -> dict:
    return {"runs": orchestrator.recent_runs(limit)}


def _require_live_ingest() -> None:
    if not settings.live_ingest_enabled:
        raise HTTPException(
            409,
            "Live ingestion is disabled on this deployment. Forecast cycles and "
            "observations are pulled in CI and shipped as a release artifact; this "
            "instance only serves them. Set LIVE_INGEST_ENABLED=true to enable pulling.",
        )


@router.post("/run-cycle")
def run_cycle(
    background: BackgroundTasks,
    init_date: Optional[date] = Query(None, description="UTC cycle date; defaults to the newest published"),
    cycle_hour: Optional[str] = Query(None, pattern="^(00|06|12|18)$"),
    force: bool = Query(False, description="re-ingest even if this cycle is already stored"),
    wait: bool = Query(False, description="run inline and return the result instead of scheduling it"),
    _admin: None = Depends(require_admin_key),
) -> dict:
    _require_live_ingest()
    if cycle_hour is not None and init_date is None:
        raise HTTPException(400, "cycle_hour requires init_date")

    if wait:
        return orchestrator.run_forecast_cycle(init_date, cycle_hour,
                                               trigger="manual", force=force)
    background.add_task(orchestrator.run_forecast_cycle, init_date, cycle_hour,
                        "manual", force)
    return {"status": "started",
            "target": (orchestrator.cycle_target(init_date, cycle_hour)
                       if init_date and cycle_hour else "newest published cycle"),
            "note": "watch /api/ingest/status or the WebSocket for completion"}


@router.post("/refresh-observations")
def refresh_observations(
    background: BackgroundTasks,
    tier: str = Query("final", pattern="^(final|provisional)$"),
    days_back: Optional[int] = Query(None, ge=1, le=92),
    wait: bool = Query(False),
    _admin: None = Depends(require_admin_key),
) -> dict:
    _require_live_ingest()
    if wait:
        return orchestrator.run_observation_refresh(tier, days_back, trigger="manual")
    background.add_task(orchestrator.run_observation_refresh, tier, days_back, "manual")
    return {"status": "started", "tier": tier}


@router.post("/backfill")
def backfill(
    background: BackgroundTasks,
    cycles: int = Query(..., ge=1, le=120, description="how many past cycles to pull"),
    stride_hours: int = Query(24, ge=6, le=168),
    _admin: None = Depends(require_admin_key),
) -> dict:
    _require_live_ingest()
    threading.Thread(
        target=orchestrator.backfill_cycles, args=(cycles, stride_hours),
        name="live-backfill", daemon=True,
    ).start()
    return {"status": "started", "cycles": cycles, "stride_hours": stride_hours,
            "note": "progress appears in /api/ingest/status; re-running resumes where it stopped"}
