from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import get_session, resolve_path
from app.db.models import IngestRun
from app.ingestion.pipeline import ingest_upload
from app.live import gefs, observations
from app.ml import inference, registry
from app.realtime.broadcaster import emit
from app.realtime.events import EventType
from app.storage import parquet_store

log = logging.getLogger("forecastguard.live.orchestrator")

LIVE_DIR = resolve_path(settings.data_dir) / "live"

_ingest_lock = threading.Lock()


FORECAST_MAPPINGS = [
    {"source_column": "t2m_c", "variable": "temperature_c", "value_type": "forecast"},
    {"source_column": "rh2m_pct", "variable": "humidity_pct", "value_type": "forecast"},
    {"source_column": "apcp_mm", "variable": "rainfall_mm", "value_type": "forecast"},
    {"source_column": "mslp_hpa", "variable": "pressure_hpa", "value_type": "forecast"},
    {"source_column": "pwat_kgm2", "variable": "atmospheric_moisture_kgm2", "value_type": "forecast"},
    {"source_column": "soilw_vol_pct", "variable": "soil_moisture_pct", "value_type": "forecast"},
    {"source_column": "wspd10m_ms", "variable": "wind_speed_ms", "value_type": "forecast"},
    {"source_column": "wdir10m_deg", "variable": "wind_direction_deg", "value_type": "forecast"},
    {"source_column": "psfc_hpa", "role": "unmapped"},
]

OBSERVATION_MAPPINGS = [
    {"source_column": "t2m_c", "variable": "temperature_c", "value_type": "observed"},
    {"source_column": "rh2m_pct", "variable": "humidity_pct", "value_type": "observed"},
    {"source_column": "precip_mm", "variable": "rainfall_mm", "value_type": "observed"},
    {"source_column": "mslp_hpa", "variable": "pressure_hpa", "value_type": "observed"},
    {"source_column": "pwat_kgm2", "variable": "atmospheric_moisture_kgm2", "value_type": "observed"},
    {"source_column": "soil_moisture_pct", "variable": "soil_moisture_pct", "value_type": "observed"},
    {"source_column": "wspd10m_ms", "variable": "wind_speed_ms", "value_type": "observed"},
    {"source_column": "wdir10m_deg", "variable": "wind_direction_deg", "value_type": "observed"},
    {"source_column": "psfc_hpa", "role": "unmapped"},
    {"source_column": "source", "role": "unmapped"},
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _start_run(session: Session, kind: str, target: str, trigger: str) -> IngestRun:
    run = IngestRun(kind=kind, target=target, trigger=trigger, status="running")
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def _finish_run(session: Session, run: IngestRun, status: str, **fields) -> None:
    run.status = status
    run.finished_at = _now()
    for k, v in fields.items():
        setattr(run, k, v)
    session.commit()


def cycle_target(init: date, cycle_hour: str) -> str:
    return f"{init.isoformat()} {cycle_hour}"


def cycle_already_ingested(session: Session, init: date, cycle_hour: str) -> bool:
    return session.query(IngestRun).filter(
        IngestRun.kind == "forecast",
        IngestRun.target == cycle_target(init, cycle_hour),
        IngestRun.status == "complete",
    ).first() is not None


def run_forecast_cycle(
    init: Optional[date] = None,
    cycle_hour: Optional[str] = None,
    trigger: str = "schedule",
    force: bool = False,
) -> dict:
    if init is None or cycle_hour is None:
        init, cycle_hour = gefs.latest_expected_cycle(
            lag_hours=settings.live_gefs_publish_lag_hours,
            cycles=settings.gefs_cycle_list,
        )
    target = cycle_target(init, cycle_hour)

    with get_session() as session:
        if not force and cycle_already_ingested(session, init, cycle_hour):
            log.info("cycle %s already ingested; skipping", target)
            return {"status": "skipped", "target": target, "reason": "already ingested"}
        run = _start_run(session, "forecast", target, trigger)
        run_id = run.id

    try:
        members = settings.gefs_member_list
        # Ensemble spread is a learned feature. Never score a model trained on one
        # ensemble size with a different size unless the model explicitly declares it.
        current_run = registry.current_run_id()
        if current_run:
            manifest_path = registry.run_dir(current_run) / "manifest.json"
            if manifest_path.exists():
                import json
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                trained_members = manifest.get("training_ensemble_member_count")
                if trained_members and len(members) != int(trained_members):
                    msg = (
                        f"ensemble-size mismatch: model trained on {trained_members} members, "
                        f"live feed configured for {len(members)}; retrain before changing live size"
                    )
                    with get_session() as session:
                        run = session.get(IngestRun, run_id)
                        _finish_run(session, run, "failed", error=msg)
                    return {"status": "blocked", "target": target, "reason": msg}

        frame, report = gefs.fetch_cycle(
            init, cycle_hour, members=members,
            workers=settings.live_download_workers,
        )
        path = gefs.write_cycle_csv(frame, init, cycle_hour, LIVE_DIR)

        with get_session() as session:
            result = ingest_upload(session, path, path.name,
                                   confirmed_mappings=FORECAST_MAPPINGS)
            detail = (
                f"transport={report.transport} steps={report.steps_fetched}/"
                f"{report.steps_expected} {report.bytes_downloaded/1048576:.1f}MB "
                f"in {report.seconds:.0f}s; members={','.join(members)}; "
                f"variables={','.join(result.canonical_variables_found)}"
            )
            if report.steps_recovered:
                detail += f"; recovered_from_s3={report.steps_recovered}"
            if report.missing_steps:
                detail += f"; missing_steps={len(report.missing_steps)}"
            if report.undersampled:
                detail += f"; undersampled={len(report.undersampled)}"
            run = session.get(IngestRun, run_id)
            _finish_run(session, run, "complete",
                        upload_batch_id=result.batch_id,
                        rows_ingested=result.row_count_ingested,
                        detail=detail)

        inference.invalidate_caches()
        emit(EventType.UPLOAD_RECEIVED, filename=path.name, source="live-gefs")
        emit(EventType.NEW_ALERT, cycle=target, rows=result.row_count_ingested)
        log.info("ingested cycle %s: %s rows", target, result.row_count_ingested)
        return {
            "status": "complete", "target": target,
            "rows_ingested": result.row_count_ingested,
            "batch_id": result.batch_id, "transport": report.transport,
            "steps": f"{report.steps_fetched}/{report.steps_expected}",
            "missing_steps": report.missing_steps,
            "undersampled": report.undersampled[:20],
            "steps_fetched": report.steps_fetched,
            "steps_expected": report.steps_expected,
            "steps_recovered": report.steps_recovered,
            "step_completeness": round(report.step_completeness, 4),
            "undersampled_count": len(report.undersampled),
            "undersampled_sum_count": len(report.undersampled_sums),
            "seconds": round(report.seconds, 1),
        }

    except gefs.CycleUnavailable as exc:
        with get_session() as session:
            _finish_run(session, session.get(IngestRun, run_id), "skipped", detail=str(exc))
        log.info("cycle %s not available: %s", target, exc)
        return {"status": "skipped", "target": target, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001
        with get_session() as session:
            _finish_run(session, session.get(IngestRun, run_id), "failed",
                        error=f"{type(exc).__name__}: {exc}")
        log.exception("cycle %s failed", target)
        return {"status": "failed", "target": target, "error": str(exc)}


def run_observation_refresh(
    tier: str = "final",
    days_back: Optional[int] = None,
    trigger: str = "schedule",
    retrain: Optional[bool] = None,
) -> dict:
    if days_back is None:
        days_back = (settings.live_obs_final_days if tier == "final"
                     else settings.live_obs_provisional_days)
    start, end = observations.default_window(days_back, tier)
    kind = f"observations_{tier}"
    target = f"{start.isoformat()}..{end.isoformat()}"

    with get_session() as session:
        run = _start_run(session, kind, target, trigger)
        run_id = run.id

    try:
        frame, report = observations.fetch_observations(start, end, tier=tier)
        if frame.empty:
            with get_session() as session:
                _finish_run(session, session.get(IngestRun, run_id), "skipped",
                            detail=f"no observations returned for {target}")
            return {"status": "skipped", "target": target, "reason": "no rows returned"}

        path = observations.write_observations_csv(frame, tier, start, end, LIVE_DIR)
        with get_session() as session:
            result = ingest_upload(session, path, path.name,
                                   confirmed_mappings=OBSERVATION_MAPPINGS,
                                   verification_status=tier)
            detail = (f"tier={tier} cities={report.cities} rows={result.row_count_ingested} "
                      f"in {report.seconds:.0f}s")
            if report.failures:
                detail += f"; failed_cities={','.join(report.failures)}"
            _finish_run(session, session.get(IngestRun, run_id), "complete",
                        upload_batch_id=result.batch_id,
                        rows_ingested=result.row_count_ingested,
                        detail=detail)

        inference.invalidate_caches()
        emit(EventType.UPLOAD_RECEIVED, filename=path.name, source=f"live-obs-{tier}")

        should_retrain = (
            retrain if retrain is not None
            else (tier == "final"
                  and result.row_count_ingested >= settings.live_retrain_min_new_rows)
        )
        out = {"status": "complete", "target": target, "tier": tier,
               "rows_ingested": result.row_count_ingested,
               "batch_id": result.batch_id, "retrain_triggered": bool(should_retrain)}

        if should_retrain:
            from app.services.upload_service import run_retrain
            threading.Thread(target=run_retrain, args=(result.batch_id,),
                             name="live-retrain", daemon=True).start()
        return out

    except Exception as exc:  # noqa: BLE001
        with get_session() as session:
            _finish_run(session, session.get(IngestRun, run_id), "failed",
                        error=f"{type(exc).__name__}: {exc}")
        log.exception("observation refresh (%s) failed", tier)
        return {"status": "failed", "target": target, "error": str(exc)}


def run_due_work(trigger: str = "schedule") -> dict:
    if not _ingest_lock.acquire(blocking=False):
        log.info("ingest already running; this tick does nothing")
        return {"status": "busy"}
    try:
        out: dict = {"forecast": run_forecast_cycle(trigger=trigger)}

        with get_session() as session:
            out["provisional"] = (
                run_observation_refresh("provisional", trigger=trigger)
                if _due(session, "observations_provisional", hours=6) else {"status": "not_due"}
            )
            final_due = _due(session, "observations_final", hours=24)
        out["final"] = (run_observation_refresh("final", trigger=trigger)
                        if final_due else {"status": "not_due"})
        return out
    finally:
        _ingest_lock.release()


def _due(session: Session, kind: str, hours: float) -> bool:
    last = (session.query(IngestRun)
            .filter(IngestRun.kind == kind, IngestRun.status == "complete")
            .order_by(IngestRun.started_at.desc()).first())
    if last is None or last.finished_at is None:
        return True
    finished = last.finished_at
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    return (_now() - finished) >= timedelta(hours=hours)


def backfill_cycles(count: int, stride_hours: int = 24, trigger: str = "backfill") -> list:
    wanted = gefs.recent_cycles(
        count, lag_hours=settings.live_gefs_publish_lag_hours,
        cycles=settings.gefs_cycle_list, stride_hours=stride_hours,
    )
    results = []
    for init, hh in reversed(wanted):
        results.append(run_forecast_cycle(init, hh, trigger=trigger))
    return results


def _run_row(r) -> dict:
    return {
        "id": r.id, "kind": r.kind, "target": r.target, "status": r.status,
        "trigger": r.trigger, "rows_ingested": r.rows_ingested,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "seconds": (round((r.finished_at - r.started_at).total_seconds(), 1)
                    if r.started_at and r.finished_at else None),
        "detail": r.detail, "error": r.error,
    }


def recent_runs(limit: int = 25) -> list:
    with get_session() as session:
        rows = (session.query(IngestRun)
                .order_by(IngestRun.started_at.desc())
                .limit(max(1, min(int(limit), 200))).all())
        return [_run_row(r) for r in rows]


def feed_status() -> dict:
    with get_session() as session:
        def last(kind: str):
            r = (session.query(IngestRun)
                 .filter(IngestRun.kind == kind)
                 .order_by(IngestRun.started_at.desc()).first())
            if r is None:
                return None
            return {"target": r.target, "status": r.status, "trigger": r.trigger,
                    "rows_ingested": r.rows_ingested,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                    "detail": r.detail, "error": r.error}

        completed = (session.query(IngestRun.target)
                     .filter(IngestRun.kind == "forecast", IngestRun.status == "complete")
                     .order_by(IngestRun.target.desc()).first())
        last_cycle = completed[0] if completed else None

    return {
        "enabled": settings.live_ingest_enabled,
        "cycles_watched": settings.gefs_cycle_list,
        "members": settings.gefs_member_list,
        "last_forecast": last("forecast"),
        "last_observations_provisional": last("observations_provisional"),
        "last_observations_final": last("observations_final"),
        "last_cycle_ingested": last_cycle,
        "latest_forecast_init_date": (
            d.isoformat() if (d := parquet_store.latest_forecast_init_date()) else None
        ),
    }
