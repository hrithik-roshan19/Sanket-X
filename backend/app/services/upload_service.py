from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.api import schemas
from app.db.base import get_session
from app.db.models import TrainingRun
from app.ingestion.pipeline import IngestResult, confirm_mapping, ingest_upload
from app.ml import inference
from app.realtime.broadcaster import emit
from app.realtime.events import EventType
from app.services import alert_service

log = logging.getLogger(__name__)

_training_lock = threading.Lock()
_training_state = {"in_progress": False, "last_error": None}


def training_in_progress() -> bool:
    return _training_state["in_progress"]


def last_training_error() -> Optional[str]:
    return _training_state["last_error"]


def save_temp_upload(filename: str, data: bytes) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="fg-upload-"))
    dest = tmp_dir / Path(filename).name
    dest.write_bytes(data)
    return dest


def to_response(result: IngestResult) -> schemas.UploadResponse:
    status = "training_started" if result.status == "ingested" else result.status
    return schemas.UploadResponse(
        batch_id=result.batch_id,
        status=status,
        detected_format=result.detected_format,
        layout=result.layout,
        row_count_raw=result.row_count_raw,
        row_count_ingested=result.row_count_ingested,
        skipped_rows=result.skipped_rows,
        canonical_variables_found=result.canonical_variables_found,
        region_resolution_rate=round(result.region_resolution_rate, 4),
        source_profile_match=result.profile_match,
        mapping_proposals=[schemas.MappingProposal(**p) for p in result.mapping_proposals],
        notes=result.notes,
    )


def handle_upload(session: Session, path: Path, filename: str) -> IngestResult:
    emit(EventType.UPLOAD_RECEIVED, filename=filename)
    result = ingest_upload(session, path, filename)
    if result.status == "pending_confirmation":
        emit(EventType.MAPPING_PENDING, batch_id=result.batch_id,
             columns=len([p for p in result.mapping_proposals
                          if p["decision"] == "needs_confirmation"]))
    return result


def handle_confirm(session: Session, batch_id: str, mappings: list) -> IngestResult:
    return confirm_mapping(session, batch_id, mappings)


def run_retrain(batch_id: Optional[str] = None) -> None:
    if not _training_lock.acquire(blocking=False):
        log.info("retrain already running; skipping duplicate trigger")
        return
    _training_state["in_progress"] = True
    _training_state["last_error"] = None
    emit(EventType.TRAINING_STARTED, triggered_by_batch_id=batch_id)

    from app.ml.train_pipeline import full_retrain

    try:
        report = full_retrain(triggered_by_batch_id=batch_id, make_current=True)
        with get_session() as session:
            session.add(TrainingRun(
                run_id=report.run_id,
                status="complete" if report.status == "success" else "failed",
                triggered_by_batch_id=batch_id,
                validation_metrics_json={
                    "regressors": report.regressor_metrics,
                    "classifier": report.classifier_metrics,
                },
                error=report.error,
                finished_at=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc),
            ))

        if report.status == "success":
            inference.invalidate_caches()
            try:
                from app.services import replay_service
                replay_service.list_cycles()
            except Exception:  # noqa: BLE001
                log.exception("guided-replay re-warm after retrain failed (endpoint still works, just cold)")
            with get_session() as session:
                n_alerts = alert_service.persist_alerts_for_run(session, report.run_id)
            emit(
                EventType.TRAINING_COMPLETE,
                run_id=report.run_id,
                modelled_variables=report.modelled_variables,
                skipped_variables=report.skipped_variables,
                regions_updated=n_alerts,
                validation_metrics={
                    "classifier": report.classifier_metrics.get("test")
                    or report.classifier_metrics.get("val"),
                },
            )
            if n_alerts:
                emit(EventType.NEW_ALERT, run_id=report.run_id, count=n_alerts)
        else:
            _training_state["last_error"] = report.error or report.status
            emit(EventType.TRAINING_FAILED, run_id=report.run_id,
                 error=(report.error or report.status)[:500])
    except Exception as exc:  # noqa: BLE001
        _training_state["last_error"] = f"{type(exc).__name__}: {exc}"
        log.exception("retrain crashed")
        emit(EventType.TRAINING_FAILED, error=str(exc)[:500])
    finally:
        _training_state["in_progress"] = False
        _training_lock.release()
        shutil.rmtree(Path(tempfile.gettempdir()) / "fg-upload-noop", ignore_errors=True)
