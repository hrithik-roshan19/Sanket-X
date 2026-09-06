from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api import schemas
from app.api.deps import get_db
from app.api.security import require_admin_key
from app.config import settings
from app.ingestion.parsers import ParseError
from app.services import upload_service

router = APIRouter(prefix="/api/upload", tags=["upload"])

MAX_BYTES = 200 * 1024 * 1024


@router.post("", response_model=schemas.UploadResponse)
@router.post("/", response_model=schemas.UploadResponse, include_in_schema=False)
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _admin: None = Depends(require_admin_key),
) -> schemas.UploadResponse:
    if not settings.upload_enabled:
        raise HTTPException(409, "file upload is disabled on this deployment")
    data = await file.read()
    if not data:
        raise HTTPException(400, "uploaded file is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file exceeds the {MAX_BYTES // (1024*1024)} MB limit")

    tmp = upload_service.save_temp_upload(file.filename or "upload.csv", data)
    try:
        result = upload_service.handle_upload(db, tmp, file.filename or tmp.name)
    except ParseError as exc:
        raise HTTPException(422, f"could not parse this file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"ingestion failed: {type(exc).__name__}: {exc}") from exc
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)

    db.commit()
    if result.should_retrain:
        background.add_task(upload_service.run_retrain, result.batch_id)
    return upload_service.to_response(result)


@router.post("/{batch_id}/confirm-mapping", response_model=schemas.UploadResponse)
def confirm_mapping(
    batch_id: str,
    body: schemas.ConfirmMappingRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    _admin: None = Depends(require_admin_key),
) -> schemas.UploadResponse:
    if not settings.upload_enabled:
        raise HTTPException(409, "file upload is disabled on this deployment")
    mappings = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in body.mappings]
    try:
        result = upload_service.handle_confirm(db, batch_id, mappings)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"canonicalisation failed: {type(exc).__name__}: {exc}") from exc

    db.commit()
    if result.should_retrain:
        background.add_task(upload_service.run_retrain, result.batch_id)
    return upload_service.to_response(result)
