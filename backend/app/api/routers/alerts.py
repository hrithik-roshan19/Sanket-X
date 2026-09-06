from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.api import schemas
from app.services import alert_service

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=schemas.AlertsResponse)
@router.get("/", response_model=schemas.AlertsResponse, include_in_schema=False)
def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    risk_band: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
) -> schemas.AlertsResponse:
    return alert_service.get_alerts(limit=limit, risk_band=risk_band)
