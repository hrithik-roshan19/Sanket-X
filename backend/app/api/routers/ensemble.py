from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query

from app.api import schemas
from app.services import ensemble_service

router = APIRouter(prefix="/api/ensemble", tags=["ensemble"])


@router.get("", response_model=schemas.EnsembleDivergenceResponse)
@router.get("/", response_model=schemas.EnsembleDivergenceResponse, include_in_schema=False)
def divergence(
    init_date: Optional[date] = Query(None, description="Forecast cycle; defaults to the latest scored."),
    region_id: Optional[str] = Query(None, description="Prefer this region if it is chartable."),
) -> schemas.EnsembleDivergenceResponse:
    return ensemble_service.get_divergence(init_date=init_date, region_id=region_id)
