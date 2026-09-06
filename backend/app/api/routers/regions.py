from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api import schemas
from app.services import region_service
from app.utils import india_state_codes

router = APIRouter(prefix="/api/regions", tags=["regions"])


@router.get("", response_model=schemas.RegionsResponse)
@router.get("/", response_model=schemas.RegionsResponse, include_in_schema=False)
def list_regions(
    lead_time_days: int = Query(1, ge=1, le=10, description="forecast lead day, 1-10"),
) -> schemas.RegionsResponse:
    return region_service.get_regions(lead_time_days)


@router.get("/all", response_model=schemas.AllRegionsResponse)
def list_regions_all_lead_days() -> schemas.AllRegionsResponse:
    return region_service.get_all_regions()


@router.get("/{region_id}", response_model=schemas.RegionDetailResponse)
def region_detail(region_id: str) -> schemas.RegionDetailResponse:
    if india_state_codes.resolve_by_region_id(region_id) is None:
        raise HTTPException(404, f"unknown region_id '{region_id}' (expected ISO 3166-2:IN, e.g. IN-MH)")
    return region_service.get_region_detail(region_id)
