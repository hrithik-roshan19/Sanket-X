from __future__ import annotations

from fastapi import APIRouter

from app.main_metrics import snapshot

router = APIRouter(prefix="/api/metrics", tags=["observability"])


@router.get("")
def metrics() -> dict:
    return snapshot()
