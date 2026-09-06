from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import schemas
from app.db.models import Alert as AlertRow
from app.ml import inference
from app.services.region_service import (
    NOT_TRAINED_MSG,
    NO_SCORE_MSG,
    _f,
    _region_name,
    _risk_band_definitions,
)


def get_alerts(limit: int = 50, risk_band: Optional[str] = None) -> schemas.AlertsResponse:
    now = datetime.now(timezone.utc)
    state = inference.load_model_state()
    if state is None:
        return schemas.AlertsResponse(generated_at=now, model_trained=False,
                                      alerts=[], message=NOT_TRAINED_MSG)

    scored = inference.score_latest_cycle(state)
    if scored is None or scored.events.empty:
        return schemas.AlertsResponse(generated_at=now, model_trained=True, alerts=[],
                                      risk_band_definitions=_risk_band_definitions(state),
                                      message=NO_SCORE_MSG)

    ev = scored.events.copy()
    if risk_band:
        ev = ev[ev["risk_band"] == risk_band.lower()]
    else:
        ev = ev[ev["risk_band"].isin(["medium", "high"])]

    ev = ev.sort_values("bust_probability", ascending=False).head(max(1, limit))
    alerts = [
        schemas.Alert(
            alert_id=f"{state.run_id}:{r.region_id}:{int(r.lead_time_days)}",
            region_id=str(r.region_id),
            region_name=_region_name(str(r.region_id)),
            lead_time_days=int(r.lead_time_days),
            valid_date=pd.to_datetime(r.valid_date).date() if pd.notna(r.valid_date) else None,
            bust_probability=_f(r.bust_probability) or 0.0,
            risk_band=r.risk_band,
            dominant_variable=getattr(r, "dominant_variable", None),
            created_at=now,
            training_run_id=state.run_id,
        )
        for r in ev.itertuples()
    ]
    return schemas.AlertsResponse(
        generated_at=now, model_trained=True,
        risk_band_definitions=_risk_band_definitions(state), alerts=alerts,
    )


def persist_alerts_for_run(session: Session, run_id: str) -> int:
    resp = get_alerts(limit=500)
    if not resp.model_trained or not resp.alerts:
        return 0
    session.query(AlertRow).filter(AlertRow.training_run_id == run_id).delete()
    for a in resp.alerts:
        session.add(AlertRow(
            training_run_id=run_id,
            region_id=a.region_id,
            region_name=a.region_name,
            lead_time_days=a.lead_time_days,
            bust_probability=a.bust_probability,
            risk_band=a.risk_band,
            dominant_variable=a.dominant_variable,
        ))
    session.flush()
    return len(resp.alerts)


def stored_alert_count(session: Session, run_id: str) -> int:
    return len(list(session.scalars(
        select(AlertRow).where(AlertRow.training_run_id == run_id)
    )))
