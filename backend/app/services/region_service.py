from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from app.api import schemas
from app.ingestion.canonical_schema import VARIABLE_UNITS, CanonicalVariable
from app.ml import inference, registry
from app.services.event_intelligence import (
    add_event_intelligence,
    build_event_intelligence,
    find_analogs,
)
from app.ml.explain import top_factors_for
from app.utils import india_state_codes


NOT_TRAINED_MSG = (
    "No model has been trained yet. Upload a forecast dataset with matching observations "
    "to train one - nothing is shown until a real model exists."
)

NO_SCORE_MSG = (
    "A model exists but the canonical store has no forecast cycle that can be scored yet."
)


def _region_name(region_id: str) -> Optional[str]:
    rec = india_state_codes.resolve_by_region_id(region_id)
    return rec.region_name if rec else None


def _f(v) -> Optional[float]:
    if v is None:
        return None

    try:
        f = float(v)
    except (TypeError, ValueError):
        return None

    return None if not np.isfinite(f) else round(f, 6)


def _last_trained_at():
    rid = registry.current_run_id()

    if not rid:
        return None

    p = registry.run_dir(rid) / "manifest.json"

    if p.exists():
        import datetime as _dt

        return _dt.datetime.fromtimestamp(
            p.stat().st_mtime,
            tz=_dt.timezone.utc,
        )

    return None


def _risk_band_definitions(state) -> dict:
    cuts = state.thresholds.risk_band_cuts

    return {
        "low": f"bust risk below {cuts['medium'] * 100:.0f}%",
        "medium": (
            f"bust risk {cuts['medium'] * 100:.0f}% "
            f"to {cuts['high'] * 100:.0f}%"
        ),
        "high": f"bust risk {cuts['high'] * 100:.0f}% or above",
        "basis": (
            "Band edges are set from how this model's own predictions "
            "were spread on its validation data, not chosen by hand."
        ),
    }


def _not_trained_day(lead_time_days: int) -> schemas.RegionsResponse:
    return schemas.RegionsResponse(
        lead_time_days=lead_time_days,
        model_trained=False,
        regions=[],
        message=NOT_TRAINED_MSG,
    )


def _no_score_day(state, lead_time_days: int) -> schemas.RegionsResponse:
    return schemas.RegionsResponse(
        lead_time_days=lead_time_days,
        model_trained=True,
        current_run_id=state.run_id,
        last_trained_at=_last_trained_at(),
        regions=[],
        message=NO_SCORE_MSG,
    )


def _day_response(
    state,
    scored,
    available: list,
    lead_time_days: int,
) -> schemas.RegionsResponse:

    ev = scored.events[
        scored.events["lead_time_days"] == lead_time_days
    ]

    if ev.empty:
        covered = (
            f" This cycle covers day {available[0]}-{available[-1]}."
            if available
            else ""
        )

        return schemas.RegionsResponse(
            lead_time_days=lead_time_days,
            model_trained=True,
            current_run_id=state.run_id,
            last_trained_at=_last_trained_at(),
            init_date=scored.init_date.date(),
            risk_band_definitions=_risk_band_definitions(state),
            regions=[],
            available_lead_days=available,
            message=(
                f"The current forecast cycle has no "
                f"day-{lead_time_days} data.{covered}"
            ),
        )

    conf_cols = [
        c for c in ev.columns
        if c.startswith("conf_")
    ]

    regions = []

    for row in ev.itertuples():
        rid = str(row.region_id)

        conf = (
            np.nanmean(
                [
                    getattr(row, c, np.nan)
                    for c in conf_cols
                ]
            )
            if conf_cols
            else np.nan
        )

        regions.append(
            schemas.RegionSummary(
                region_id=rid,
                region_name=_region_name(rid),
                bust_probability=_f(row.bust_probability),
                risk_band=row.risk_band,
                confidence=_f(conf),
                dominant_variable=getattr(
                    row,
                    "dominant_variable",
                    None,
                ),
                data_available=True,
            )
        )

    regions.sort(
        key=lambda r: (
            r.bust_probability is None,
            -(r.bust_probability or 0),
        )
    )

    valid = ev["valid_date"].max()

    return schemas.RegionsResponse(
        lead_time_days=lead_time_days,
        model_trained=True,
        current_run_id=state.run_id,
        last_trained_at=_last_trained_at(),
        init_date=scored.init_date.date(),
        valid_date=(
            pd.to_datetime(valid).date()
            if pd.notna(valid)
            else None
        ),
        risk_band_definitions=_risk_band_definitions(state),
        regions=regions,
        available_lead_days=available,
    )


def _available_leads(scored) -> list:
    return sorted(
        int(d)
        for d in scored.events["lead_time_days"]
        .dropna()
        .unique()
    )


def get_regions(
    lead_time_days: int,
) -> schemas.RegionsResponse:

    state = inference.load_model_state()

    if state is None:
        return _not_trained_day(lead_time_days)

    scored = inference.score_latest_cycle(state)

    if scored is None or scored.events.empty:
        return _no_score_day(
            state,
            lead_time_days,
        )

    return _day_response(
        state,
        scored,
        _available_leads(scored),
        lead_time_days,
    )


def get_all_regions() -> schemas.AllRegionsResponse:

    state = inference.load_model_state()

    if state is None:
        return schemas.AllRegionsResponse(
            model_trained=False,
            message=NOT_TRAINED_MSG,
            days=[
                _not_trained_day(d)
                for d in range(1, 11)
            ],
        )

    scored = inference.score_latest_cycle(state)

    if scored is None or scored.events.empty:
        return schemas.AllRegionsResponse(
            model_trained=True,
            current_run_id=state.run_id,
            last_trained_at=_last_trained_at(),
            message=NO_SCORE_MSG,
            days=[
                _no_score_day(state, d)
                for d in range(1, 11)
            ],
        )

    available = _available_leads(scored)

    return schemas.AllRegionsResponse(
        model_trained=True,
        current_run_id=state.run_id,
        last_trained_at=_last_trained_at(),
        init_date=scored.init_date.date(),
        risk_band_definitions=_risk_band_definitions(state),
        available_lead_days=available,
        days=[
            _day_response(
                state,
                scored,
                available,
                d,
            )
            for d in range(1, 11)
        ],
    )


def get_region_detail(
    region_id: str,
) -> schemas.RegionDetailResponse:

    state = inference.load_model_state()

    if state is None:
        return schemas.RegionDetailResponse(
            region_id=region_id,
            region_name=_region_name(region_id),
            model_trained=False,
            message=NOT_TRAINED_MSG,
        )

    scored = inference.score_latest_cycle(state)

    if scored is None:
        return schemas.RegionDetailResponse(
            region_id=region_id,
            region_name=_region_name(region_id),
            model_trained=True,
            current_run_id=state.run_id,
            message=NO_SCORE_MSG,
        )

    # ------------------------------------------------------------------
    # Filter the scored data to the requested region.
    # ------------------------------------------------------------------

    pv = scored.per_variable[
        scored.per_variable["region_id"].astype(str)
        == region_id
    ]

    ev = scored.events[
        scored.events["region_id"].astype(str)
        == region_id
    ]

    if pv.empty and ev.empty:
        return schemas.RegionDetailResponse(
            region_id=region_id,
            region_name=_region_name(region_id),
            model_trained=True,
            current_run_id=state.run_id,
            init_date=scored.init_date.date(),
            message=(
                f"No forecast data for {region_id} "
                "in the current cycle."
            ),
        )

    # ------------------------------------------------------------------
    # Model validation metrics.
    # ------------------------------------------------------------------

    metrics = inference.model_validation_metrics(
        state
    ).get(
        "regressors",
        {},
    )

    # ------------------------------------------------------------------
    # Variable forecast series.
    # ------------------------------------------------------------------

    variables = []

    for var in state.variables:

        sub = pv[
            pv["variable"] == var
        ].sort_values(
            "lead_time_days"
        )

        m = metrics.get(var, {})

        if sub.empty:
            variables.append(
                schemas.VariableSeries(
                    variable=var,
                    available=False,
                    unit=_unit(var),
                    bust_threshold=_f(
                        state.thresholds.bust_threshold.get(
                            var
                        )
                    ),
                )
            )
            continue

        points = []

        for r in sub.itertuples():

            observed_value = getattr(
                r,
                "observed_value",
                None,
            )

            points.append(
                schemas.VariablePoint(
                    lead_time_days=int(
                        r.lead_time_days
                    ),
                    valid_date=(
                        pd.to_datetime(
                            r.valid_date
                        ).date()
                        if pd.notna(r.valid_date)
                        else None
                    ),
                    predicted_value=_f(
                        r.predicted_value
                    ),
                    observed_value=_f(
                        observed_value
                    ),
                    observed_status=(
                        getattr(
                            r,
                            "verification_status",
                            None,
                        )
                        if pd.notna(
                            observed_value
                        )
                        else None
                    ),
                    predicted_error=_f(
                        r.predicted_error
                    ),
                    confidence=_f(
                        r.confidence
                    ),
                    ensemble_spread=_f(
                        r.ensemble_spread
                    ),
                    ensemble_member_count=(
                        int(r.ensemble_member_count)
                        if pd.notna(
                            r.ensemble_member_count
                        )
                        else None
                    ),
                )
            )

        variables.append(
            schemas.VariableSeries(
                variable=var,
                available=True,
                unit=_unit(var),
                bust_threshold=_f(
                    state.thresholds.bust_threshold.get(
                        var
                    )
                ),
                model_mae=_f(
                    m.get("mae")
                ),
                model_rmse=_f(
                    m.get("rmse")
                ),
                model_r2=_f(
                    m.get("r2")
                ),
                metrics_split=m.get(
                    "split"
                ),
                points=points,
            )
        )

    # ------------------------------------------------------------------
    # M10 EVENT INTELLIGENCE
    #
    # Build forecast-side event rows FIRST.
    #
    # Important:
    # - Only predicted/forecast values are used.
    # - Observed values are NOT used for event classification.
    # - Event intelligence is calculated before constructing the
    #   BustProbabilityCurve.
    # ------------------------------------------------------------------

    event_rows = []

    for (
        lead,
        valid_date,
    ), group in pv.groupby(
        [
            "lead_time_days",
            "valid_date",
        ],
        observed=True,
    ):

        row = {
            "lead_time_days": int(lead),
            "valid_date": valid_date,
        }

        # Add forecast values for every canonical variable.
        for r in group.itertuples():

            predicted_value = getattr(
                r,
                "predicted_value",
                np.nan,
            )

            row[str(r.variable)] = predicted_value

        # Attach the bust probability corresponding to this lead.
        lead_events = ev[
            ev["lead_time_days"] == lead
        ]

        if not lead_events.empty:
            row["bust_probability"] = float(
                lead_events[
                    "bust_probability"
                ].iloc[0]
            )
        else:
            row["bust_probability"] = np.nan

        event_rows.append(row)

    if event_rows:

        event_df = add_event_intelligence(
            pd.DataFrame(event_rows)
        )

    else:

        event_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Create event lookup keyed by:
    #
    #     (lead_time_days, valid_date)
    #
    # This allows the exact event intelligence generated from the
    # forecast values to be attached to the corresponding curve point.
    # ------------------------------------------------------------------

    event_lookup = {}

    if not event_df.empty:

        for r in event_df.itertuples():

            raw_valid_date = getattr(
                r,
                "valid_date",
                None,
            )

            if (
                raw_valid_date is not None
                and pd.notna(raw_valid_date)
            ):
                valid_date = (
                    pd.to_datetime(
                        raw_valid_date
                    ).date()
                )
            else:
                valid_date = None

            event_lookup[
                (
                    int(r.lead_time_days),
                    valid_date,
                )
            ] = {
                "event_types": list(
                    getattr(
                        r,
                        "event_types",
                        [],
                    )
                    or []
                ),
                "potential_impacts": list(
                    getattr(
                        r,
                        "potential_impacts",
                        [],
                    )
                    or []
                ),
                "recommended_actions": list(
                    getattr(
                        r,
                        "recommended_actions",
                        [],
                    )
                    or []
                ),
            }

    # ------------------------------------------------------------------
    # Bust probability curve
    #
    # Event intelligence is now attached from event_lookup instead of
    # reading non-existent event fields from scored.events.
    # ------------------------------------------------------------------

    curve = []

    for r in ev.sort_values(
        "lead_time_days"
    ).itertuples():

        raw_valid_date = getattr(
            r,
            "valid_date",
            None,
        )

        if (
            raw_valid_date is not None
            and pd.notna(raw_valid_date)
        ):
            valid_date = (
                pd.to_datetime(
                    raw_valid_date
                ).date()
            )
        else:
            valid_date = None

        intelligence = event_lookup.get(
            (
                int(r.lead_time_days),
                valid_date,
            ),
            {
                "event_types": [],
                "potential_impacts": [],
                "recommended_actions": [],
            },
        )

        curve.append(
            schemas.BustProbabilityPoint(
                lead_time_days=int(
                    r.lead_time_days
                ),
                valid_date=valid_date,
                bust_probability=(
                    _f(
                        r.bust_probability
                    )
                    or 0.0
                ),
                risk_band=r.risk_band,
                dominant_variable=getattr(
                    r,
                    "dominant_variable",
                    None,
                ),
                event_types=intelligence[
                    "event_types"
                ],
                potential_impacts=intelligence[
                    "potential_impacts"
                ],
                recommended_actions=intelligence[
                    "recommended_actions"
                ],
            )
        )

    # ------------------------------------------------------------------
    # Headline event intelligence
    #
    # Prefer the highest-risk lead that actually contains an event.
    #
    # This prevents a high-risk but event-free lead from hiding a
    # lower/equal-risk lead that has meaningful weather intelligence.
    # ------------------------------------------------------------------

    headline = {}

    if not event_df.empty:

        event_rows_with_events = event_df[
            event_df["event_types"].apply(
                lambda x: bool(x)
            )
        ]

        if not event_rows_with_events.empty:

            if "bust_probability" in (
                event_rows_with_events.columns
            ):

                headline = (
                    event_rows_with_events
                    .sort_values(
                        "bust_probability",
                        ascending=False,
                    )
                    .iloc[0]
                    .to_dict()
                )

            else:

                headline = (
                    event_rows_with_events
                    .iloc[0]
                    .to_dict()
                )

        else:

            if "bust_probability" in event_df.columns:

                headline = (
                    event_df
                    .sort_values(
                        "bust_probability",
                        ascending=False,
                    )
                    .iloc[0]
                    .to_dict()
                )

            else:

                headline = (
                    event_df
                    .iloc[0]
                    .to_dict()
                )

    if headline:

        event_intel = build_event_intelligence(
            headline
        )

    else:

        event_intel = {
            "event_types": [],
            "evidence": {},
            "potential_impacts": [],
            "recommended_actions": [],
        }

    # ------------------------------------------------------------------
    # M10 ANALOG SEARCH
    #
    # This is enhancement-only.
    # No fabricated analog cases are returned.
    # Missing historical evaluation data must never break the API.
    # ------------------------------------------------------------------

    analog_cases = []

    try:

        eval_dir = (
            inference.resolve_path(
                state.manifest.get(
                    "data_dir",
                    "data",
                )
            )
            if hasattr(
                inference,
                "resolve_path",
            )
            else None
        )

        if eval_dir is None:

            from app.db.base import resolve_path

            eval_dir = resolve_path(
                "data"
            )

        history_path = (
            eval_dir
            / "analysis"
            / "eval_events"
        )

        files = (
            sorted(
                history_path.glob(
                    "*.parquet"
                )
            )
            if history_path.exists()
            else []
        )

        if files and headline:

            hist = pd.concat(
                [
                    pd.read_parquet(
                        f
                    )
                    for f in files
                ],
                ignore_index=True,
            )

            feature_cols = [
                c
                for c in [
                    "spread_mean",
                    "spread_max",
                    "lead_time_days",
                    "pred_err_rainfall_mm",
                    "pred_err_temperature_c",
                    "pred_err_pressure_hpa",
                ]
                if (
                    c in hist.columns
                    and c in headline
                )
            ]

            if feature_cols:

                analog_cases = find_analogs(
                    pd.Series(headline),
                    hist,
                    feature_cols,
                    k=5,
                )

    except Exception:

        # Analog search is enhancement-only and must never make
        # the region endpoint fail.
        analog_cases = []

    # ------------------------------------------------------------------
    # SHAP / top factors.
    # ------------------------------------------------------------------

    factors, method = _top_factors(
        state,
        region_id,
        ev,
    )

    # ------------------------------------------------------------------
    # Final region detail response.
    # ------------------------------------------------------------------

    return schemas.RegionDetailResponse(
        region_id=region_id,
        region_name=_region_name(
            region_id
        ),
        model_trained=True,
        current_run_id=state.run_id,
        init_date=scored.init_date.date(),
        variables=variables,
        bust_probability_curve=curve,
        top_factors=factors,
        top_factors_method=method,
        analog_cases=analog_cases,
        event_intelligence=event_intel,
        potential_impacts=event_intel.get(
            "potential_impacts",
            [],
        ),
        recommended_actions=event_intel.get(
            "recommended_actions",
            [],
        ),
    )


def _unit(var: str) -> Optional[str]:
    try:
        return VARIABLE_UNITS[
            CanonicalVariable(var)
        ]
    except (
        ValueError,
        KeyError,
    ):
        return None


def _top_factors(
    state,
    region_id: str,
    ev: pd.DataFrame,
):

    if (
        state.shap_summary.empty
        or ev.empty
    ):
        return [], None

    lead = int(
        ev.sort_values(
            "bust_probability",
            ascending=False,
        )[
            "lead_time_days"
        ].iloc[0]
    )

    raw = top_factors_for(
        state.shap_summary,
        region_id,
        lead,
        model="classifier",
        k=6,
    )

    factors = [
        schemas.TopFactor(**f)
        for f in raw
    ]

    method = (
        factors[0].method
        if factors
        else None
    )

    return factors, method