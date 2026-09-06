from __future__ import annotations

from typing import Optional

import pandas as pd

from app.api import schemas
from app.ml import inference
from app.services.region_service import (
    NOT_TRAINED_MSG,
    _f,
    _region_name,
    _risk_band_definitions,
    _unit,
)


_MAX_CYCLES = 10

_cycles_memo: "tuple[str, list[schemas.ReplayCycleSummary]] | None" = None


def _cycle_summary(
    state,
    init,
) -> Optional[schemas.ReplayCycleSummary]:
    sc = inference.score_cycle(state, init)

    if sc is None or sc.events.empty:
        return None

    ev = sc.events

    leads = sorted(
        int(d)
        for d in ev["lead_time_days"].dropna().unique()
    )

    # Use the same deterministic ordering used by get_replay().
    ev_ordered = ev.sort_values(
        ["lead_time_days", "bust_probability"],
        ascending=[True, False],
        kind="mergesort",
    )

    peak = ev_ordered.iloc[0]

    peak_lead = int(peak["lead_time_days"])

    at_peak = ev[
        ev["lead_time_days"] == peak_lead
    ]

    verified_leads = 0
    peak_abs_err = None
    growth = 0.0

    pv = sc.per_variable

    if not pv.empty and "observed_value" in pv.columns:
        vmask = (
            pv["observed_value"].notna()
            & pv["predicted_value"].notna()
        )

        verified_leads = int(
            pv.loc[
                vmask,
                "lead_time_days",
            ].nunique()
        )

        dom = peak.get("dominant_variable")

        prz = pv[
            vmask
            & (
                pv["region_id"].astype(str)
                == str(peak["region_id"])
            )
            & (pv["variable"] == dom)
        ]

        if not prz.empty:
            peak_abs_err = float(
                (
                    prz["predicted_value"]
                    - prz["observed_value"]
                ).abs().mean()
            )

    near = ev.loc[
        ev["lead_time_days"] <= 3,
        "bust_probability",
    ].mean()

    far = ev.loc[
        ev["lead_time_days"] >= 4,
        "bust_probability",
    ].mean()

    if pd.notna(near) and pd.notna(far):
        growth = float(far - near)

    return schemas.ReplayCycleSummary(
        init_date=pd.Timestamp(init).date(),
        lead_days=leads,
        n_regions=int(
            ev["region_id"].nunique()
        ),
        peak_bust_probability=_f(
            peak["bust_probability"]
        ),
        peak_lead_day=peak_lead,
        peak_region_id=str(
            peak["region_id"]
        ),
        peak_region_name=_region_name(
            str(peak["region_id"])
        ),
        n_high_regions_peak=int(
            (
                at_peak["risk_band"]
                == "high"
            ).sum()
        ),
        verified=verified_leads > 0,
        verified_lead_days=verified_leads,
        peak_region_abs_error=_f(
            peak_abs_err
        ),
        medium_range_growth=round(
            growth,
            4,
        ),
    )


def list_cycles() -> list[schemas.ReplayCycleSummary]:
    global _cycles_memo

    state = inference.load_model_state()

    if state is None:
        return []

    if (
        _cycles_memo is not None
        and _cycles_memo[0] == state.run_id
    ):
        return _cycles_memo[1]

    out: list[schemas.ReplayCycleSummary] = []

    for init in inference.available_cycles()[
        :_MAX_CYCLES
    ]:
        s = _cycle_summary(
            state,
            init,
        )

        if s is not None:
            out.append(s)

    out.sort(
        key=lambda c: (
            c.verified,
            round(
                max(
                    c.medium_range_growth,
                    0.0,
                ),
                3,
            ),
            round(
                c.peak_bust_probability
                or 0.0,
                3,
            ),
            c.verified_lead_days,
        ),
        reverse=True,
    )

    _cycles_memo = (
        state.run_id,
        out,
    )

    return out


def get_replay(
    init_date: Optional[str] = None,
    focus_region: Optional[str] = None,
) -> schemas.ReplayResponse:
    state = inference.load_model_state()

    if state is None:
        return schemas.ReplayResponse(
            model_trained=False,
            message=NOT_TRAINED_MSG,
        )

    cycles = list_cycles()

    if not cycles:
        return schemas.ReplayResponse(
            model_trained=True,
            current_run_id=state.run_id,
            available_cycles=[],
            message=(
                "No forecast cycle in the "
                "store can be scored yet."
            ),
        )

    target = (
        init_date
        or str(cycles[0].init_date)
    )

    sc = inference.score_cycle(
        state,
        target,
    )

    if sc is None or sc.events.empty:
        return schemas.ReplayResponse(
            model_trained=True,
            current_run_id=state.run_id,
            available_cycles=cycles,
            message=(
                f"Cycle {target} is not in "
                "the store or has nothing to score."
            ),
        )

    cuts = state.thresholds.risk_band_cuts

    ev = sc.events.sort_values(
        [
            "lead_time_days",
            "bust_probability",
        ],
        ascending=[
            True,
            False,
        ],
        kind="mergesort",
    )

    steps: list[schemas.ReplayLeadStep] = []

    prev: dict[str, float] = {}
    prev_dom = ""

    for lead, g in ev.groupby(
        "lead_time_days",
        sort=True,
    ):
        cur = {
            str(r.region_id): float(
                r.bust_probability
            )
            for r in g.itertuples()
        }

        cur_dom = _pretty(
            g.iloc[0].get(
                "dominant_variable"
            )
        )

        regs = [
            schemas.ReplayRegionStep(
                region_id=str(
                    r.region_id
                ),
                region_name=_region_name(
                    str(r.region_id)
                ),
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
            )
            for r in g.itertuples()
        ]

        vd = g["valid_date"].iloc[0]

        steps.append(
            schemas.ReplayLeadStep(
                lead_time_days=int(lead),
                valid_date=(
                    pd.to_datetime(vd).date()
                    if pd.notna(vd)
                    else None
                ),
                regions=regs,
                n_high=int(
                    (
                        g["risk_band"]
                        == "high"
                    ).sum()
                ),
                n_medium=int(
                    (
                        g["risk_band"]
                        == "medium"
                    ).sum()
                ),
                mean_bust_probability=_f(
                    g["bust_probability"].mean()
                ),
                narration=_narrate(
                    int(lead),
                    g,
                    prev,
                    cur,
                    cuts["high"],
                    prev_dom,
                ),
            )
        )

        prev = cur
        prev_dom = cur_dom

    default_focus, focus_options = _build_focus(
        sc,
        state,
        focus_region,
    )

    return schemas.ReplayResponse(
        model_trained=True,
        current_run_id=state.run_id,
        init_date=pd.Timestamp(
            target
        ).date(),
        available_cycles=cycles,
        steps=steps,
        focus=default_focus,
        focus_options=focus_options,
        risk_band_definitions=_risk_band_definitions(
            state
        ),
        summary_narration=_summarise(
            sc,
            steps,
        ),
    )


def _pretty(var) -> str:
    return (
        str(var).replace("_", " ")
        if isinstance(var, str) and var
        else ""
    )


def _narrate(
    lead: int,
    g: pd.DataFrame,
    prev: dict,
    cur: dict,
    hi: float,
    prev_dom: str,
) -> str:
    top = g.iloc[0]

    tname = (
        _region_name(
            str(top["region_id"])
        )
        or str(top["region_id"])
    )

    bits: list[str] = []

    if prev:
        deltas = {
            rid: cur[rid] - prev[rid]
            for rid in cur
            if rid in prev
        }

        if deltas:
            rid_up, dlt = max(
                deltas.items(),
                key=lambda kv: kv[1],
            )

            if dlt >= 0.05:
                nm = (
                    _region_name(rid_up)
                    or rid_up
                )

                bits.append(
                    f"bust risk over {nm} "
                    f"climbs "
                    f"{prev[rid_up]:.2f}"
                    f"→"
                    f"{cur[rid_up]:.2f}"
                )

    if not bits:
        verb = (
            "opens with"
            if not prev
            else "still carries"
        )

        bits.append(
            f"{tname} {verb} "
            "the highest bust risk at "
            f"{float(top['bust_probability']):.2f}"
        )

    dom = _pretty(
        top.get("dominant_variable")
    )

    if dom and dom != prev_dom:
        bits.append(
            (
                f"now driven by {dom} error"
                if prev_dom
                else f"driven by {dom} error"
            )
        )

    n_high = int(
        (
            g["bust_probability"] >= hi
        ).sum()
    )

    if n_high:
        verb = (
            ""
            if not prev
            else "now "
        )

        bits.append(
            f"{n_high} region"
            f"{'s' if n_high != 1 else ''} "
            f"{verb}high-risk".replace(
                "  ",
                " ",
            )
        )

    return (
        f"Day {lead} — "
        + "; ".join(bits)
        + "."
    )


def _summarise(
    sc: inference.ScoredCycle,
    steps: list[schemas.ReplayLeadStep],
) -> str:
    ev = sc.events

    # Same deterministic ordering as get_replay().
    ev_ordered = ev.sort_values(
        [
            "lead_time_days",
            "bust_probability",
        ],
        ascending=[
            True,
            False,
        ],
        kind="mergesort",
    )

    peak = ev_ordered.iloc[0]

    pr = (
        _region_name(
            str(peak["region_id"])
        )
        or str(peak["region_id"])
    )

    pl = int(
        peak["lead_time_days"]
    )

    first = (
        steps[0].mean_bust_probability
        or 0.0
    )

    last = (
        steps[-1].mean_bust_probability
        or 0.0
    )

    trend = (
        "climbs"
        if last > first + 0.02
        else (
            "eases"
            if last < first - 0.02
            else "holds"
        )
    )

    dom = _pretty(
        peak.get("dominant_variable")
    )

    dtxt = (
        f", led by {dom} error."
        if dom
        else "."
    )

    verified = (
        not sc.per_variable.empty
        and "observed_value"
        in sc.per_variable.columns
        and sc.per_variable[
            "observed_value"
        ].notna().any()
    )

    tail = (
        " Its forecast-vs-observed track "
        "is charted below."
        if verified
        else (
            " This cycle has not verified yet, "
            "so no observed track is drawn."
        )
    )

    return (
        f"Init "
        f"{pd.Timestamp(sc.init_date).date()}: "
        f"mean bust risk {trend} from "
        f"{first:.2f} on Day "
        f"{steps[0].lead_time_days} to "
        f"{last:.2f} on Day "
        f"{steps[-1].lead_time_days}. "
        f"It peaks at "
        f"{float(peak['bust_probability']):.2f} "
        f"over {pr} on Day {pl}"
        f"{dtxt}{tail}"
    )


def _focus_variable_for(
    v_region: pd.DataFrame,
    dominant: Optional[str],
    state,
) -> Optional[str]:
    if v_region.empty:
        return None

    # Prefer the dominant variable when available.
    if (
        dominant
        and dominant != "wind_direction_deg"
        and (
            v_region["variable"]
            == dominant
        ).any()
    ):
        return dominant

    v = v_region.copy()

    v["abs_err"] = (
        v["predicted_value"]
        - v["observed_value"]
    ).abs()

    thr = v["variable"].map(
        lambda x: (
            state.thresholds.bust_threshold.get(
                x
            )
            or float("nan")
        )
    )

    ranked = (
        v.assign(
            _r=v["abs_err"] / thr
        )
        .groupby(
            "variable",
            observed=True,
        )["_r"]
        .mean()
    )

    if ranked.dropna().empty:
        ranked = (
            v.groupby(
                "variable",
                observed=True,
            )["abs_err"]
            .mean()
        )

    return (
        None
        if ranked.empty
        else str(ranked.idxmax())
    )


def _focus_for_region(
    sc: inference.ScoredCycle,
    state,
    region_id: str,
    dominant: Optional[str],
) -> Optional[schemas.ReplayFocusSeries]:
    pv = sc.per_variable

    if (
        pv.empty
        or "observed_value" not in pv.columns
    ):
        return None

    v = pv[
        (
            pv["region_id"].astype(str)
            == str(region_id)
        )
        & pv["observed_value"].notna()
        & pv["predicted_value"].notna()
        & (
            pv["variable"]
            != "wind_direction_deg"
        )
    ].copy()

    var = _focus_variable_for(
        v,
        dominant,
        state,
    )

    if var is None:
        return None

    sub = (
        v[v["variable"] == var]
        .sort_values(
            "lead_time_days"
        )
    )

    points = [
        schemas.ReplayFocusPoint(
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
                r.observed_value
            ),
            observed_status=(
                getattr(
                    r,
                    "verification_status",
                    None,
                )
                or "final"
            ),
            ensemble_spread=_f(
                getattr(
                    r,
                    "ensemble_spread",
                    None,
                )
            ),
        )
        for r in sub.itertuples()
    ]

    if not points:
        return None

    return schemas.ReplayFocusSeries(
        region_id=str(region_id),
        region_name=_region_name(
            str(region_id)
        ),
        variable=var,
        unit=_unit(var),
        bust_threshold=_f(
            state.thresholds.bust_threshold.get(
                var
            )
        ),
        points=points,
    )


def _build_focus(
    sc: inference.ScoredCycle,
    state,
    want_region: Optional[str],
) -> "tuple[Optional[schemas.ReplayFocusSeries], list[schemas.ReplayFocusSeries]]":
    """
    Build replay focus options.

    IMPORTANT:
    The replay screen orders events by:

        lead_time_days ASC
        bust_probability DESC

    The default focus must use the same deterministic
    ordering. This avoids inconsistent region selection
    when multiple regions have the same bust probability.
    """

    if sc.events.empty:
        return None, []

    # ---------------------------------------------------------
    # EXACT SAME ORDERING AS get_replay()
    # ---------------------------------------------------------
    ev = sc.events.sort_values(
        [
            "lead_time_days",
            "bust_probability",
        ],
        ascending=[
            True,
            False,
        ],
        kind="mergesort",
    ).copy()

    # The first row represents the same peak-risk row
    # used by the replay screen/test.
    peak_rid = str(
        ev.iloc[0]["region_id"]
    )

    # ---------------------------------------------------------
    # Determine the worst row for every region.
    # ---------------------------------------------------------
    rid_col = ev["region_id"].astype(str)

    worst_indices = (
        ev.groupby(
            rid_col,
            sort=False,
        )["bust_probability"]
        .idxmax()
    )

    worst_row = ev.loc[
        worst_indices
    ].copy()

    dom_by_region = {
        str(r.region_id): getattr(
            r,
            "dominant_variable",
            None,
        )
        for r in worst_row.itertuples()
    }

    # ---------------------------------------------------------
    # Rank focus options by each region's maximum risk.
    # ---------------------------------------------------------
    worst_row["_rid"] = (
        worst_row["region_id"]
        .astype(str)
    )

    rest = (
        worst_row
        .sort_values(
            "bust_probability",
            ascending=False,
            kind="mergesort",
        )["_rid"]
        .tolist()
    )

    # Global peak region is always attempted first.
    order = [
        peak_rid
    ] + [
        rid
        for rid in rest
        if rid != peak_rid
    ]

    options: list[
        schemas.ReplayFocusSeries
    ] = []

    for rid in order:
        fs = _focus_for_region(
            sc,
            state,
            rid,
            dom_by_region.get(rid),
        )

        if fs is not None:
            options.append(fs)

    if not options:
        return None, []

    # ---------------------------------------------------------
    # Explicit user-selected region.
    # ---------------------------------------------------------
    if want_region:
        requested_region = str(
            want_region
        )

        requested = next(
            (
                option
                for option in options
                if option.region_id
                == requested_region
            ),
            None,
        )

        if requested is not None:
            return requested, options

    # ---------------------------------------------------------
    # DEFAULT:
    # Always use the same peak region selected above.
    # ---------------------------------------------------------
    peak_focus = next(
        (
            option
            for option in options
            if option.region_id
            == peak_rid
        ),
        None,
    )

    if peak_focus is not None:
        return peak_focus, options

    # Defensive fallback.
    return options[0], options


def invalidate() -> None:
    global _cycles_memo
    _cycles_memo = None