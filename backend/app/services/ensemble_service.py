from __future__ import annotations

import logging
import threading
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from app.api import schemas
from app.ml import inference
from app.storage import parquet_store

log = logging.getLogger(__name__)

NOT_TRAINED_MSG = "No model has been trained yet, so no cycle can be scored."
NO_SCORE_MSG = "No forecast cycle in the store could be scored by the current model."

# 0 and 360 degrees are the same bearing, so member traces around north look like a collapse.
_UNCHARTABLE = {"wind_direction_deg"}

_PRIOR_MAX_GAP = timedelta(days=7)

_lock = threading.Lock()
_memo: dict = {}
_MEMO_MAX = 8


def invalidate() -> None:
    with _lock:
        _memo.clear()


def get_divergence(
    init_date: Optional[date] = None,
    region_id: Optional[str] = None,
) -> schemas.EnsembleDivergenceResponse:
    state = inference.load_model_state()
    if state is None:
        return schemas.EnsembleDivergenceResponse(model_trained=False, message=NOT_TRAINED_MSG)

    key = (state.run_id, str(init_date), str(region_id), parquet_store.store_fingerprint())
    with _lock:
        hit = _memo.get(key)
    if hit is not None:
        return hit

    scored = inference.score_cycle(state, init_date=init_date)
    if scored is None or scored.events.empty:
        return schemas.EnsembleDivergenceResponse(
            model_trained=True, current_run_id=state.run_id, message=NO_SCORE_MSG,
        )

    result = _build(state, scored, region_id)
    with _lock:
        if len(_memo) >= _MEMO_MAX:
            _memo.pop(next(iter(_memo)))
        _memo[key] = result
    return result


def _build(state, scored, want_region: Optional[str]) -> schemas.EnsembleDivergenceResponse:
    ev = scored.events.copy()
    ev["region_id"] = ev["region_id"].astype(str)
    init = pd.Timestamp(scored.init_date).normalize()

    region_id, variable, pick = _pick_subject(ev, scored.per_variable, want_region)
    base = schemas.EnsembleDivergenceResponse(
        model_trained=True,
        current_run_id=state.run_id,
        init_date=init.date(),
        n_scored_regions=int(ev["region_id"].nunique()),
    )
    if region_id is None or variable is None:
        base.message = "This cycle has no chartable variable to show a divergence for."
        return base

    region_rows = ev[ev["region_id"] == region_id].sort_values("lead_time_days")
    members, mean_pts, observed = _traces(scored, init, region_id, variable)

    high_cut = float(state.thresholds.risk_band_cuts["high"])
    crossed = region_rows[region_rows["bust_probability"] >= high_cut]
    crossover = int(crossed["lead_time_days"].iloc[0]) if not crossed.empty else None

    high_rows = ev[ev["bust_probability"] >= high_cut]
    n_high = int(high_rows["region_id"].nunique())
    high_by_lead = int(high_rows["lead_time_days"].min()) if not high_rows.empty else None

    mean_p = float(ev["bust_probability"].mean())
    prior_p, prior_init, prior_note = _prior_cycle_mean(state, init)

    base.region_id = region_id
    base.region_name = _region_name(region_id)
    base.variable = variable
    base.unit = _unit(variable)
    base.members = members
    base.ensemble_mean = mean_pts
    base.observed = observed
    base.crossover_lead = crossover
    base.peak_bust_probability = float(region_rows["bust_probability"].max())
    base.mean_bust_probability = mean_p
    base.prior_mean_bust_probability = prior_p
    base.prior_init_date = prior_init
    base.prior_note = prior_note
    base.n_high_regions = n_high
    base.high_by_lead = high_by_lead
    base.spread_growth = float(pick["gain"]) if pick else None
    base.subject_reason = _subject_reason(pick, base.unit, pinned=bool(want_region))
    base.member_count = len(members)
    base.source = _provenance(observed)
    base.national = _national(ev)
    base.skill = _skill(state)
    base.national_note = _national_note(base)
    base.eyebrow = (
        _eyebrow(init, base.region_name or region_id, variable) if want_region
        else _national_eyebrow(init, base)
    )
    base.headline_note = _note(base, mean_pts, observed)
    return base


def _national(ev: pd.DataFrame) -> list:
    if ev.empty:
        return []
    out = []
    for lead, grp in ev.groupby("lead_time_days", observed=True):
        p = grp["bust_probability"].astype(float)
        valid = grp["valid_date"].dropna()
        out.append(schemas.NationalRiskPoint(
            lead_time_days=int(lead),
            valid_date=_d(valid.iloc[0]) if not valid.empty else None,
            mean_bust_probability=float(p.mean()),
            min_bust_probability=float(p.min()),
            max_bust_probability=float(p.max()),
            n_regions=int(grp["region_id"].nunique()),
            n_high_regions=int((grp["risk_band"] == "high").sum()),
        ))
    return sorted(out, key=lambda x: x.lead_time_days)


def _skill(state) -> schemas.ModelSkill:
    clf = ((state.metrics or {}).get("classifier") or {})
    for split in ("test", "val", "train"):
        m = clf.get(split)
        if isinstance(m, dict) and m.get("n"):
            skill = schemas.ModelSkill(
                split=split,
                n=int(m.get("n", 0)),
                roc_auc=_f(m.get("roc_auc")),
                precision=_f(m.get("precision")),
                recall=_f(m.get("recall")),
                f1=_f(m.get("f1")),
                brier=_f(m.get("brier")),
                bust_rate=_f(m.get("bust_rate")),
                calibration=[schemas.CalibrationBin(**b) for b in (m.get("calibration") or [])],
            )
            if split != "test":
                skill.note = f"No held-out test split in this run - showing {split}."
            elif not skill.calibration:
                skill.note = (
                    "Reliability bins are recorded from the next retrain onward; this "
                    "model predates them."
                )
            return skill
    return schemas.ModelSkill(note="This run recorded no classifier metrics.")


def _national_eyebrow(init: pd.Timestamp, r: schemas.EnsembleDivergenceResponse) -> str:
    return f"all india · {r.n_scored_regions} regions · cycle {init.date()}".upper()


def _national_note(r: schemas.EnsembleDivergenceResponse) -> Optional[str]:
    if not r.national:
        return None
    first, last = r.national[0], r.national[-1]
    rise = last.mean_bust_probability - first.mean_bust_probability
    direction = "climbs" if rise >= 0 else "falls"
    return (
        f"Mean bust probability {direction} from {first.mean_bust_probability:.2f} at Day "
        f"{first.lead_time_days} to {last.mean_bust_probability:.2f} at Day "
        f"{last.lead_time_days}, across all {last.n_regions} scored regions."
    )


def _pick_subject(ev: pd.DataFrame, per_variable: pd.DataFrame, want_region: Optional[str]):
    if ev.empty or per_variable.empty:
        return None, None, None

    pv = per_variable.copy()
    pv["region_id"] = pv["region_id"].astype(str)
    pv = pv[~pv["variable"].isin(_UNCHARTABLE)]
    pv = pv[pv["ensemble_spread"].notna() & pv["bust_threshold"].notna()]
    if pv.empty:
        return None, None, None

    peak_by_region = ev.groupby("region_id")["bust_probability"].max()

    rows = []
    for (rid, var), grp in pv.groupby(["region_id", "variable"], observed=True):
        grp = grp.sort_values("lead_time_days")
        if len(grp) < 3:
            continue
        spread = grp["ensemble_spread"].to_numpy(dtype=float)
        threshold = float(grp["bust_threshold"].iloc[0])
        if not (threshold > 0):
            continue
        window = max(1, len(spread) // 3)
        early, late = float(spread[:window].mean()), float(spread[-window:].mean())
        if late < 0.15 * threshold:
            continue
        rows.append({
            "region_id": rid,
            "variable": var,
            "gain": (late - early) / threshold,
            "late": late,
            "early": early,
            "threshold": threshold,
            "worst_lead": int(grp["lead_time_days"].iloc[-1]),
            "peak_p": float(peak_by_region.get(rid, 0.0)),
        })
    if not rows:
        return None, None, None

    ranked = pd.DataFrame(rows).sort_values(["gain", "peak_p"], ascending=False)
    if want_region:
        preferred = ranked[ranked["region_id"] == str(want_region)]
        if not preferred.empty:
            ranked = pd.concat([preferred, ranked[ranked["region_id"] != str(want_region)]])

    top = ranked.iloc[0].to_dict()
    return str(top["region_id"]), str(top["variable"]), top


def _traces(scored, init: pd.Timestamp, region_id: str, variable: str):
    pv = scored.per_variable
    pv = pv[(pv["region_id"].astype(str) == region_id) & (pv["variable"] == variable)]
    pv = pv.sort_values("lead_time_days")

    mean_pts = [
        schemas.EnsemblePoint(
            lead_time_days=int(r.lead_time_days),
            valid_date=_d(r.valid_date),
            value=_f(r.predicted_value),
        )
        for r in pv.itertuples()
    ]
    observed = [
        schemas.EnsemblePoint(
            lead_time_days=int(r.lead_time_days),
            valid_date=_d(r.valid_date),
            value=_f(r.observed_value),
        )
        for r in pv.itertuples()
        if pd.notna(getattr(r, "observed_value", None))
    ]

    raw = parquet_store.read_dataset(
        variables=[variable], value_types=["forecast"], init_dates=[init.date()],
        columns=["region_id", "ensemble_member_id", "lead_time_days", "valid_date", "value"],
    )
    members: list = []
    if not raw.empty:
        raw = raw[raw["region_id"].astype(str) == region_id]
        raw = raw[raw["ensemble_member_id"].notna()]
        for member_id, grp in raw.groupby(raw["ensemble_member_id"].astype(str)):
            grp = grp.sort_values("lead_time_days")
            members.append(schemas.EnsembleMemberTrace(
                member_id=member_id,
                is_control=member_id.lower().endswith("c00"),
                points=[
                    schemas.EnsemblePoint(
                        lead_time_days=int(r.lead_time_days),
                        valid_date=_d(r.valid_date),
                        value=_f(r.value),
                    )
                    for r in grp.itertuples()
                ],
            ))
        members.sort(key=lambda m: (not m.is_control, m.member_id))
    return members, mean_pts, observed


def _prior_cycle_mean(state, init: pd.Timestamp):
    try:
        cycles = [pd.Timestamp(c).normalize() for c in inference.available_cycles()]
    except Exception:  # noqa: BLE001
        log.exception("could not list cycles for the prior-cycle comparison")
        return None, None, "prior cycle unavailable"

    earlier = sorted(c for c in cycles if c < init)
    if not earlier:
        return None, None, "no earlier cycle in the store"

    prev = earlier[-1]
    if (init - prev) > _PRIOR_MAX_GAP:
        return None, None, (
            f"nearest earlier cycle is {prev.date()}, too far back to compare"
        )
    prior = inference.score_cycle(state, init_date=prev.date())
    if prior is None or prior.events.empty:
        return None, None, f"cycle {prev.date()} could not be scored"
    return float(prior.events["bust_probability"].mean()), prev.date(), None


def _provenance(observed) -> str:
    verified = "verified against ERA5 reanalysis" if observed else "not yet verified"
    return f"NOAA GEFS 0.25 deg ensemble - {verified}"


def _subject_reason(pick: Optional[dict], unit: Optional[str], pinned: bool) -> Optional[str]:
    if not pick:
        return None
    u = f" {unit}" if unit else ""
    scope = "for this region" if pinned else "in this cycle"
    return (
        f"Widest ensemble disagreement {scope}: members spread "
        f"{pick['late']:.1f}{u} apart by Day {pick['worst_lead']}, against the "
        f"{pick['threshold']:.1f}{u} error that counts as a bust."
    )


def _eyebrow(init: pd.Timestamp, region_name: str, variable: str) -> str:
    season = {12: "winter", 1: "winter", 2: "winter", 3: "pre-monsoon", 4: "pre-monsoon",
              5: "pre-monsoon", 6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
              10: "post-monsoon", 11: "post-monsoon"}[int(init.month)]
    return f"{season} · {region_name} · {_label(variable)}".upper()


def _note(r: schemas.EnsembleDivergenceResponse, mean_pts, observed) -> str:
    bits = []
    if r.n_high_regions and r.high_by_lead:
        bits.append(
            f"{r.n_high_regions} of {r.n_scored_regions} regions reach the top risk band "
            f"by Day {r.high_by_lead}."
        )
    elif r.n_scored_regions:
        bits.append(f"No region reaches the top risk band across {r.n_scored_regions} scored.")

    region = r.region_name or r.region_id
    if r.crossover_lead:
        bits.append(
            f"{region} crosses at Day {r.crossover_lead} and peaks at "
            f"{r.peak_bust_probability:.2f}, driven by {_label(r.variable or '')}."
        )
    elif r.peak_bust_probability is not None:
        bits.append(
            f"{region} tops the cycle at {r.peak_bust_probability:.2f}, driven by "
            f"{_label(r.variable or '')}."
        )

    by_lead = {p.lead_time_days: p.value for p in mean_pts if p.value is not None}
    gaps = [
        (o.lead_time_days, by_lead[o.lead_time_days] - o.value)
        for o in observed
        if o.value is not None and o.lead_time_days in by_lead
    ]
    if gaps:
        worst_lead, gap = max(gaps, key=lambda g: abs(g[1]))
        direction = "under" if gap < 0 else "over"
        bits.append(
            f"Against what verified, the ensemble {direction}-forecasts by "
            f"{abs(gap):.1f}{(' ' + r.unit) if r.unit else ''} at Day {worst_lead}."
        )
    else:
        bits.append("This cycle has not verified yet, so no observed trace is drawn.")
    return " ".join(bits)


def _label(variable: str) -> str:
    return variable.rsplit("_", 1)[0].replace("_", " ") if variable else ""


def _region_name(region_id: str) -> Optional[str]:
    from app.services.region_service import _region_name as rn
    return rn(region_id)


def _unit(variable: str) -> Optional[str]:
    from app.services.region_service import _unit as u
    return u(variable)


def _f(v) -> Optional[float]:
    return None if v is None or pd.isna(v) else float(v)


def _d(v):
    return pd.to_datetime(v).date() if pd.notna(v) else None
