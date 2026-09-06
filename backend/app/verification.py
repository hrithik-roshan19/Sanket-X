"""Fail-closed forecast verification and bust-label primitives.

The verification layer is deliberately independent of the ML stack. It converts canonical
forecast/observation rows into audited pairs, computes event/member errors, and creates
labels using thresholds fitted on the training period only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

MAX_LEAD_DAYS = 10
REQUIRED_FORECAST = {"region_id", "variable", "init_date", "valid_date", "lead_time_days", "value"}
REQUIRED_OBS = {"region_id", "variable", "valid_date", "value"}


@dataclass(frozen=True)
class VerificationReport:
    forecast_rows: int
    observation_rows: int
    paired_rows: int
    unmatched_forecasts: int
    duplicate_observations: int
    invalid_lead_rows: int
    misaligned_rows: int
    provisional_observations: int
    status: str
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _missing(df: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(df.columns))


def validate_temporal_contract(forecast: pd.DataFrame, *, max_lead_days: int = MAX_LEAD_DAYS) -> pd.DataFrame:
    """Return rows violating the canonical lead contract: valid = init + lead - 1 days."""
    if forecast.empty:
        return forecast.iloc[0:0].copy()
    f = forecast.copy()
    init = pd.to_datetime(f["init_date"], errors="coerce")
    valid = pd.to_datetime(f["valid_date"], errors="coerce")
    lead = pd.to_numeric(f["lead_time_days"], errors="coerce")
    bad = init.isna() | valid.isna() | lead.isna() | (lead < 1) | (lead > max_lead_days)
    expected = init + pd.to_timedelta(lead - 1, unit="D")
    bad |= valid.dt.normalize() != expected.dt.normalize()
    return f.loc[bad].copy()


def pair_forecasts_observations(
    canonical: pd.DataFrame,
    *,
    exclude_provisional: bool = True,
    require_aligned_lead: bool = True,
) -> tuple[pd.DataFrame, VerificationReport]:
    """Pair forecast rows to observations at the exact region/date/variable grain.

    Observations are collapsed only when they are exact duplicates at the canonical grain;
    conflicting values are rejected rather than silently averaged.
    """
    f = canonical[canonical["value_type" == "forecast"]].copy() if False else canonical[canonical["value_type"] == "forecast"].copy()
    o = canonical[canonical["value_type"] == "observed"].copy()
    errors: list[str] = []
    mf = _missing(f, REQUIRED_FORECAST)
    mo = _missing(o, REQUIRED_OBS)
    if mf:
        errors.append(f"forecast missing columns: {mf}")
    if mo:
        errors.append(f"observation missing columns: {mo}")
    if errors:
        return pd.DataFrame(), VerificationReport(len(f), len(o), 0, len(f), 0, 0, 0, 0, "invalid", tuple(errors))

    for df in (f, o):
        df["valid_date"] = pd.to_datetime(df["valid_date"], errors="coerce").dt.normalize()
        if "init_date" in df.columns:
            df["init_date"] = pd.to_datetime(df["init_date"], errors="coerce").dt.normalize()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    bad = validate_temporal_contract(f) if require_aligned_lead else f.iloc[0:0]
    if not bad.empty:
        errors.append(f"{len(bad)} forecast rows violate init/valid/lead contract")
        f = f.drop(index=bad.index)

    provisional = 0
    if exclude_provisional and "verification_status" in o.columns:
        provisional = int((o["verification_status"].astype(str).str.lower() == "provisional").sum())
        o = o[o["verification_status"].astype(str).str.lower() != "provisional"].copy()

    key = ["region_id", "valid_date", "variable"]
    duplicate_count = int(o.duplicated(key, keep=False).sum())
    if duplicate_count:
        nunique = o.groupby(key, dropna=False)["value"].nunique(dropna=True)
        conflicts = nunique[nunique > 1]
        if len(conflicts):
            errors.append(f"{len(conflicts)} observation keys have conflicting values")
            bad_keys = conflicts.index
            bad_idx = pd.MultiIndex.from_frame(o[key]).isin(pd.MultiIndex.from_tuples(bad_keys, names=key))
            o = o.loc[~bad_idx].copy()
        o = o.drop_duplicates(key, keep="first")

    join_keys = ["region_id", "valid_date", "variable"]
    paired = f.merge(o[join_keys + ["value"]].rename(columns={"value": "observed_value"}),
                     on=join_keys, how="inner", validate="many_to_one")
    paired = paired.rename(columns={"value": "forecast_value"})
    paired["abs_error"] = (paired["forecast_value"] - paired["observed_value"]).abs()
    paired = paired.dropna(subset=["forecast_value", "observed_value", "abs_error"])

    matched_ids = pd.MultiIndex.from_frame(paired[["region_id", "valid_date", "variable"]].drop_duplicates())
    f_ids = pd.MultiIndex.from_frame(f[["region_id", "valid_date", "variable"]].drop_duplicates())
    unmatched = int((~f_ids.isin(matched_ids)).sum())
    status = "valid" if not errors else "invalid"
    report = VerificationReport(
        forecast_rows=len(canonical[canonical["value_type"] == "forecast"]),
        observation_rows=len(canonical[canonical["value_type"] == "observed"]),
        paired_rows=len(paired), unmatched_forecasts=unmatched,
        duplicate_observations=duplicate_count, invalid_lead_rows=int((pd.to_numeric(f.get("lead_time_days"), errors="coerce").isna() | (pd.to_numeric(f.get("lead_time_days"), errors="coerce") < 1) | (pd.to_numeric(f.get("lead_time_days"), errors="coerce") > MAX_LEAD_DAYS)).sum()) if not f.empty else 0,
        misaligned_rows=len(bad), provisional_observations=provisional,
        status=status, errors=tuple(errors),
    )
    return paired.reset_index(drop=True), report


def fit_bust_thresholds(paired_train: pd.DataFrame, percentile: float = 90.0) -> dict[str, float]:
    """Fit variable-specific thresholds on train data only."""
    if not 50 <= percentile < 100:
        raise ValueError("percentile must be in [50, 100)")
    out: dict[str, float] = {}
    for var, g in paired_train.groupby("variable"):
        x = pd.to_numeric(g["abs_error"], errors="coerce").dropna().to_numpy(float)
        if len(x):
            out[str(var)] = float(np.percentile(x, percentile))
    return out


def label_busts(paired: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Apply frozen train thresholds. No test/validation data are used to derive labels."""
    out = paired.copy()
    out["bust_threshold"] = out["variable"].map(thresholds)
    out["bust_ratio"] = out["abs_error"] / out["bust_threshold"]
    out["y_bust"] = (out["bust_ratio"] >= 1.0).astype("Int64")
    out.loc[out["bust_threshold"].isna() | (out["bust_threshold"] <= 0), ["bust_ratio", "y_bust"]] = pd.NA
    return out


def event_level_labels(paired: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Aggregate members to one forecast event before labelling the event as a bust."""
    keys = ["region_id", "init_date", "valid_date", "lead_time_days", "variable"]
    g = (paired.groupby(keys, observed=True, dropna=False)
         .agg(fc_mean=("forecast_value", "mean"), obs=("observed_value", "mean"),
              member_count=("forecast_value", "size"), spread=("forecast_value", "std"))
         .reset_index())
    g["actual_error"] = (g["fc_mean"] - g["obs"]).abs()
    g["bust_threshold"] = g["variable"].map(thresholds)
    g["bust_ratio"] = g["actual_error"] / g["bust_threshold"]
    g["y_bust"] = (g["bust_ratio"] >= 1.0).astype("Int64")
    g.loc[g["bust_threshold"].isna() | (g["bust_threshold"] <= 0), "y_bust"] = pd.NA
    return g
