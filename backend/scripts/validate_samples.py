"""Validate Sanket forecast/observation sample data before ML work.

This script is deliberately fail-closed: a sample is not considered ML-ready merely
because it can be read. It checks schema, temporal alignment, ensemble membership,
lead coverage, coordinates, units/ranges, duplicates, provenance, and forecast/observation
pairability.

Usage (from backend):
    python scripts/validate_samples.py
    python scripts/validate_samples.py --gefs data/samples/gefs_reforecast_india_2019.parquet \
        --obs data/samples/era5_observations_india_2019.parquet

The runtime dependencies are the same as the project (pandas + pyarrow).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MEMBERS = {
    "c00",
    "p01",
    "p02",
    "p03",
    "p04",
}


# Surface pressure can legitimately fall below 800 hPa
# for high-altitude regions in India (e.g. Himalayan locations).
# Therefore the validation lower bound is intentionally relaxed
# while still rejecting physically unreasonable values.
SURFACE_PRESSURE_RANGE_HPA = (500, 1100)


FORECAST_VARS = {
    "t2m_c": (-50, 60),
    "rh2m_pct": (0, 100),
    "apcp_mm": (0, 2000),
    "mslp_hpa": (800, 1100),
    "psfc_hpa": SURFACE_PRESSURE_RANGE_HPA,
    "pwat_kgm2": (0, 90),
    "wspd10m_ms": (0, 100),
    "wdir10m_deg": (0, 360),
    "soilw_vol_pct": (0, 100),
}


OBS_VARS = {
    "t2m_c": (-50, 60),
    "rh2m_pct": (0, 100),
    "precip_mm": (0, 2000),
    "mslp_hpa": (800, 1100),
    "psfc_hpa": SURFACE_PRESSURE_RANGE_HPA,
    "pwat_kgm2": (0, 90),
    "wspd10m_ms": (0, 100),
    "wdir10m_deg": (0, 360),
    "soil_moisture_pct": (0, 100),
}


REQUIRED_GEFS = {
    "city",
    "state",
    "region",
    "latitude",
    "longitude",
    "init_date",
    "valid_date",
    "lead_day",
    "member",
    *FORECAST_VARS,
}


REQUIRED_OBS = {
    "city",
    "state",
    "region",
    "latitude",
    "longitude",
    "date",
    *OBS_VARS,
}


def issue(
    bucket: list[dict],
    severity: str,
    code: str,
    message: str,
    **extra,
) -> None:
    """Add a structured validation issue."""

    row = {
        "severity": severity,
        "code": code,
        "message": message,
    }

    row.update(extra)
    bucket.append(row)


def check_columns(
    df: pd.DataFrame,
    required: set[str],
    issues: list[dict],
    name: str,
) -> None:
    """Check that all required columns exist."""

    missing = sorted(required - set(df.columns))

    if missing:
        issue(
            issues,
            "ERROR",
            "MISSING_COLUMNS",
            f"{name} is missing required columns",
            columns=missing,
        )


def check_numeric_ranges(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]],
    issues: list[dict],
    name: str,
) -> None:
    """Validate numeric columns against documented physical ranges."""

    for col, (lo, hi) in ranges.items():
        if col not in df:
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        non_numeric = int(
            s.isna().sum() - df[col].isna().sum()
        )

        if non_numeric:
            issue(
                issues,
                "ERROR",
                "NON_NUMERIC",
                f"{name}.{col} contains non-numeric values",
                count=non_numeric,
            )

        bad = (
            s.notna()
            & ~s.between(
                lo,
                hi,
                inclusive="both",
            )
        )

        if bad.any():
            issue(
                issues,
                "ERROR",
                "OUT_OF_RANGE",
                f"{name}.{col} has physically implausible values",
                count=int(bad.sum()),
                lo=lo,
                hi=hi,
            )


def validate_gefs(
    df: pd.DataFrame,
    issues: list[dict],
) -> dict:
    """Validate GEFS forecast data."""

    check_columns(
        df,
        REQUIRED_GEFS,
        issues,
        "GEFS",
    )

    if not REQUIRED_GEFS.issubset(df.columns):
        return {
            "rows": len(df),
        }

    init = pd.to_datetime(
        df["init_date"],
        errors="coerce",
    )

    valid = pd.to_datetime(
        df["valid_date"],
        errors="coerce",
    )

    bad_dates = init.isna() | valid.isna()

    if bad_dates.any():
        issue(
            issues,
            "ERROR",
            "BAD_DATES",
            "GEFS has invalid init_date/valid_date",
            count=int(bad_dates.sum()),
        )

    lead = pd.to_numeric(
        df["lead_day"],
        errors="coerce",
    )

    bad_lead = (
        lead.isna()
        | ~lead.between(
            1,
            10,
            inclusive="both",
        )
        | ((lead % 1).fillna(0) != 0)
    )

    if bad_lead.any():
        issue(
            issues,
            "ERROR",
            "BAD_LEAD",
            "GEFS lead_day must be an integer from 1 to 10",
            count=int(bad_lead.sum()),
        )

    offset = (
        valid - init
    ).dt.total_seconds() / 86400.0

    align = (
        offset.notna()
        & lead.notna()
        & (
            (offset - (lead - 1)).abs()
            > 1e-9
        )
    )

    if align.any():
        issue(
            issues,
            "ERROR",
            "TIME_ALIGNMENT",
            "valid_date != init_date + (lead_day - 1)",
            count=int(align.sum()),
        )

    members = set(
        df["member"]
        .dropna()
        .astype(str)
        .unique()
    )

    unexpected = sorted(
        members - EXPECTED_MEMBERS
    )

    if unexpected:
        issue(
            issues,
            "ERROR",
            "UNEXPECTED_MEMBERS",
            "Unexpected historical GEFS members found",
            members=unexpected,
        )

    missing = sorted(
        EXPECTED_MEMBERS - members
    )

    if missing:
        issue(
            issues,
            "ERROR",
            "MISSING_MEMBERS",
            "Expected historical GEFS members are absent",
            members=missing,
        )

    coordinates = df[
        ["latitude", "longitude"]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not coordinates.notna().all().all():
        issue(
            issues,
            "ERROR",
            "BAD_COORDINATES",
            "GEFS contains non-numeric coordinates",
        )

    latitude = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    longitude = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    if (
        (latitude < -90)
        | (latitude > 90)
    ).any():
        issue(
            issues,
            "ERROR",
            "LAT_RANGE",
            "GEFS latitude outside [-90, 90]",
        )

    if (
        (longitude < -180)
        | (longitude > 180)
    ).any():
        issue(
            issues,
            "ERROR",
            "LON_RANGE",
            "GEFS longitude outside [-180, 180]",
        )

    check_numeric_ranges(
        df,
        FORECAST_VARS,
        issues,
        "GEFS",
    )

    key = [
        "city",
        "init_date",
        "member",
        "valid_date",
    ]

    dup = df.duplicated(
        key,
        keep=False,
    )

    if dup.any():
        issue(
            issues,
            "ERROR",
            "DUPLICATES",
            "Duplicate GEFS forecast keys found",
            rows=int(dup.sum()),
            key=key,
        )

    provenance = [
        c
        for c in df.columns
        if c.startswith("src_")
    ]

    if not provenance:
        issue(
            issues,
            "ERROR",
            "NO_PROVENANCE",
            "GEFS sample has no source-message provenance columns",
        )
    else:
        empty = (
            df[provenance].isna().all(axis=1)
            | df[provenance]
            .astype(str)
            .eq("")
            .all(axis=1)
        )

        if empty.any():
            issue(
                issues,
                "WARNING",
                "MISSING_PROVENANCE",
                "Some GEFS rows have no source provenance",
                rows=int(empty.sum()),
            )

    coverage = (
        df.groupby(
            ["member", "lead_day"],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values(
            ["member", "lead_day"]
        )
    )

    return {
        "rows": int(len(df)),
        "cities": int(df["city"].nunique()),
        "members": sorted(members),
        "init_dates": int(
            df["init_date"].nunique()
        ),
        "lead_days": sorted(
            pd.to_numeric(
                df["lead_day"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "coverage_rows": coverage.to_dict(
            orient="records"
        ),
    }


def validate_obs(
    df: pd.DataFrame,
    issues: list[dict],
) -> dict:
    """Validate ERA5 observation data."""

    check_columns(
        df,
        REQUIRED_OBS,
        issues,
        "ERA5",
    )

    if not REQUIRED_OBS.issubset(df.columns):
        return {
            "rows": len(df),
        }

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if dates.isna().any():
        issue(
            issues,
            "ERROR",
            "BAD_DATES",
            "ERA5 contains invalid dates",
            count=int(dates.isna().sum()),
        )

    lat = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    lon = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    if (
        lat.isna().any()
        or (~lat.between(-90, 90)).any()
    ):
        issue(
            issues,
            "ERROR",
            "LAT_RANGE",
            "ERA5 latitude invalid",
        )

    if (
        lon.isna().any()
        or (~lon.between(-180, 180)).any()
    ):
        issue(
            issues,
            "ERROR",
            "LON_RANGE",
            "ERA5 longitude invalid",
        )

    check_numeric_ranges(
        df,
        OBS_VARS,
        issues,
        "ERA5",
    )

    key = [
        "city",
        "date",
    ]

    dup = df.duplicated(
        key,
        keep=False,
    )

    if dup.any():
        issue(
            issues,
            "ERROR",
            "DUPLICATES",
            "Duplicate ERA5 observation keys found",
            rows=int(dup.sum()),
            key=key,
        )

    return {
        "rows": int(len(df)),
        "cities": int(df["city"].nunique()),
        "date_min": (
            str(dates.min().date())
            if dates.notna().any()
            else None
        ),
        "date_max": (
            str(dates.max().date())
            if dates.notna().any()
            else None
        ),
    }


def validate_pairability(
    gefs: pd.DataFrame,
    obs: pd.DataFrame,
    issues: list[dict],
) -> dict:
    """Check forecast/observation date pairability."""

    if not {
        "city",
        "valid_date",
        "t2m_c",
    }.issubset(gefs.columns) or not {
        "city",
        "date",
        "t2m_c",
    }.issubset(obs.columns):
        return {}

    f = gefs[
        [
            "city",
            "valid_date",
            "t2m_c",
        ]
    ].copy()

    o = obs[
        [
            "city",
            "date",
            "t2m_c",
        ]
    ].copy()

    f["valid_date"] = (
        pd.to_datetime(
            f["valid_date"],
            errors="coerce",
        )
        .dt.date
    )

    o["date"] = (
        pd.to_datetime(
            o["date"],
            errors="coerce",
        )
        .dt.date
    )

    # Collapse members only for pairability statistics;
    # never use this table for labels.
    f = f.drop_duplicates(
        [
            "city",
            "valid_date",
        ]
    )

    j = f.merge(
        o,
        left_on=[
            "city",
            "valid_date",
        ],
        right_on=[
            "city",
            "date",
        ],
        how="inner",
    )

    rate = len(j) / max(
        len(f),
        1,
    )

    if rate < 0.80:
        issue(
            issues,
            "ERROR",
            "LOW_PAIR_RATE",
            "Forecast/observation pairing rate is below 80%",
            rate=rate,
        )

    return {
        "forecast_keys": int(len(f)),
        "paired_keys": int(len(j)),
        "pair_rate": float(rate),
    }


def main() -> int:
    """Run complete sample validation."""

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--gefs",
        default=(
            "data/samples/"
            "gefs_reforecast_india_2019.parquet"
        ),
    )

    ap.add_argument(
        "--obs",
        default=(
            "data/samples/"
            "era5_observations_india_2019.parquet"
        ),
    )

    ap.add_argument(
        "--out",
        default=(
            "data/analysis/"
            "data_validation_report.json"
        ),
    )

    args = ap.parse_args()

    issues: list[dict] = []

    gefs_path = Path(args.gefs)
    obs_path = Path(args.obs)

    if not gefs_path.exists():
        issue(
            issues,
            "ERROR",
            "FILE_MISSING",
            "GEFS sample file does not exist",
            path=str(gefs_path),
        )

    if not obs_path.exists():
        issue(
            issues,
            "ERROR",
            "FILE_MISSING",
            "ERA5 sample file does not exist",
            path=str(obs_path),
        )

    if issues:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "issues": issues,
                },
                indent=2,
            )
        )
        return 2

    gefs = pd.read_parquet(
        gefs_path
    )

    obs = pd.read_parquet(
        obs_path
    )

    summary = {
        "gefs": validate_gefs(
            gefs,
            issues,
        ),
        "era5": validate_obs(
            obs,
            issues,
        ),
        "pairability": validate_pairability(
            gefs,
            obs,
            issues,
        ),
    }

    errors = [
        x
        for x in issues
        if x["severity"] == "ERROR"
    ]

    report = {
        "status": (
            "FAIL"
            if errors
            else "PASS_WITH_WARNINGS"
            if issues
            else "PASS"
        ),
        "summary": summary,
        "issues": issues,
    }

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())