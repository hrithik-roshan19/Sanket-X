"""Spatial/temporal alignment helpers for Sanket-X verification.

The module keeps aggregation deterministic and auditable:
- forecast and observation grids are never silently reprojected;
- grid cells are assigned to IMD subdivision polygons by cell-centre containment;
- latitude cosine weights approximate equal-area weighting on a regular lat/lon grid;
- temporal joins use an explicit init_time + lead convention.

For production, exact polygon-cell intersection can be enabled later without changing
these contracts. Cell-centre containment is intentionally the M1 baseline because it is
fast, reproducible, and easy to validate on the 0.25-degree products.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep


def validate_grid(frame: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> None:
    """Fail closed on malformed geographic grids."""
    missing = [c for c in (lat_col, lon_col) if c not in frame.columns]
    if missing:
        raise ValueError(f"Grid is missing columns: {missing}")
    lat = pd.to_numeric(frame[lat_col], errors="coerce")
    lon = pd.to_numeric(frame[lon_col], errors="coerce")
    if lat.isna().any() or lon.isna().any():
        raise ValueError("Grid contains non-numeric coordinates")
    if not lat.between(-90, 90).all():
        raise ValueError("Latitude outside [-90, 90]")
    if not lon.between(-180, 180).all():
        raise ValueError("Longitude outside [-180, 180]")


def _cell_weights(latitudes: pd.Series) -> pd.Series:
    # A regular lon/lat cell has area proportional to cos(latitude).
    w = np.cos(np.deg2rad(pd.to_numeric(latitudes, errors="coerce")))
    if (~np.isfinite(w)).any() or (w <= 0).any():
        raise ValueError("Cannot compute positive geographic cell weights")
    return pd.Series(w, index=latitudes.index, dtype="float64")


def assign_subdivisions(
    frame: pd.DataFrame,
    geometries: Iterable[tuple[str, object]],
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Assign each grid point to at most one subdivision polygon.

    ``geometries`` is an iterable of (subdivision_id, shapely geometry). Boundary
    points are accepted with ``covers`` so points exactly on a polygon edge are not
    unnecessarily discarded.
    """
    validate_grid(frame, lat_col, lon_col)
    prepared = [(sid, prep(geom)) for sid, geom in geometries if geom is not None and not geom.is_empty]
    if not prepared:
        raise ValueError("No valid subdivision geometries supplied")

    out = frame.copy()
    assignments: list[str | None] = []
    for lat, lon in zip(out[lat_col], out[lon_col]):
        point = Point(float(lon), float(lat))
        matches = [sid for sid, geom in prepared if geom.covers(point)]
        if len(matches) > 1:
            raise ValueError(f"Grid point ({lat}, {lon}) belongs to multiple subdivisions: {matches}")
        assignments.append(matches[0] if matches else None)
    out["subdivision_id"] = assignments
    return out


def aggregate_grid_to_subdivision(
    frame: pd.DataFrame,
    geometries: Iterable[tuple[str, object]],
    value_col: str,
    group_cols: Iterable[str],
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Area-weight a regular lat/lon grid into subdivision-level values."""
    if value_col not in frame.columns:
        raise ValueError(f"Missing value column: {value_col}")
    group_cols = list(group_cols)
    missing = [c for c in group_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")

    assigned = assign_subdivisions(frame, geometries, lat_col, lon_col)
    assigned[value_col] = pd.to_numeric(assigned[value_col], errors="coerce")
    assigned["_weight"] = _cell_weights(assigned[lat_col])
    assigned = assigned.dropna(subset=["subdivision_id", value_col])
    if assigned.empty:
        return pd.DataFrame(columns=group_cols + ["subdivision_id", value_col, "grid_cell_count"])

    keys = group_cols + ["subdivision_id"]
    numerator = (assigned[value_col] * assigned["_weight"]).groupby(
        [assigned[k] for k in keys], dropna=False
    ).sum()
    denominator = assigned["_weight"].groupby(
        [assigned[k] for k in keys], dropna=False
    ).sum()
    counts = assigned.groupby(keys, dropna=False).size().rename("grid_cell_count")

    result = (numerator / denominator).rename(value_col).reset_index()
    result = result.merge(counts.reset_index(), on=keys, how="left")
    return result.sort_values(keys).reset_index(drop=True)


def validate_forecast_time_alignment(
    frame: pd.DataFrame,
    init_col: str = "init_time",
    valid_col: str = "valid_time",
    lead_col: str = "lead_day",
) -> None:
    """Validate Sanket-X convention: lead 1 = initialization day."""
    required = [init_col, valid_col, lead_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing temporal columns: {missing}")
    init = pd.to_datetime(frame[init_col], errors="coerce", utc=True)
    valid = pd.to_datetime(frame[valid_col], errors="coerce", utc=True)
    lead = pd.to_numeric(frame[lead_col], errors="coerce")
    bad = init.isna() | valid.isna() | lead.isna() | (lead % 1 != 0) | ~lead.between(1, 10)
    if bad.any():
        raise ValueError(f"Invalid forecast temporal records: {int(bad.sum())}")
    observed_offset = (valid - init).dt.total_seconds() / 86400.0
    expected_offset = lead - 1
    mismatch = (observed_offset - expected_offset).abs() > 1e-9
    if mismatch.any():
        raise ValueError(f"Forecast valid_time/lead_day misalignment: {int(mismatch.sum())} rows")


def weighted_coverage(
    frame: pd.DataFrame,
    geometries: Iterable[tuple[str, object]],
    value_col: str,
    group_cols: Iterable[str],
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Return valid geographic weight coverage per group/subdivision.

    Coverage is the ratio of valid-value cell weight to all assigned cell weight.
    It is deliberately exposed separately so callers can enforce a minimum quality
    threshold before producing verification labels.
    """
    if value_col not in frame.columns:
        raise ValueError(f"Missing value column: {value_col}")
    group_cols = list(group_cols)
    assigned = assign_subdivisions(frame, geometries, lat_col, lon_col)
    assigned["_weight"] = _cell_weights(assigned[lat_col])
    assigned["_valid"] = pd.to_numeric(assigned[value_col], errors="coerce").notna()
    assigned = assigned.dropna(subset=["subdivision_id"])
    keys = group_cols + ["subdivision_id"]
    total = assigned.groupby(keys, dropna=False)["_weight"].sum().rename("total_weight")
    valid = assigned.loc[assigned["_valid"]].groupby(keys, dropna=False)["_weight"].sum().rename("valid_weight")
    result = pd.concat([total, valid], axis=1).fillna(0).reset_index()
    result["coverage_ratio"] = np.where(result["total_weight"] > 0,
                                         result["valid_weight"] / result["total_weight"], 0.0)
    result["coverage_ratio"] = result["coverage_ratio"].clip(0.0, 1.0)
    return result.sort_values(keys).reset_index(drop=True)
