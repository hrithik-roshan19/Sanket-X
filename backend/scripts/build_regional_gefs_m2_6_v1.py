"""
Sanket-X M2.6 Regional GEFS Extraction v1

Purpose
-------
Convert a small set of native 0.25-degree GEFS GRIB files into compact
IMD meteorological-subdivision forecasts. This is a benchmark/processing
stage: it never regrids, interpolates, nearest-neighbours, or fabricates data.

Input
-----
GRIB2 files for one init/cycle, members gec00/gep01..gep04 and one or more
forecast leads. Files are discovered recursively from --input-root.

Output
------
One Parquet file per init_date x lead_day containing one row per usable IMD
subdivision and the five compatible members' regional statistics. A separate
ensemble summary is included in each row.

The script is intentionally independent of the existing M2.6 downloader. It
lets us benchmark the regional reduction using the already downloaded smoke
files before changing the downloader.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
import cfgrib

MEMBERS = ["gec00", "gep01", "gep02", "gep03", "gep04"]
LEADS = [24, 48, 72, 96, 120]
LAT_MIN, LAT_MAX = 6.5, 38.5
LON_MIN, LON_MAX = 66.5, 100.0
GRID_STEP = 0.25

FILE_RE = re.compile(
    r"(?P<member>ge[cp]\d{2})\.t(?P<cycle>\d{2})z\.pgrb2s\.0p25\.f(?P<lead>\d{3})$",
    re.IGNORECASE,
)

VAR_MAP = {
    "2t": "temperature_c",
    "2r": "humidity_pct",
    "10u": "wind_u10_ms",
    "10v": "wind_v10_ms",
    "sp": "pressure_hpa",
    "tp": "rainfall_mm",
    "pwat": "atmospheric_moisture_kgm2",
}

REQUIRED = set(VAR_MAP.values())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--subdivisions", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--cycle", default="00Z")
    p.add_argument("--leads", nargs="+", type=int, default=LEADS)
    return p.parse_args()


def parse_filename(path: Path) -> tuple[str, str, int]:
    m = FILE_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected GEFS filename: {path.name}")
    member = m.group("member").lower()
    cycle = m.group("cycle")
    lead = int(m.group("lead"))
    if member not in MEMBERS:
        raise ValueError(f"Unsupported member: {member}")
    if lead not in LEADS:
        raise ValueError(f"Unsupported lead: {lead}")
    return member, cycle, lead


def find_files(root: Path, date_text: str, cycle: str, leads: list[int]) -> dict[tuple[str, int], Path]:
    cycle_num = cycle.replace("Z", "").zfill(2)
    found: dict[tuple[str, int], Path] = {}
    for path in root.rglob("*.grib2"):
        try:
            member, file_cycle, lead = parse_filename(path)
        except ValueError:
            continue
        if file_cycle != cycle_num or lead not in leads:
            continue
        # The date is validated from the directory/path to avoid assuming that
        # a GRIB's internal time is enough for provenance.
        if date_text not in str(path):
            continue
        key = (member, lead)
        if key in found:
            raise ValueError(f"Duplicate GRIB for {key}: {found[key]} and {path}")
        found[key] = path
    return found


def _normalise_dataset(ds: Any) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    if lat_name not in ds.coords or lon_name not in ds.coords:
        raise ValueError("GRIB dataset has no latitude/longitude coordinates")

    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Only 1-D regular latitude/longitude coordinates are supported")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("Non-finite latitude/longitude coordinates")

    lat = lat.copy()
    lon = lon.copy()
    # GEFS latitude is often descending. Normalize to ascending without
    # changing the underlying values.
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat_sorted = lat[lat_order]
    lon_sorted = lon[lon_order]

    if not np.isclose(np.diff(lat_sorted), GRID_STEP, atol=1e-6).all():
        raise ValueError("Latitude grid is not native 0.25-degree spacing")
    if not np.isclose(np.diff(lon_sorted), GRID_STEP, atol=1e-6).all():
        raise ValueError("Longitude grid is not native 0.25-degree spacing")

    lat_mask = (lat_sorted >= LAT_MIN - 1e-9) & (lat_sorted <= LAT_MAX + 1e-9)
    lon_mask = (lon_sorted >= LON_MIN - 1e-9) & (lon_sorted <= LON_MAX + 1e-9)
    lat_idx = lat_order[lat_mask]
    lon_idx = lon_order[lon_mask]
    crop_lat = lat_sorted[lat_mask]
    crop_lon = lon_sorted[lon_mask]
    if len(crop_lat) != 129 or len(crop_lon) != 135:
        raise ValueError(f"Unexpected native crop dimensions: {len(crop_lat)}x{len(crop_lon)}")

    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for raw, canonical in VAR_MAP.items():
        if raw not in ds.data_vars:
            raise ValueError(f"Missing required GRIB variable: {raw}")
        arr = np.asarray(ds[raw].values)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Variable {raw} is not a 2-D field")
        cropped = arr[np.ix_(lat_idx, lon_idx)]
        result[canonical] = (crop_lat, crop_lon, cropped.astype(float, copy=False))
    return result


def load_member(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    try:
        merged: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for ds in datasets:
            fields = _normalise_dataset(ds)
            for name, value in fields.items():
                if name in merged:
                    # The same canonical variable appearing in two filtered
                    # datasets must be numerically identical; otherwise fail.
                    old = merged[name][2]
                    new = value[2]
                    if old.shape != new.shape or not np.allclose(old, new, equal_nan=True):
                        raise ValueError(f"Conflicting GRIB datasets for {name}")
                else:
                    merged[name] = value
        missing = REQUIRED - set(merged)
        if missing:
            raise ValueError(f"Missing canonical variables: {sorted(missing)}")
        return merged
    finally:
        for ds in datasets:
            ds.close()


def load_grid(subdivisions_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(subdivisions_path)
    required = {"sanket_x_id", "sanket_x_subdivision", "geometry"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Subdivision geometry missing columns: {sorted(missing)}")
    if gdf.crs is None:
        raise ValueError("Subdivision geometry has no CRS")
    gdf = gdf.to_crs("EPSG:4326").copy()
    if len(gdf) != 36:
        raise ValueError(f"Expected 36 IMD subdivisions, found {len(gdf)}")
    if gdf.geometry.is_empty.any() or (~gdf.geometry.is_valid).any():
        raise ValueError("Subdivision geometry contains empty/invalid geometries")
    return gdf[["sanket_x_id", "sanket_x_subdivision", "geometry"]]


def grid_points(lat: np.ndarray, lon: np.ndarray) -> gpd.GeoDataFrame:
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    return gpd.GeoDataFrame(
        {"latitude": yy.ravel(), "longitude": xx.ravel()},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs="EPSG:4326",
    )


def assign_regions(points: gpd.GeoDataFrame, regions: gpd.GeoDataFrame) -> pd.Series:
    joined = gpd.sjoin(points, regions, how="left", predicate="within")
    if joined.index.duplicated().any():
        # A grid center on a shared boundary can be matched to multiple
        # polygons; use covers-based single assignment, otherwise fail.
        joined = gpd.sjoin(points, regions, how="left", predicate="covered_by")
    if joined.index.duplicated().any():
        raise ValueError("A GEFS grid center matched multiple IMD subdivisions")
    return joined["sanket_x_id"].reindex(points.index)


def aggregate_member(fields: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], regions: gpd.GeoDataFrame) -> pd.DataFrame:
    lat, lon, _ = fields["temperature_c"]
    points = grid_points(lat, lon)
    region_ids = assign_regions(points, regions)
    valid = region_ids.notna().to_numpy()
    if not valid.any():
        raise ValueError("No GEFS grid centers intersect IMD subdivisions")

    lat_values = points.loc[valid, "latitude"].to_numpy(float)
    weights = np.cos(np.deg2rad(lat_values))
    weights = weights / weights.sum() * len(weights)
    region_values = region_ids.loc[valid].to_numpy()
    out: list[dict[str, Any]] = []
    for region_id in sorted(pd.unique(region_values)):
        mask = region_values == region_id
        row: dict[str, Any] = {"sanket_x_id": region_id, "grid_cell_count": int(mask.sum())}
        w = weights[mask]
        w = w / w.sum()
        for name, (_, _, values) in fields.items():
            flat = values.ravel()[valid][mask]
            finite = np.isfinite(flat)
            if not finite.any():
                row[name] = np.nan
                continue
            wf = w[finite]
            wf = wf / wf.sum()
            row[name] = float(np.sum(flat[finite] * wf))
        out.append(row)
    return pd.DataFrame(out)


def build(date_text: str, cycle: str, found: dict[tuple[str, int], Path], regions: gpd.GeoDataFrame, leads: list[int], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    region_names = regions.set_index("sanket_x_id")["sanket_x_subdivision"].to_dict()
    rows: list[pd.DataFrame] = []
    files_used: list[str] = []

    for lead in leads:
        member_frames: list[pd.DataFrame] = []
        for member in MEMBERS:
            path = found.get((member, lead))
            if path is None:
                raise ValueError(f"Missing {member} lead {lead} GRIB")
            fields = load_member(path)
            frame = aggregate_member(fields, regions)
            frame["member"] = member
            member_frames.append(frame)
            files_used.append(str(path))

        merged = pd.concat(member_frames, ignore_index=True)
        merged["sanket_x_subdivision"] = merged["sanket_x_id"].map(region_names)

        ensemble = merged.groupby("sanket_x_id", as_index=False).agg(
            ensemble_member_count=("member", "nunique"),
            temperature_c_mean=("temperature_c", "mean"),
            temperature_c_spread=("temperature_c", "std"),
            humidity_pct_mean=("humidity_pct", "mean"),
            humidity_pct_spread=("humidity_pct", "std"),
            wind_u10_ms_mean=("wind_u10_ms", "mean"),
            wind_u10_ms_spread=("wind_u10_ms", "std"),
            wind_v10_ms_mean=("wind_v10_ms", "mean"),
            wind_v10_ms_spread=("wind_v10_ms", "std"),
            pressure_hpa_mean=("pressure_hpa", "mean"),
            pressure_hpa_spread=("pressure_hpa", "std"),
            rainfall_mm_mean=("rainfall_mm", "mean"),
            rainfall_mm_spread=("rainfall_mm", "std"),
            atmospheric_moisture_kgm2_mean=("atmospheric_moisture_kgm2", "mean"),
            atmospheric_moisture_kgm2_spread=("atmospheric_moisture_kgm2", "std"),
        )
        merged = merged.merge(ensemble, on="sanket_x_id", how="left", validate="many_to_one")
        merged["init_date"] = date_text
        merged["valid_date"] = (pd.Timestamp(date_text) + pd.Timedelta(hours=lead)).date().isoformat()
        merged["cycle"] = cycle.replace("Z", "").zfill(2)
        merged["lead_hour"] = lead
        merged["lead_day"] = lead // 24
        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["lead_day", "sanket_x_id", "member"]).reset_index(drop=True)
    out = output_root / f"regional_{date_text}_cycle{cycle.replace('Z','').zfill(2)}.parquet"
    result.to_parquet(out, index=False)

    expected_rows = len(leads) * 36 * len(MEMBERS)
    usable = int(result["sanket_x_id"].nunique())
    return {
        "status": "PASS" if len(result) == expected_rows and usable == 36 else "NOT_PASS",
        "date": date_text,
        "cycle": cycle,
        "leads": leads,
        "members": MEMBERS,
        "rows": int(len(result)),
        "expected_rows": expected_rows,
        "subdivisions_present": usable,
        "expected_subdivisions": 36,
        "files_used": sorted(set(files_used)),
        "output": str(out),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "no_interpolation": True,
        "no_regridding": True,
        "no_nearest_neighbour": True,
        "no_fabrication": True,
    }


def main() -> int:
    args = parse_args()
    cycle = args.cycle.upper()
    leads = sorted(set(args.leads))
    if any(lead not in LEADS for lead in leads):
        raise SystemExit(f"Leads must be from {LEADS}")
    found = find_files(args.input_root, args.date, cycle, leads)
    expected = {(m, lead) for m in MEMBERS for lead in leads}
    missing = sorted(expected - set(found))
    if missing:
        print("Missing files:")
        for item in missing:
            print(f"  {item[0]} f{item[1]:03d}")
        return 1
    regions = load_grid(args.subdivisions)
    report = build(args.date, cycle, found, regions, leads, args.output_root)
    report_path = args.output_root / "regional_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
