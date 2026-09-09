"""
Sanket-X M2.7 — Canonical GEFS Dataset Builder & Validator

Purpose
-------
Build a memory-safe canonical dataset catalog from the M2.6 native-grid Parquet
partitions. This stage DOES NOT concatenate hundreds of millions of grid rows
into one giant DataFrame.

M2.6 already writes one atomic Parquet partition per:
    init_date × member × lead_day

M2.7 therefore:
1. discovers completed forecast.parquet partitions;
2. validates every partition's schema, keys, coordinates, row counts and
   physical ranges;
3. checks cross-partition consistency;
4. creates a compact partition catalog;
5. creates a dataset manifest with coverage and integrity statistics.

Default contract:
- GEFSv12 operational window: 2020-09-23 through 2025-12-31
- cycle: 00Z
- members: gec00, gep01, gep02, gep03, gep04
- leads: 24, 48, 72, 96, 120 h
- India native grid crop: 6.5..38.5 N, 66.5..100 E
- no interpolation/regridding/nearest-neighbour/fabrication

Important
---------
A partial historical download is NOT treated as a complete training dataset.
The manifest explicitly records missing expected partitions. Use
--require-complete only when the full target window has been downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

INPUT_ROOT = BACKEND_DIR / "data" / "canonical_gefs_historical"
OUTPUT_ROOT = BACKEND_DIR / "data" / "canonical_gefs_m2_7"

MANIFEST_PATH = OUTPUT_ROOT / "dataset_manifest.json"
CATALOG_PATH = OUTPUT_ROOT / "partition_catalog.parquet"
REPORT_PATH = BACKEND_DIR / "data" / "analysis" / "gefs_m2_7_canonical_dataset.json"

START_DATE = date(2020, 9, 23)
END_DATE = date(2025, 12, 31)
CYCLE = "00"

MEMBERS = ["gec00", "gep01", "gep02", "gep03", "gep04"]
LEADS = [24, 48, 72, 96, 120]

INDIA_LAT_MIN = 6.5
INDIA_LAT_MAX = 38.5
INDIA_LON_MIN = 66.5
INDIA_LON_MAX = 100.0

REQUIRED_COLUMNS = {
    "init_date",
    "valid_date",
    "cycle",
    "member",
    "lead_hour",
    "lead_day",
    "latitude",
    "longitude",
    "temperature_c",
    "humidity_pct",
    "wind_u10_ms",
    "wind_v10_ms",
    "pressure_hpa",
    "rainfall_mm",
    "atmospheric_moisture_kgm2",
}

NUMERIC_COLUMNS = {
    "latitude",
    "longitude",
    "temperature_c",
    "humidity_pct",
    "wind_u10_ms",
    "wind_v10_ms",
    "pressure_hpa",
    "rainfall_mm",
    "atmospheric_moisture_kgm2",
}

# These are deliberately broad physical acceptance ranges. They validate
# corruption/unit mistakes without clipping or changing source values.
RANGES = {
    "latitude": (INDIA_LAT_MIN, INDIA_LAT_MAX),
    "longitude": (INDIA_LON_MIN, INDIA_LON_MAX),
    "temperature_c": (-90.0, 65.0),
    "humidity_pct": (0.0, 120.0),
    "wind_u10_ms": (-150.0, 150.0),
    "wind_v10_ms": (-150.0, 150.0),
    "pressure_hpa": (250.0, 1100.0),
    "rainfall_mm": (0.0, 2000.0),
    "atmospheric_moisture_kgm2": (0.0, 150.0),
}

PARTITION_RE = re.compile(
    r"init_date=(?P<init>\d{4}-\d{2}-\d{2})"
    r".*member=(?P<member>[^\\/]+)"
    r".*lead_day=(?P<lead>\d+)"
    r".*forecast\.parquet$"
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".part",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".parquet.part",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_partition(path: Path) -> tuple[date, str, int]:
    match = PARTITION_RE.search(str(path))
    if not match:
        raise ValueError(f"Unexpected partition path: {path}")

    init_date = date.fromisoformat(match.group("init"))
    member = match.group("member")
    lead_day = int(match.group("lead"))

    if member not in MEMBERS:
        raise ValueError(f"Unexpected member in partition path: {member}")
    if lead_day not in {lead // 24 for lead in LEADS}:
        raise ValueError(f"Unexpected lead_day in partition path: {lead_day}")

    return init_date, member, lead_day


def expected_partition_key(init_date: date, member: str, lead_day: int) -> str:
    return f"{init_date.isoformat()}|{member}|{lead_day}"


def expected_keys() -> list[tuple[date, str, int]]:
    days = []
    current = START_DATE
    while current <= END_DATE:
        days.append(current)
        current += timedelta(days=1)

    return [
        (day, member, lead // 24)
        for day in days
        for member in MEMBERS
        for lead in LEADS
    ]


def finite_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(np.isfinite(pd.to_numeric(series, errors="coerce")).mean())


def validate_partition(path: Path) -> dict[str, Any]:
    init_date, member, lead_day = parse_partition(path)

    if not path.is_file():
        raise ValueError(f"Partition is not a file: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError("Partition is empty.")

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate DataFrame column names detected.")

    # Partition path and in-file identity must agree.
    init_values = pd.to_datetime(df["init_date"], errors="coerce").dt.date
    valid_values = pd.to_datetime(df["valid_date"], errors="coerce").dt.date

    if init_values.isna().any() or valid_values.isna().any():
        raise ValueError("Invalid init_date or valid_date values.")

    if set(init_values) != {init_date}:
        raise ValueError("init_date does not match partition path.")

    expected_valid = init_date + timedelta(days=lead_day)
    if set(valid_values) != {expected_valid}:
        raise ValueError(
            f"valid_date mismatch: expected {expected_valid.isoformat()}"
        )

    if set(df["cycle"].astype(str)) != {CYCLE}:
        raise ValueError("Unexpected cycle value.")

    if set(df["member"].astype(str)) != {member}:
        raise ValueError("member does not match partition path.")

    lead_hours = pd.to_numeric(df["lead_hour"], errors="coerce")
    lead_days = pd.to_numeric(df["lead_day"], errors="coerce")

    if lead_hours.isna().any() or lead_days.isna().any():
        raise ValueError("lead_hour/lead_day contains non-numeric values.")

    if set(lead_hours.astype(int)) != {lead_day * 24}:
        raise ValueError("lead_hour does not match partition path.")

    if set(lead_days.astype(int)) != {lead_day}:
        raise ValueError("lead_day does not match partition path.")

    # Native GEFS crop must remain inside the declared India bounding box.
    for column, (lower, upper) in RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values.")
        if values.min() < lower or values.max() > upper:
            raise ValueError(
                f"{column} outside accepted range "
                f"[{lower}, {upper}]: min={values.min()} max={values.max()}"
            )

    # Validate the native 0.25-degree GEFS grid without relying on raw
    # floating-point diffs. M2.6 writes a 129 x 135 India crop:
    #   latitude  = 6.5 .. 38.5 by 0.25
    #   longitude = 66.5 .. 100.0 by 0.25
    # The lattice check below is tolerant to Parquet/Arrow floating-point
    # representation while still rejecting shifted/regridded coordinates.
    coord = df[["latitude", "longitude"]].drop_duplicates()
    if coord.empty:
        raise ValueError("No grid coordinates found.")

    unique_lat = np.sort(
        df["latitude"].drop_duplicates().to_numpy(dtype=np.float64)
    )
    unique_lon = np.sort(
        df["longitude"].drop_duplicates().to_numpy(dtype=np.float64)
    )

    expected_lat_count = int(round((INDIA_LAT_MAX - INDIA_LAT_MIN) / 0.25)) + 1
    expected_lon_count = int(round((INDIA_LON_MAX - INDIA_LON_MIN) / 0.25)) + 1
    expected_grid_cells = expected_lat_count * expected_lon_count

    if len(unique_lat) != expected_lat_count:
        raise ValueError(
            f"Unexpected latitude count: {len(unique_lat)}; "
            f"expected {expected_lat_count}."
        )
    if len(unique_lon) != expected_lon_count:
        raise ValueError(
            f"Unexpected longitude count: {len(unique_lon)}; "
            f"expected {expected_lon_count}."
        )

    if not np.isclose(unique_lat.min(), INDIA_LAT_MIN, atol=1e-6):
        raise ValueError("Latitude minimum is not the expected native-grid origin.")
    if not np.isclose(unique_lat.max(), INDIA_LAT_MAX, atol=1e-6):
        raise ValueError("Latitude maximum is not the expected native-grid bound.")
    if not np.isclose(unique_lon.min(), INDIA_LON_MIN, atol=1e-6):
        raise ValueError("Longitude minimum is not the expected native-grid origin.")
    if not np.isclose(unique_lon.max(), INDIA_LON_MAX, atol=1e-6):
        raise ValueError("Longitude maximum is not the expected native-grid bound.")

    lat_lattice = (unique_lat - INDIA_LAT_MIN) / 0.25
    lon_lattice = (unique_lon - INDIA_LON_MIN) / 0.25
    if not np.allclose(lat_lattice, np.round(lat_lattice), atol=1e-6):
        raise ValueError("Latitude coordinates are not on the native 0.25-degree lattice.")
    if not np.allclose(lon_lattice, np.round(lon_lattice), atol=1e-6):
        raise ValueError("Longitude coordinates are not on the native 0.25-degree lattice.")

    # Every native grid cell must occur exactly once in a partition.
    duplicate_grid = df.duplicated(["latitude", "longitude"])
    if duplicate_grid.any():
        raise ValueError(
            f"Duplicate grid-cell rows detected: {int(duplicate_grid.sum())}"
        )

    if len(df) != expected_grid_cells:
        raise ValueError(
            f"Unexpected partition row count: {len(df)}; "
            f"expected {expected_grid_cells} for the complete 129 x 135 grid."
        )

    # Check the full Cartesian product using integer lattice indices rather
    # than floating-point MultiIndex equality. This avoids false failures
    # caused by tiny Arrow/Parquet representation differences.
    lat_idx = np.rint((df["latitude"].to_numpy(dtype=np.float64) - INDIA_LAT_MIN) / 0.25).astype(np.int64)
    lon_idx = np.rint((df["longitude"].to_numpy(dtype=np.float64) - INDIA_LON_MIN) / 0.25).astype(np.int64)

    if len(np.unique(lat_idx)) != expected_lat_count:
        raise ValueError("Latitude lattice does not contain all expected rows.")
    if len(np.unique(lon_idx)) != expected_lon_count:
        raise ValueError("Longitude lattice does not contain all expected columns.")

    linear_idx = lat_idx * expected_lon_count + lon_idx
    unique_linear = np.unique(linear_idx)

    if len(unique_linear) != expected_grid_cells:
        missing = expected_grid_cells - len(unique_linear)
        raise ValueError(
            f"Native grid is incomplete: unique_cells={len(unique_linear)}, "
            f"expected={expected_grid_cells}, missing_or_duplicate_cells={missing}."
        )

    expected_linear = np.arange(expected_grid_cells, dtype=np.int64)
    if not np.array_equal(unique_linear, expected_linear):
        missing = int(np.setdiff1d(expected_linear, unique_linear).size)
        extra = int(np.setdiff1d(unique_linear, expected_linear).size)
        raise ValueError(
            f"Native grid mismatch: missing_cells={missing}, extra_cells={extra}."
        )


    return {
        "init_date": init_date.isoformat(),
        "valid_date": expected_valid.isoformat(),
        "cycle": CYCLE,
        "member": member,
        "lead_day": lead_day,
        "lead_hour": lead_day * 24,
        "rows": int(len(df)),
        "grid_cells": int(len(coord)),
        "latitude_min": float(coord["latitude"].min()),
        "latitude_max": float(coord["latitude"].max()),
        "longitude_min": float(coord["longitude"].min()),
        "longitude_max": float(coord["longitude"].max()),
        "file_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "path": str(path),
    }


def discover_partitions() -> list[Path]:
    if not INPUT_ROOT.exists():
        return []
    return sorted(INPUT_ROOT.rglob("forecast.parquet"))


def build_catalog(partitions: list[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, path in enumerate(partitions, start=1):
        try:
            record = validate_partition(path)
            records.append(record)
            print(
                f"[{index:05d}/{len(partitions):05d}] PASS "
                f"{record['init_date']} {record['member']} "
                f"lead={record['lead_day']} rows={record['rows']:,}"
            )
        except Exception as exc:
            failure = {
                "path": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"[{index:05d}/{len(partitions):05d}] FAIL "
                f"{path}: {type(exc).__name__}: {exc}"
            )

    catalog = pd.DataFrame(records)
    if not catalog.empty:
        catalog = catalog.sort_values(
            ["init_date", "member", "lead_day"]
        ).reset_index(drop=True)

    return catalog, failures


def coverage_report(catalog: pd.DataFrame) -> dict[str, Any]:
    expected = expected_keys()
    expected_set = {
        expected_partition_key(*item)
        for item in expected
    }

    actual_set: set[str] = set()
    if not catalog.empty:
        actual_set = {
            expected_partition_key(
                date.fromisoformat(row.init_date),
                str(row.member),
                int(row.lead_day),
            )
            for row in catalog.itertuples(index=False)
        }

    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)

    days_present = (
        sorted(catalog["init_date"].unique().tolist())
        if not catalog.empty
        else []
    )
    members_present = (
        sorted(catalog["member"].unique().tolist())
        if not catalog.empty
        else []
    )
    leads_present = (
        sorted(catalog["lead_day"].unique().tolist())
        if not catalog.empty
        else []
    )

    return {
        "expected_partitions": len(expected_set),
        "present_partitions": len(actual_set),
        "missing_partitions": len(missing),
        "unexpected_partitions": len(unexpected),
        "missing_examples": missing[:25],
        "unexpected_examples": unexpected[:25],
        "days_present": len(days_present),
        "first_init_date": days_present[0] if days_present else None,
        "last_init_date": days_present[-1] if days_present else None,
        "members_present": members_present,
        "lead_days_present": leads_present,
        "complete_target_window": not missing and not unexpected,
    }


def cross_partition_checks(catalog: pd.DataFrame) -> dict[str, Any]:
    if catalog.empty:
        return {
            "valid": False,
            "reason": "No valid partitions were produced.",
        }

    checks: dict[str, Any] = {
        "valid": True,
        "rows_positive": bool((catalog["rows"] > 0).all()),
        "grid_cells_positive": bool((catalog["grid_cells"] > 0).all()),
        "cycle_consistent": set(catalog["cycle"].astype(str)) == {CYCLE},
        "member_set_valid": set(catalog["member"].astype(str)).issubset(
            set(MEMBERS)
        ),
        "lead_set_valid": set(catalog["lead_day"].astype(int)).issubset(
            {lead // 24 for lead in LEADS}
        ),
        "grid_cell_count_consistent": int(catalog["grid_cells"].nunique()) == 1,
        "latitude_bounds_consistent": (
            float(catalog["latitude_min"].min()) >= INDIA_LAT_MIN
            and float(catalog["latitude_max"].max()) <= INDIA_LAT_MAX
        ),
        "longitude_bounds_consistent": (
            float(catalog["longitude_min"].min()) >= INDIA_LON_MIN
            and float(catalog["longitude_max"].max()) <= INDIA_LON_MAX
        ),
    }

    checks["valid"] = all(
        value for key, value in checks.items() if key != "valid"
    )
    return checks


def main() -> int:
    global INPUT_ROOT, OUTPUT_ROOT, MANIFEST_PATH, CATALOG_PATH, REPORT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_ROOT,
        help="M2.6 canonical partition root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="M2.7 manifest/catalog output directory.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="M2.7 validation report.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless the complete 2020-09-23..2025-12-31 target "
        "partition set is present.",
    )
    args = parser.parse_args()

    INPUT_ROOT = args.input
    OUTPUT_ROOT = args.output_root
    MANIFEST_PATH = OUTPUT_ROOT / "dataset_manifest.json"
    CATALOG_PATH = OUTPUT_ROOT / "partition_catalog.parquet"
    REPORT_PATH = args.report

    print("=" * 72)
    print("SANKET-X M2.7 — CANONICAL GEFS DATASET BUILDER")
    print("=" * 72)
    print(f"Input root       : {INPUT_ROOT}")
    print(f"Output root      : {OUTPUT_ROOT}")
    print(f"Target window    : {START_DATE} -> {END_DATE}")
    print(f"Cycle            : {CYCLE}Z")
    print(f"Members          : {', '.join(MEMBERS)}")
    print(f"Lead days        : {[lead // 24 for lead in LEADS]}")
    print("Concatenation    : NO — partitioned, memory-safe catalog")
    print("Interpolation    : NO")
    print("Regridding       : NO")
    print("Fabrication      : NO")
    print()

    partitions = discover_partitions()
    print(f"Partitions found : {len(partitions)}")

    catalog, failures = build_catalog(partitions)

    if failures:
        status = "NOT_PASS"
    else:
        coverage = coverage_report(catalog)
        cross_checks = cross_partition_checks(catalog)
        status = "PASS"

        if not cross_checks["valid"]:
            status = "NOT_PASS"

        if args.require_complete and not coverage["complete_target_window"]:
            status = "NOT_PASS"

    coverage = coverage_report(catalog)
    cross_checks = cross_partition_checks(catalog)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not catalog.empty:
        atomic_write_parquet(catalog, CATALOG_PATH)

    total_rows = int(catalog["rows"].sum()) if not catalog.empty else 0
    total_bytes = (
        int(catalog["file_bytes"].sum()) if not catalog.empty else 0
    )

    manifest = {
        "dataset": "Sanket-X canonical historical GEFSv12",
        "builder": "M2.7",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA GEFSv12 operational archive",
        "target_window": {
            "start": START_DATE.isoformat(),
            "end": END_DATE.isoformat(),
        },
        "cycle": CYCLE,
        "members": MEMBERS,
        "lead_hours": LEADS,
        "lead_days": [lead // 24 for lead in LEADS],
        "india_bbox": {
            "latitude_min": INDIA_LAT_MIN,
            "latitude_max": INDIA_LAT_MAX,
            "longitude_min": INDIA_LON_MIN,
            "longitude_max": INDIA_LON_MAX,
        },
        "required_columns": sorted(REQUIRED_COLUMNS),
        "partition_count": int(len(catalog)),
        "total_rows": total_rows,
        "total_partition_bytes": total_bytes,
        "coverage": coverage,
        "cross_partition_checks": cross_checks,
        "failed_partitions": failures,
        "catalog_path": str(CATALOG_PATH) if not catalog.empty else None,
        "interpolation": False,
        "regridding": False,
        "nearest_neighbor": False,
        "fabricated_values": False,
        "complete_training_dataset": bool(coverage["complete_target_window"]),
        "status": status,
    }

    atomic_write_text(
        MANIFEST_PATH,
        json.dumps(manifest, indent=2, default=str),
    )
    atomic_write_text(
        REPORT_PATH,
        json.dumps(manifest, indent=2, default=str),
    )

    print()
    print("=" * 72)
    print("M2.7 VERDICT")
    print("=" * 72)
    print(f"Valid partitions : {len(catalog)}")
    print(f"Failed partitions: {len(failures)}")
    print(f"Rows represented  : {total_rows:,}")
    print(f"Coverage          : {coverage['present_partitions']}/"
          f"{coverage['expected_partitions']}")
    print(f"Complete target   : "
          f"{coverage['complete_target_window']}")
    print(f"Catalog           : {CATALOG_PATH if not catalog.empty else 'not written'}")
    print(f"Manifest          : {MANIFEST_PATH}")
    print(f"Report            : {REPORT_PATH}")
    print(f"STATUS            : {status}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
