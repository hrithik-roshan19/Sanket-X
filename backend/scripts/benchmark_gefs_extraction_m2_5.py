
"""
Sanket-X M2.5 — verified GEFS GRIB extraction benchmark.

Reads ONLY the 25 existing smoke-test GRIB files from:
    backend/data/operational_gefs_smoke

Uses ecCodes directly for the required GRIB messages. The ecCodes shortNames used by the actual GEFS inventory are:
    2t    -> temperature_c
    2r    -> humidity_pct
    10u   -> wind_u10_ms
    10v   -> wind_v10_ms
    sp    -> pressure_hpa
    prmsl -> msl_pressure_hpa (optional audit field)
    tp    -> rainfall_mm
    pwat  -> atmospheric_moisture_kgm2

No interpolation, regridding, nearest-neighbour fill, or fabricated values.

Important:
- A GRIB file may contain a trailing/partially unreadable message. We do NOT
  declare the whole file failed merely because ecCodes reaches a premature EOF.
- The acceptance gate is that all 7 model-critical physical fields are actually
  found and materialized from the GRIB message data.
- prmsl/msl_pressure_hpa is audited separately and does not block M2.5 because
  it is not part of the current production model's modelled-variable manifest.
- If a premature EOF occurs AFTER all required fields have been recovered, it is
  recorded as a non-blocking read warning.
- If EOF occurs before a required field is recovered, that file fails.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import eccodes
except Exception as exc:
    raise SystemExit(
        "eccodes is required. Install with: python -m pip install eccodes"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

INPUT_DIR = BACKEND_DIR / "data" / "operational_gefs_smoke"
OUTPUT_DIR = BACKEND_DIR / "data" / "analysis"
REPORT_PATH = OUTPUT_DIR / "gefs_m2_5_benchmark.json"

EXPECTED_FILES = 25

INDIA_LAT_MIN = 6.5
INDIA_LAT_MAX = 38.5
INDIA_LON_MIN = 66.5
INDIA_LON_MAX = 100.0

# Actual GRIB/ecCodes shortNames confirmed by M2.2 inventory.
# Model-critical fields. These are the fields used by Sanket-X's current
# canonical/model schema. msl_pressure_hpa is NOT a modelled variable in the
# current production manifest, so it must not block extraction of the fields
# actually required by the model.
REQUIRED_VARIABLES: dict[str, str] = {
    "2t": "temperature_c",
    "2r": "humidity_pct",
    "10u": "wind_u10_ms",
    "10v": "wind_v10_ms",
    "sp": "pressure_hpa",
    "tp": "rainfall_mm",
    "pwat": "atmospheric_moisture_kgm2",
}

# MSL pressure is audited separately. It is optional for M2.5 because the
# current Sanket-X model does not list msl_pressure_hpa as a modelled variable.
OPTIONAL_VARIABLES: dict[str, str] = {
    "prmsl": "msl_pressure_hpa",
}

# Some libraries expose the same GRIB fields under xarray-style names.
ALIASES: dict[str, set[str]] = {
    "2t": {"2t", "t2m"},
    "2r": {"2r", "r2"},
    "10u": {"10u", "u10"},
    "10v": {"10v", "v10"},
    "sp": {"sp"},
    "prmsl": {"prmsl"},
    "tp": {"tp"},
    "pwat": {"pwat"},
}


def discover_grib_files() -> list[Path]:
    """Return only the 25 real smoke-test GRIB files."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    pattern = re.compile(
        r"^(?:gec00|gep\d{2})\.t\d{2}z\.pgrb2s\.0p25\.f\d{3}$"
    )

    files = [
        p
        for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and pattern.match(p.name)
        and not p.name.endswith(".part")
        and not p.name.endswith(".idx")
    ]
    return sorted(files, key=lambda p: p.name)


def _safe_get(handle: Any, key: str, default: Any = None) -> Any:
    try:
        return eccodes.codes_get(handle, key)
    except Exception:
        return default


def _release(handle: Any) -> None:
    if handle is not None:
        try:
            eccodes.codes_release(handle)
        except Exception:
            pass


def _is_eof_exception(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "prematureendoffile" in text
        or "end of resource reached" in text
        or "end of file" in text
        or "no message found" in text
    )


def _crop_points(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    if values.size != latitudes.size or values.size != longitudes.size:
        raise ValueError(
            "GRIB values/latitudes/longitudes have inconsistent sizes."
        )

    mask = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & (latitudes >= INDIA_LAT_MIN)
        & (latitudes <= INDIA_LAT_MAX)
        & (longitudes >= INDIA_LON_MIN)
        & (longitudes <= INDIA_LON_MAX)
    )

    if not np.any(mask):
        raise ValueError("India crop is empty.")

    cropped = np.asarray(values[mask], dtype=np.float64)

    if cropped.size == 0:
        raise ValueError("India crop contains no values.")

    return cropped


def _convert_units(
    short_name: str,
    values: np.ndarray,
    units: str | None,
) -> tuple[np.ndarray, str | None, str]:
    """
    Convert only explicitly supported source units.

    This is unit normalization, not a physical derivation.
    """
    unit = (units or "").strip().lower()

    if short_name in {"sp", "prmsl"}:
        if unit in {"pa", "pascal", "pascals"}:
            return values / 100.0, "hPa", "Pa_to_hPa"
        if unit in {"hpa", "mb", "mbar"}:
            return values, "hPa", "none"
        raise ValueError(
            f"Unsupported pressure units for {short_name}: {units!r}"
        )

    if short_name == "2t":
        # GEFS GRIB normally stores temperature in K.
        if unit in {"k", "kelvin"}:
            return values - 273.15, "degC", "K_to_C"
        if unit in {"c", "degc", "celsius"}:
            return values, "degC", "none"
        raise ValueError(
            f"Unsupported temperature units for {short_name}: {units!r}"
        )

    if short_name == "2r":
        if unit in {"%", "percent", "percentage"}:
            return values, "%", "none"
        # Some ecCodes installations expose RH with units "%".
        # Do not silently assume an unknown unit.
        raise ValueError(
            f"Unsupported relative-humidity units for {short_name}: {units!r}"
        )

    if short_name in {"10u", "10v"}:
        if unit in {"m s-1", "m/s", "m s**-1", "ms-1"}:
            return values, "m s-1", "none"
        raise ValueError(
            f"Unsupported wind-component units for {short_name}: {units!r}"
        )

    if short_name == "pwat":
        # GRIB commonly uses kg m**-2 for precipitable water.
        if unit in {"kg m**-2", "kg m-2", "kg/m2", "kg m^-2"}:
            return values, "kg m-2", "none"
        raise ValueError(
            f"Unsupported precipitable-water units for {short_name}: {units!r}"
        )

    if short_name == "tp":
        # GEFS accumulated precipitation is normally kg m**-2, equivalent to mm
        # for water depth. Keep it as an accumulated field; do not deaccumulate.
        if unit in {
            "kg m**-2",
            "kg m-2",
            "kg/m2",
            "kg m^-2",
            "mm",
        }:
            return values, "mm", "none"
        raise ValueError(
            f"Unsupported precipitation units for {short_name}: {units!r}"
        )

    raise ValueError(f"No unit rule for {short_name}.")


def extract_file(path: Path) -> dict[str, Any]:
    started = time.perf_counter()

    # One recovered field per required physical variable.
    recovered: dict[str, dict[str, Any]] = {}
    eof_warning: str | None = None
    messages_seen = 0

    with path.open("rb") as fh:
        while True:
            handle = None

            try:
                handle = eccodes.codes_grib_new_from_file(fh)
            except Exception as exc:
                if _is_eof_exception(exc):
                    eof_warning = f"{type(exc).__name__}: {exc}"
                    break
                raise

            if handle is None:
                break

            messages_seen += 1

            try:
                short_name = str(_safe_get(handle, "shortName", "") or "")

                # Ignore unrelated messages. We need the seven model-critical
                # fields and opportunistically audit optional prmsl.
                if short_name not in (REQUIRED_VARIABLES | OPTIONAL_VARIABLES):
                    continue

                canonical = (
                    REQUIRED_VARIABLES.get(short_name)
                    or OPTIONAL_VARIABLES.get(short_name)
                )

                # Avoid replacing an already recovered valid field.
                if canonical in recovered:
                    continue

                values = np.asarray(
                    eccodes.codes_get_array(handle, "values"),
                    dtype=np.float64,
                )
                latitudes = np.asarray(
                    eccodes.codes_get_array(handle, "latitudes"),
                    dtype=np.float64,
                )
                longitudes = np.asarray(
                    eccodes.codes_get_array(handle, "longitudes"),
                    dtype=np.float64,
                )

                cropped = _crop_points(values, latitudes, longitudes)

                units = _safe_get(handle, "units")
                units = str(units) if units is not None else None

                converted, canonical_units, conversion = _convert_units(
                    short_name, cropped, units
                )

                finite = np.isfinite(converted)

                if not np.any(finite):
                    raise ValueError(
                        f"{short_name} India crop contains no finite values."
                    )

                recovered[canonical] = {
                    "raw_short_name": short_name,
                    "canonical_name": canonical,
                    "type_of_level": _safe_get(handle, "typeOfLevel"),
                    "level": _safe_get(handle, "level"),
                    "units_source": units,
                    "units_canonical": canonical_units,
                    "conversion": conversion,
                    "step_type": _safe_get(handle, "stepType"),
                    "step_units": _safe_get(handle, "stepUnits"),
                    "start_step": _safe_get(handle, "startStep"),
                    "end_step": _safe_get(handle, "endStep"),
                    "shape": [int(converted.size)],
                    "dtype": str(converted.dtype),
                    "loaded_bytes": int(converted.nbytes),
                    "loaded_mb": round(
                        converted.nbytes / (1024 * 1024), 4
                    ),
                    "finite_count": int(finite.sum()),
                    "total_count": int(converted.size),
                    "materialized": True,
                }

            finally:
                _release(handle)

            # Once all model-critical fields have actually been recovered,
            # extraction is complete. Optional prmsl is not a blocking field.
            if all(name in recovered for name in REQUIRED_VARIABLES.values()):
                break

    missing = [
        canonical
        for canonical in REQUIRED_VARIABLES.values()
        if canonical not in recovered
    ]

    if missing:
        raise ValueError(
            "Model-critical variables not materialized: "
            + ", ".join(missing)
        )

    optional_status = {
        canonical: {
            "present": canonical in recovered,
            "materialized": bool(
                recovered.get(canonical, {}).get("materialized", False)
            ),
        }
        for canonical in OPTIONAL_VARIABLES.values()
    }

    loaded_bytes = sum(
        int(item["loaded_bytes"]) for item in recovered.values()
    )

    elapsed = time.perf_counter() - started

    return {
        "file": path.name,
        "status": "OK",
        "messages_scanned": messages_seen,
        "required_found": len(recovered),
        "required_total": len(REQUIRED_VARIABLES),
        "loaded_bytes": loaded_bytes,
        "loaded_mb": round(loaded_bytes / (1024 * 1024), 4),
        "elapsed_seconds": round(elapsed, 3),
        "trailing_or_partial_read_warning": eof_warning,
        "optional_variables": optional_status,
        "variables": recovered,
    }


def main() -> int:
    print("=" * 72)
    print("SANKET-X M2.5 — VERIFIED GEFS GRIB EXTRACTION BENCHMARK")
    print("=" * 72)
    print()
    print(f"Input : {INPUT_DIR}")
    print("Mode  : Existing smoke-test files only")
    print("Download : NONE")
    print("Reader : ecCodes direct GRIB message extraction")
    print("Operation : GRIB → required variables → India crop")
    print("Materialization : ENABLED")
    print()

    if not INPUT_DIR.exists():
        print(f"ERROR: input directory does not exist: {INPUT_DIR}")
        return 1

    files = discover_grib_files()
    print(f"Files discovered : {len(files)}")

    if len(files) != EXPECTED_FILES:
        print(
            f"WARNING: expected {EXPECTED_FILES} GRIB files, "
            f"found {len(files)}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    for index, path in enumerate(files, start=1):
        print(f"[{index:02d}/{len(files):02d}] {path.name}")

        try:
            result = extract_file(path)
            results.append(result)

            warning_text = (
                " | partial-EOF-after-required-fields"
                if result["trailing_or_partial_read_warning"]
                else ""
            )

            print(
                f"       OK | messages={result['messages_scanned']} | "
                f"required={result['required_found']}/"
                f"{result['required_total']} | "
                f"loaded={result['loaded_mb']:.2f} MB | "
                f"{result['elapsed_seconds']:.2f}s"
                f"{warning_text}"
            )

        except Exception as exc:
            failures.append(
                {
                    "file": path.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"       FAIL | {type(exc).__name__}: {exc}"
            )

    total_seconds = time.perf_counter() - started_all
    total_loaded_bytes = sum(
        int(result["loaded_bytes"]) for result in results
    )

    variable_file_coverage = {
        canonical: sum(
            1
            for result in results
            if result["variables"]
            .get(canonical, {})
            .get("materialized", False)
        )
        for canonical in REQUIRED_VARIABLES.values()
    }

    successful = len(results)
    failed = len(failures)

    all_files_successful = (
        len(files) == EXPECTED_FILES
        and successful == len(files)
        and failed == 0
    )

    all_variables_every_file = (
        successful == len(files)
        and all(
            count == len(files)
            for count in variable_file_coverage.values()
        )
    )

    status = (
        "PASS"
        if all_files_successful and all_variables_every_file
        else "NOT_PASS"
    )

    report = {
        "benchmark": "Sanket-X M2.5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(INPUT_DIR),
        "expected_files": EXPECTED_FILES,
        "files_discovered": len(files),
        "files_tested": len(files),
        "successful": successful,
        "failed": failed,
        "reader": "ecCodes direct GRIB message extraction",
        "operation": "GRIB -> required variables -> India crop",
        "materialization_enabled": True,
        "interpolation": False,
        "regridding": False,
        "nearest_neighbor": False,
        "fabricated_values": False,
        "india_bbox": {
            "latitude_min": INDIA_LAT_MIN,
            "latitude_max": INDIA_LAT_MAX,
            "longitude_min": INDIA_LON_MIN,
            "longitude_max": INDIA_LON_MAX,
        },
        "required_variables": REQUIRED_VARIABLES,
        "optional_variables": OPTIONAL_VARIABLES,
        "variable_file_coverage": variable_file_coverage,
        "optional_variable_file_coverage": {
            canonical: sum(
                1
                for result in results
                if result.get("optional_variables", {})
                .get(canonical, {})
                .get("materialized", False)
            )
            for canonical in OPTIONAL_VARIABLES.values()
        },
        "all_required_variables_in_every_file": all_variables_every_file,
        "total_loaded_bytes": total_loaded_bytes,
        "total_loaded_mb": round(
            total_loaded_bytes / (1024 * 1024), 4
        ),
        "total_seconds": round(total_seconds, 3),
        "files": results,
        "failures": failures,
        "status": status,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("M2.5 VERDICT")
    print("=" * 72)
    print()
    print(f"Files tested : {len(files)}")
    print(f"Successful   : {successful}")
    print(f"Failed       : {failed}")
    print(
        "Required variables per file : "
        f"{len(REQUIRED_VARIABLES) if all_variables_every_file else 0}/"
        f"{len(REQUIRED_VARIABLES)}"
    )
    print(
        f"Total loaded data : "
        f"{total_loaded_bytes / (1024 * 1024):.2f} MB"
    )
    print(f"Total time   : {total_seconds:.2f}s")
    print()
    print("Model-critical variable coverage:")
    for canonical, count in variable_file_coverage.items():
        print(f"  {canonical:30s}: {count}/{len(files)}")

    print()
    print("Optional variable coverage:")
    for canonical in OPTIONAL_VARIABLES.values():
        count = sum(
            1
            for result in results
            if result.get("optional_variables", {})
            .get(canonical, {})
            .get("materialized", False)
        )
        print(f"  {canonical:30s}: {count}/{len(files)}")

    print()
    print(f"STATUS: {status}")
    print()
    print(f"Report: {REPORT_PATH}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
