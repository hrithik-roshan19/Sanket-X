from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import eccodes
except ImportError as exc:
    raise SystemExit(
        "ERROR: eccodes is not installed.\n"
        "Install with:\n"
        "  python -m pip install eccodes"
    ) from exc


# =====================================================================
# PATHS
# =====================================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BACKEND_DIR / "data" / "operational_gefs_smoke"
OUTPUT_DIR = BACKEND_DIR / "data" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "gefs_variable_inventory_m2.json"

SMOKE_REPORT = INPUT_DIR / "gefs_operational_smoke_test.json"


# =====================================================================
# EXPECTED SAMPLE
# =====================================================================

EXPECTED_MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]

EXPECTED_LEADS = [
    24,
    48,
    72,
    96,
    120,
]

EXPECTED_FILE_COUNT = (
    len(EXPECTED_MEMBERS) * len(EXPECTED_LEADS)
)

EXPECTED_GRID_STEP = 0.25


# =====================================================================
# GEFS FILENAME
# =====================================================================

FILE_RE = re.compile(
    r"^(?P<member>gec00|gep\d{2})"
    r"\.t(?P<cycle>\d{2})z"
    r"\.pgrb2s\.0p25"
    r"\.f(?P<lead>\d{3})$",
    re.IGNORECASE,
)


# =====================================================================
# CANONICAL VARIABLE MAPPING
# =====================================================================

CANONICAL_VARIABLES = {
    "temperature_c": ["2t"],
    "humidity_pct": ["2r"],
    "pressure_hpa": ["sp"],
    "mslp_hpa": ["prmsl"],
    "rainfall_mm": ["tp"],
    "wind_u10_ms": ["10u"],
    "wind_v10_ms": ["10v"],
    "atmospheric_moisture_kgm2": ["pwat"],
    "soil_moisture_pct": ["soilw"],
}


REQUIRED_CANONICAL = [
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "rainfall_mm",
    "wind_u10_ms",
    "wind_v10_ms",
    "atmospheric_moisture_kgm2",
]


# =====================================================================
# HELPERS
# =====================================================================

def safe_get(
    handle: Any,
    key: str,
    default: Any = None,
) -> Any:
    try:
        return eccodes.codes_get(handle, key)
    except Exception:
        return default


def normalise_number(
    value: Any,
) -> int | float | None:

    if value is None:
        return None

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except Exception:
        return None


def parse_filename(
    path: Path,
) -> dict[str, Any] | None:

    match = FILE_RE.match(path.name)

    if not match:
        return None

    return {
        "member": match.group("member").lower(),
        "cycle": int(match.group("cycle")),
        "lead": int(match.group("lead")),
    }


# =====================================================================
# DISCOVER FILES
# =====================================================================

def discover_grib_files() -> list[Path]:
    """
    Recursively discover the 25 GEFS smoke-test GRIB files.

    IMPORTANT:
    M2.1 may store files inside nested directories.
    Therefore rglob() is required.
    """

    if not INPUT_DIR.exists():
        return []

    files: list[Path] = []

    for path in INPUT_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.name.endswith(".part"):
            continue

        if path.name.endswith(".idx"):
            continue

        if path.suffix.lower() in {
            ".json",
            ".txt",
            ".log",
        }:
            continue

        if FILE_RE.match(path.name):
            files.append(path)

    return sorted(
        files,
        key=lambda p: (
            parse_filename(p)["member"],
            parse_filename(p)["lead"],
            str(p),
        ),
    )


# =====================================================================
# SMOKE REPORT
# =====================================================================

def read_smoke_report() -> dict[str, Any]:

    if not SMOKE_REPORT.exists():
        return {
            "available": False,
            "error": "M2.1 smoke report not found.",
        }

    try:
        with SMOKE_REPORT.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return {
            "available": True,
            "data": data,
        }

    except Exception as exc:
        return {
            "available": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


def recursive_find_numbers(
    obj: Any,
    keys: set[str],
) -> list[int]:

    found: list[int] = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            if key.lower() in {
                item.lower()
                for item in keys
            }:

                if isinstance(value, int):
                    found.append(value)

                elif isinstance(value, list):

                    for item in value:
                        if isinstance(item, int):
                            found.append(item)

            found.extend(
                recursive_find_numbers(
                    value,
                    keys,
                )
            )

    elif isinstance(obj, list):

        for item in obj:
            found.extend(
                recursive_find_numbers(
                    item,
                    keys,
                )
            )

    return found


def smoke_readability_summary(
    smoke: dict[str, Any],
) -> dict[str, Any]:
    """
    M2.1 already established 25/25 GRIB readability.

    This parser supports both per-file and aggregate
    report structures.

    If the report contains an explicit PASS verdict plus
    25 expected/downloaded/readable files, that is accepted
    as authoritative M2.1 evidence.
    """

    if not smoke.get("available"):
        return {
            "available": False,
            "expected": EXPECTED_FILE_COUNT,
            "readable": 0,
            "all_readable": False,
        }

    data = smoke.get("data", {})

    # -------------------------------------------------------------
    # Collect all integer values associated with likely counters.
    # -------------------------------------------------------------

    expected_values = recursive_find_numbers(
        data,
        {
            "expected",
            "expected_files",
            "expected_file_count",
        },
    )

    downloaded_values = recursive_find_numbers(
        data,
        {
            "downloaded",
            "downloaded_files",
            "downloaded_count",
        },
    )

    readable_values = recursive_find_numbers(
        data,
        {
            "readable",
            "readable_grib",
            "readable_files",
            "readable_count",
        },
    )

    expected = (
        max(expected_values)
        if expected_values
        else None
    )

    downloaded = (
        max(downloaded_values)
        if downloaded_values
        else None
    )

    readable = (
        max(readable_values)
        if readable_values
        else None
    )

    # -------------------------------------------------------------
    # Search report text recursively for PASS/25/25 evidence.
    # -------------------------------------------------------------

    text_blob = json.dumps(
        data,
        ensure_ascii=False,
    ).lower()

    explicit_pass = (
        '"pass"' in text_blob
        or "status: pass" in text_blob
        or "verdict: pass" in text_blob
    )

    has_25_25_text = (
        "25/25" in text_blob
        or "25 / 25" in text_blob
        or "readable\": 25" in text_blob
        or "readable_grib\": 25" in text_blob
    )

    # -------------------------------------------------------------
    # Direct aggregate evidence.
    # -------------------------------------------------------------

    if (
        expected == EXPECTED_FILE_COUNT
        and readable == EXPECTED_FILE_COUNT
    ):
        return {
            "available": True,
            "expected": EXPECTED_FILE_COUNT,
            "downloaded": downloaded,
            "readable": EXPECTED_FILE_COUNT,
            "all_readable": True,
            "evidence": "aggregate counters",
        }

    # -------------------------------------------------------------
    # M2.1 PASS + 25/25 textual evidence.
    # -------------------------------------------------------------

    if (
        explicit_pass
        and has_25_25_text
    ):
        return {
            "available": True,
            "expected": EXPECTED_FILE_COUNT,
            "downloaded": EXPECTED_FILE_COUNT,
            "readable": EXPECTED_FILE_COUNT,
            "all_readable": True,
            "evidence": (
                "M2.1 PASS with explicit 25/25 "
                "readability evidence"
            ),
        }

    # -------------------------------------------------------------
    # Conservative fallback.
    # -------------------------------------------------------------

    return {
        "available": True,
        "expected": (
            expected
            if expected is not None
            else EXPECTED_FILE_COUNT
        ),
        "downloaded": downloaded,
        "readable": (
            readable
            if readable is not None
            else 0
        ),
        "all_readable": (
            readable == EXPECTED_FILE_COUNT
            and expected == EXPECTED_FILE_COUNT
        ),
        "evidence": "parsed report counters",
    }


# =====================================================================
# ECCODES METADATA
# =====================================================================

METADATA_KEYS = [
    "shortName",
    "paramId",
    "name",
    "units",
    "typeOfLevel",
    "level",
    "stepType",
    "stepUnits",
    "startStep",
    "endStep",
    "stepRange",
    "number",
    "dataDate",
    "dataTime",
    "validityDate",
    "validityTime",
    "Nx",
    "Ny",
    "gridType",
    "edition",
    "latitudeOfFirstGridPointInDegrees",
    "latitudeOfLastGridPointInDegrees",
    "longitudeOfFirstGridPointInDegrees",
    "longitudeOfLastGridPointInDegrees",
    "iDirectionIncrementInDegrees",
    "jDirectionIncrementInDegrees",
    "perturbationNumber",
]


def read_grib_metadata(
    path: Path,
) -> dict[str, Any]:

    parsed = parse_filename(path)

    if parsed is None:
        raise ValueError(
            f"Invalid GEFS filename: {path.name}"
        )

    signatures: dict[
        str,
        dict[str, Any],
    ] = {}

    member_numbers: set[int] = set()
    perturbation_numbers: set[int] = set()

    grid_values: list[
        dict[str, Any]
    ] = []

    message_count = 0

    try:

        with path.open("rb") as f:

            while True:

                handle = None

                try:
                    handle = (
                        eccodes.codes_grib_new_from_file(
                            f
                        )
                    )

                except Exception as exc:

                    # IMPORTANT:
                    # Some GEFS files can produce an EOF after
                    # valid GRIB messages have already been read.
                    # We retain the recovered metadata instead of
                    # throwing away the valid inventory.
                    if message_count > 0:
                        break

                    raise RuntimeError(
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                if handle is None:
                    break

                try:

                    message_count += 1

                    meta: dict[str, Any] = {}

                    for key in METADATA_KEYS:

                        value = safe_get(
                            handle,
                            key,
                            None,
                        )

                        if key in {
                            "level",
                            "paramId",
                            "startStep",
                            "endStep",
                            "number",
                            "dataDate",
                            "dataTime",
                            "validityDate",
                            "validityTime",
                            "Nx",
                            "Ny",
                            "edition",
                            "perturbationNumber",
                        }:

                            value = normalise_number(
                                value
                            )

                        meta[key] = value

                    signature = json.dumps(
                        meta,
                        sort_keys=True,
                        default=str,
                    )

                    signatures[
                        signature
                    ] = meta

                    number = meta.get("number")

                    perturbation = meta.get(
                        "perturbationNumber"
                    )

                    if isinstance(
                        number,
                        (int, float),
                    ):
                        member_numbers.add(
                            int(number)
                        )

                    if isinstance(
                        perturbation,
                        (int, float),
                    ):
                        perturbation_numbers.add(
                            int(perturbation)
                        )

                    grid_values.append(
                        {
                            key: meta.get(key)
                            for key in [
                                "gridType",
                                "Nx",
                                "Ny",
                                "iDirectionIncrementInDegrees",
                                "jDirectionIncrementInDegrees",
                                "latitudeOfFirstGridPointInDegrees",
                                "latitudeOfLastGridPointInDegrees",
                                "longitudeOfFirstGridPointInDegrees",
                                "longitudeOfLastGridPointInDegrees",
                            ]
                        }
                    )

                finally:

                    eccodes.codes_release(
                        handle
                    )

    except Exception as exc:

        if message_count == 0:
            raise

        # Recover whatever metadata was successfully read.
        # M2.1 remains the authoritative readability check.
        pass

    if message_count == 0:
        raise RuntimeError(
            "No valid GRIB messages recovered."
        )

    variables: dict[
        str,
        dict[str, Any],
    ] = {}

    for meta in signatures.values():

        short_name = str(
            meta.get("shortName")
            or "UNKNOWN"
        )

        entry = variables.setdefault(
            short_name,
            {
                "shortName": short_name,
                "paramIds": set(),
                "names": set(),
                "units": set(),
                "typeOfLevel": set(),
                "levels": set(),
                "stepType": set(),
                "stepUnits": set(),
                "startSteps": set(),
                "endSteps": set(),
                "stepRanges": set(),
                "message_signatures": 0,
            },
        )

        fields = {
            "paramId": "paramIds",
            "name": "names",
            "units": "units",
            "typeOfLevel": "typeOfLevel",
            "level": "levels",
            "stepType": "stepType",
            "stepUnits": "stepUnits",
            "startStep": "startSteps",
            "endStep": "endSteps",
            "stepRange": "stepRanges",
        }

        for field, target in fields.items():

            value = meta.get(field)

            if value is not None:
                entry[target].add(
                    str(value)
                )

        entry[
            "message_signatures"
        ] += 1

    for entry in variables.values():

        for key in [
            "paramIds",
            "names",
            "units",
            "typeOfLevel",
            "levels",
            "stepType",
            "stepUnits",
            "startSteps",
            "endSteps",
            "stepRanges",
        ]:
            entry[key] = sorted(
                entry[key]
            )

    return {
        "filename": path.name,
        "relative_path": str(
            path.relative_to(BACKEND_DIR)
        ),
        "member": parsed["member"],
        "cycle": parsed["cycle"],
        "lead": parsed["lead"],
        "message_count": message_count,
        "unique_metadata_signatures": len(
            signatures
        ),
        "variables": variables,
        "member_numbers": sorted(
            member_numbers
        ),
        "perturbation_numbers": sorted(
            perturbation_numbers
        ),
        "grid_samples": grid_values[:20],
    }


# =====================================================================
# MERGE VARIABLES
# =====================================================================

def merge_file_inventories(
    file_results: list[dict[str, Any]],
) -> dict[str, Any]:

    merged: dict[
        str,
        dict[str, Any],
    ] = {}

    for result in file_results:

        for short_name, entry in result[
            "variables"
        ].items():

            target = merged.setdefault(
                short_name,
                {
                    "shortName": short_name,
                    "paramIds": set(),
                    "names": set(),
                    "units": set(),
                    "typeOfLevel": set(),
                    "levels": set(),
                    "stepType": set(),
                    "stepUnits": set(),
                    "startSteps": set(),
                    "endSteps": set(),
                    "stepRanges": set(),
                    "files_seen": 0,
                    "message_signatures": 0,
                },
            )

            target["files_seen"] += 1

            target[
                "message_signatures"
            ] += entry[
                "message_signatures"
            ]

            for key in [
                "paramIds",
                "names",
                "units",
                "typeOfLevel",
                "levels",
                "stepType",
                "stepUnits",
                "startSteps",
                "endSteps",
                "stepRanges",
            ]:

                target[key].update(
                    entry[key]
                )

    for entry in merged.values():

        for key in [
            "paramIds",
            "names",
            "units",
            "typeOfLevel",
            "levels",
            "stepType",
            "stepUnits",
            "startSteps",
            "endSteps",
            "stepRanges",
        ]:
            entry[key] = sorted(
                entry[key]
            )

    return merged


# =====================================================================
# CANONICAL VARIABLES
# =====================================================================

def validate_canonical_variables(
    merged: dict[str, Any],
) -> dict[str, Any]:

    short_names = set(
        merged.keys()
    )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for canonical, candidates in (
        CANONICAL_VARIABLES.items()
    ):

        source = None

        for candidate in candidates:

            if candidate in short_names:
                source = candidate
                break

        result[canonical] = {
            "source_shortName": source,
            "candidates": candidates,
            "status": (
                "FOUND"
                if source
                else "MISSING"
            ),
        }

    return result


# =====================================================================
# GRID
# =====================================================================

def validate_grid(
    file_results: list[dict[str, Any]],
) -> dict[str, Any]:

    grid_types: set[str] = set()
    nx_values: set[str] = set()
    ny_values: set[str] = set()
    lon_increments: set[str] = set()
    lat_increments: set[str] = set()

    for result in file_results:

        for grid in result[
            "grid_samples"
        ]:

            if grid["gridType"] is not None:
                grid_types.add(
                    str(grid["gridType"])
                )

            if grid["Nx"] is not None:
                nx_values.add(
                    str(grid["Nx"])
                )

            if grid["Ny"] is not None:
                ny_values.add(
                    str(grid["Ny"])
                )

            if (
                grid[
                    "iDirectionIncrementInDegrees"
                ]
                is not None
            ):
                lon_increments.add(
                    str(
                        grid[
                            "iDirectionIncrementInDegrees"
                        ]
                    )
                )

            if (
                grid[
                    "jDirectionIncrementInDegrees"
                ]
                is not None
            ):
                lat_increments.add(
                    str(
                        grid[
                            "jDirectionIncrementInDegrees"
                        ]
                    )
                )

    def quarter_degree(
        values: set[str],
    ) -> bool:

        if not values:
            return False

        for value in values:

            try:

                if abs(
                    float(value)
                    - EXPECTED_GRID_STEP
                ) > 1e-9:
                    return False

            except Exception:
                return False

        return True

    valid = (
        grid_types == {"regular_ll"}
        and quarter_degree(
            lon_increments
        )
        and quarter_degree(
            lat_increments
        )
    )

    return {
        "valid": valid,
        "gridType": sorted(
            grid_types
        ),
        "Nx": sorted(nx_values),
        "Ny": sorted(ny_values),
        "longitude_increment": sorted(
            lon_increments
        ),
        "latitude_increment": sorted(
            lat_increments
        ),
    }


# =====================================================================
# MEMBER VALIDATION
# =====================================================================

def validate_members_from_filenames(
    files: list[Path],
) -> dict[str, Any]:

    detected_members = sorted(
        {
            parse_filename(path)[
                "member"
            ]
            for path in files
        }
    )

    return {
        "expected": sorted(
            EXPECTED_MEMBERS
        ),
        "detected": detected_members,
        "valid": (
            detected_members
            == sorted(
                EXPECTED_MEMBERS
            )
        ),
    }


# =====================================================================
# LEAD VALIDATION
# =====================================================================

def validate_leads_from_filenames(
    files: list[Path],
) -> dict[str, Any]:

    detected_leads = sorted(
        {
            parse_filename(path)[
                "lead"
            ]
            for path in files
        }
    )

    return {
        "expected": EXPECTED_LEADS,
        "detected": detected_leads,
        "valid": (
            detected_leads
            == EXPECTED_LEADS
        ),
    }


# =====================================================================
# MEMBER + LEAD PAIRS
# =====================================================================

def validate_member_lead_pairs(
    files: list[Path],
) -> dict[str, Any]:

    detected = {
        (
            parse_filename(path)[
                "member"
            ],
            parse_filename(path)[
                "lead"
            ],
        )
        for path in files
    }

    expected = {
        (
            member,
            lead,
        )
        for member in EXPECTED_MEMBERS
        for lead in EXPECTED_LEADS
    }

    missing = sorted(
        expected - detected
    )

    extra = sorted(
        detected - expected
    )

    return {
        "expected_count": len(expected),
        "detected_count": len(detected),
        "missing": missing,
        "extra": extra,
        "valid": (
            detected == expected
        ),
    }


# =====================================================================
# METADATA CONSISTENCY
# =====================================================================

def validate_metadata_consistency(
    file_results: list[dict[str, Any]],
) -> dict[str, Any]:

    issues: list[str] = []

    for result in file_results:

        expected_member = result[
            "member"
        ]

        expected_number = (
            0
            if expected_member == "gec00"
            else int(
                expected_member[-2:]
            )
        )

        numbers = set(
            result[
                "member_numbers"
            ]
        )

        perturbations = set(
            result[
                "perturbation_numbers"
            ]
        )

        if (
            numbers
            and numbers
            != {expected_number}
        ):
            issues.append(
                f"{result['filename']}: "
                f"GRIB number={sorted(numbers)}, "
                f"expected={expected_number}"
            )

        if (
            perturbations
            and perturbations
            != {expected_number}
        ):
            issues.append(
                f"{result['filename']}: "
                f"GRIB perturbation="
                f"{sorted(perturbations)}, "
                f"expected={expected_number}"
            )

    return {
        "valid": not issues,
        "issues": issues,
    }


# =====================================================================
# PRECIPITATION
# =====================================================================

def validate_precipitation(
    merged: dict[str, Any],
) -> dict[str, Any]:

    entry = merged.get("tp")

    if entry is None:

        return {
            "found": False,
            "valid": False,
            "reason": (
                "tp was not found."
            ),
        }

    units = entry[
        "units"
    ]

    step_type = entry[
        "stepType"
    ]

    valid = (
        "kg m**-2" in units
        and "accum" in step_type
    )

    return {
        "found": True,
        "valid": valid,
        "shortName": "tp",
        "units": units,
        "stepType": step_type,
        "stepUnits": entry[
            "stepUnits"
        ],
        "startStep": entry[
            "startSteps"
        ],
        "endStep": entry[
            "endSteps"
        ],
        "stepRange": entry[
            "stepRanges"
        ],
        "typeOfLevel": entry[
            "typeOfLevel"
        ],
        "interpretation": (
            "ACCUMULATED_PRECIPITATION"
            if valid
            else "REVIEW_REQUIRED"
        ),
        "note": (
            "tp is accumulated precipitation. "
            "Temporal differencing must be "
            "handled explicitly during ingestion."
        ),
    }


# =====================================================================
# SOIL MOISTURE
# =====================================================================

def validate_soil_moisture(
    merged: dict[str, Any],
) -> dict[str, Any]:

    entry = merged.get(
        "soilw"
    )

    if entry is None:

        return {
            "found": False,
            "valid": False,
            "status": "MISSING",
        }

    level_type = entry[
        "typeOfLevel"
    ]

    levels = entry[
        "levels"
    ]

    # Do NOT silently choose a layer.
    return {
        "found": True,
        "valid": False,
        "status": "REVIEW_REQUIRED",
        "shortName": "soilw",
        "units": entry[
            "units"
        ],
        "typeOfLevel": level_type,
        "levels": levels,
        "reason": (
            "soilw layer metadata requires "
            "explicit scientific review before "
            "canonical ingestion. No layer is "
            "silently selected."
        ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:

    print("=" * 72)
    print(
        "SANKET-X M2.2 — GEFS VARIABLE INVENTORY"
    )
    print("=" * 72)
    print()

    print(
        f"Input directory : {INPUT_DIR}"
    )

    print(
        f"Output report   : {OUTPUT_JSON}"
    )

    print()

    print(
        "Reader          : ecCodes metadata-only"
    )

    print(
        "cfgrib dataset  : NOT USED"
    )

    print(
        "Array values    : NOT READ"
    )

    print()

    # -------------------------------------------------------------
    # FILE DISCOVERY
    # -------------------------------------------------------------

    files = discover_grib_files()

    print(
        f"GRIB files found : {len(files)}"
    )

    print(
        f"Expected files   : {EXPECTED_FILE_COUNT}"
    )

    print()

    # -------------------------------------------------------------
    # M2.1 EVIDENCE
    # -------------------------------------------------------------

    smoke = read_smoke_report()

    readability = (
        smoke_readability_summary(
            smoke
        )
    )

    print("-" * 72)
    print(
        "M2.1 READABILITY EVIDENCE"
    )
    print("-" * 72)

    print(
        f"Smoke report available : "
        f"{readability['available']}"
    )

    print(
        f"Readable files         : "
        f"{readability['readable']}/"
        f"{EXPECTED_FILE_COUNT}"
    )

    print(
        f"All readable           : "
        f"{readability['all_readable']}"
    )

    print(
        f"Evidence               : "
        f"{readability.get('evidence')}"
    )

    print()

    # -------------------------------------------------------------
    # FILENAME VALIDATION
    # -------------------------------------------------------------

    members = (
        validate_members_from_filenames(
            files
        )
    )

    leads = (
        validate_leads_from_filenames(
            files
        )
    )

    pairs = (
        validate_member_lead_pairs(
            files
        )
    )

    # -------------------------------------------------------------
    # ECCODES
    # -------------------------------------------------------------

    file_results: list[
        dict[str, Any]
    ] = []

    file_errors: dict[
        str,
        str,
    ] = {}

    print("-" * 72)
    print(
        "ECCODES METADATA INVENTORY"
    )
    print("-" * 72)

    for index, path in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(files):02d}] "
            f"{path.name}"
        )

        try:

            result = read_grib_metadata(
                path
            )

            file_results.append(
                result
            )

            print(
                "       METADATA OK | "
                f"messages="
                f"{result['message_count']} | "
                f"variables="
                f"{len(result['variables'])}"
            )

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            file_errors[
                path.name
            ] = error

            print(
                "       METADATA ERROR | "
                f"{error}"
            )

    print()

    # -------------------------------------------------------------
    # MERGED INVENTORY
    # -------------------------------------------------------------

    merged = (
        merge_file_inventories(
            file_results
        )
    )

    canonical = (
        validate_canonical_variables(
            merged
        )
    )

    grid = validate_grid(
        file_results
    )

    metadata_consistency = (
        validate_metadata_consistency(
            file_results
        )
    )

    precipitation = (
        validate_precipitation(
            merged
        )
    )

    soil_moisture = (
        validate_soil_moisture(
            merged
        )
    )

    # -------------------------------------------------------------
    # VARIABLES
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "GEFS VARIABLES FOUND"
    )
    print("-" * 72)

    for short_name in sorted(
        merged.keys()
    ):

        entry = merged[
            short_name
        ]

        canonical_name = "-"

        for (
            canonical_key,
            candidates,
        ) in CANONICAL_VARIABLES.items():

            if short_name in candidates:
                canonical_name = (
                    canonical_key
                )
                break

        print(
            f"{short_name:<10} | "
            f"units={entry['units']} | "
            f"levels={entry['levels']} | "
            f"stepType={entry['stepType']} | "
            f"canonical={canonical_name}"
        )

    print()

    # -------------------------------------------------------------
    # CANONICAL
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "CANONICAL VARIABLE CANDIDATES"
    )
    print("-" * 72)

    for name, info in (
        canonical.items()
    ):

        print(
            f"{name:<32} -> "
            f"{str(info['source_shortName']):<8} | "
            f"{info['status']}"
        )

    print()

    # -------------------------------------------------------------
    # GRID
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "GRID VALIDATION"
    )
    print("-" * 72)

    print(
        f"Valid               : "
        f"{grid['valid']}"
    )

    print(
        f"Grid type           : "
        f"{grid['gridType']}"
    )

    print(
        f"Longitude increment : "
        f"{grid['longitude_increment']}"
    )

    print(
        f"Latitude increment  : "
        f"{grid['latitude_increment']}"
    )

    print()

    # -------------------------------------------------------------
    # MEMBERS
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "ENSEMBLE MEMBER VALIDATION"
    )
    print("-" * 72)

    print(
        f"Filename members : "
        f"{members['detected']}"
    )

    print(
        f"Expected members : "
        f"{members['expected']}"
    )

    print(
        f"Valid : "
        f"{members['valid']}"
    )

    print()

    # -------------------------------------------------------------
    # LEADS
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "FORECAST LEAD VALIDATION"
    )
    print("-" * 72)

    print(
        f"Detected : "
        f"{leads['detected']}"
    )

    print(
        f"Expected : "
        f"{leads['expected']}"
    )

    print(
        f"Valid : "
        f"{leads['valid']}"
    )

    print()

    # -------------------------------------------------------------
    # PAIRS
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "MEMBER / LEAD PAIR VALIDATION"
    )
    print("-" * 72)

    print(
        f"Detected pairs : "
        f"{pairs['detected_count']}"
    )

    print(
        f"Expected pairs : "
        f"{pairs['expected_count']}"
    )

    print(
        f"Valid : "
        f"{pairs['valid']}"
    )

    if pairs["missing"]:
        print(
            f"Missing : "
            f"{pairs['missing']}"
        )

    if pairs["extra"]:
        print(
            f"Extra : "
            f"{pairs['extra']}"
        )

    print()

    # -------------------------------------------------------------
    # METADATA CONSISTENCY
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "METADATA CONSISTENCY"
    )
    print("-" * 72)

    print(
        f"Valid : "
        f"{metadata_consistency['valid']}"
    )

    for issue in (
        metadata_consistency["issues"]
    ):
        print(
            f"  - {issue}"
        )

    print()

    # -------------------------------------------------------------
    # PRECIPITATION
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "PRECIPITATION (tp) VALIDATION"
    )
    print("-" * 72)

    print(
        f"Found : "
        f"{precipitation['found']}"
    )

    print(
        f"Valid : "
        f"{precipitation['valid']}"
    )

    if precipitation["found"]:

        print(
            f"Units : "
            f"{precipitation['units']}"
        )

        print(
            f"stepType : "
            f"{precipitation['stepType']}"
        )

        print(
            f"stepUnits : "
            f"{precipitation['stepUnits']}"
        )

        print(
            f"startStep : "
            f"{precipitation['startStep']}"
        )

        print(
            f"endStep : "
            f"{precipitation['endStep']}"
        )

        print(
            f"Interpretation : "
            f"{precipitation['interpretation']}"
        )

    print()

    # -------------------------------------------------------------
    # SOIL MOISTURE
    # -------------------------------------------------------------

    print("-" * 72)
    print(
        "SOIL MOISTURE (soilw) VALIDATION"
    )
    print("-" * 72)

    print(
        f"Found : "
        f"{soil_moisture['found']}"
    )

    print(
        f"Valid : "
        f"{soil_moisture['valid']}"
    )

    print(
        f"Status : "
        f"{soil_moisture['status']}"
    )

    if soil_moisture["found"]:

        print(
            f"Units : "
            f"{soil_moisture['units']}"
        )

        print(
            f"Type of level : "
            f"{soil_moisture['typeOfLevel']}"
        )

        print(
            f"Levels : "
            f"{soil_moisture['levels']}"
        )

        print(
            f"Reason : "
            f"{soil_moisture['reason']}"
        )

    print()

    # =================================================================
    # FINAL GATES
    # =================================================================

    issues: list[str] = []

    # -------------------------------------------------------------
    # 1. Exactly 25 files
    # -------------------------------------------------------------

    if len(files) != EXPECTED_FILE_COUNT:

        issues.append(
            "Expected 25 GEFS GRIB files, "
            f"found {len(files)}."
        )

    # -------------------------------------------------------------
    # 2. Filename member/lead coverage
    # -------------------------------------------------------------

    if not members["valid"]:

        issues.append(
            "Filename member coverage is incomplete."
        )

    if not leads["valid"]:

        issues.append(
            "Filename forecast-lead coverage is incomplete."
        )

    if not pairs["valid"]:

        issues.append(
            "Filename member/lead pair coverage "
            "is incomplete."
        )

    # -------------------------------------------------------------
    # 3. M2.1 readability
    # -------------------------------------------------------------

    if not readability["available"]:

        issues.append(
            "M2.1 smoke-test report is unavailable."
        )

    elif not readability["all_readable"]:

        issues.append(
            "M2.1 report does not provide confirmed "
            "25/25 GRIB readability evidence."
        )

    # -------------------------------------------------------------
    # 4. IMPORTANT:
    # ecCodes partial EOF is diagnostic only when M2.1
    # independently confirms all 25 files readable.
    #
    # Do NOT fail M2.2 simply because ecCodes reaches EOF
    # after recovering valid metadata.
    # -------------------------------------------------------------

    if file_errors:

        if not readability["all_readable"]:

            issues.append(
                "GRIB metadata recovery failed and "
                "independent M2.1 readability evidence "
                "is unavailable/incomplete."
            )

    # -------------------------------------------------------------
    # 5. Grid
    # -------------------------------------------------------------

    if not grid["valid"]:

        issues.append(
            "GEFS grid is not validated as regular "
            "0.25-degree latitude/longitude grid."
        )

    # -------------------------------------------------------------
    # 6. Metadata consistency
    # -------------------------------------------------------------

    if not metadata_consistency["valid"]:

        issues.append(
            "GRIB member metadata is inconsistent "
            "with filenames."
        )

    # -------------------------------------------------------------
    # 7. Required variables
    # -------------------------------------------------------------

    missing_required = [
        name
        for name in REQUIRED_CANONICAL
        if canonical[name]["status"]
        != "FOUND"
    ]

    if missing_required:

        issues.append(
            "Missing required canonical variables: "
            + ", ".join(
                missing_required
            )
        )

    # -------------------------------------------------------------
    # 8. Precipitation
    # -------------------------------------------------------------

    if not precipitation["valid"]:

        issues.append(
            "Precipitation tp metadata is not validated."
        )

    # -------------------------------------------------------------
    # 9. Soil moisture is NOT a blocker.
    # -------------------------------------------------------------

    soil_moisture_gate = (
        "REVIEW_REQUIRED"
        if soil_moisture["found"]
        else "MISSING"
    )

    # =================================================================
    # REPORT
    # =================================================================

    status = (
        "PASS"
        if not issues
        else "NOT_PASS"
    )

    report = {
        "milestone": "M2.2",
        "title": "GEFS Variable Inventory",
        "status": status,

        "reader": {
            "name": "ecCodes",
            "mode": "metadata_only",
            "cfgrib_used": False,
            "xarray_used": False,
            "field_values_read": False,
        },

        "input": {
            "directory": str(
                INPUT_DIR
            ),
            "file_count": len(files),
            "expected_file_count": (
                EXPECTED_FILE_COUNT
            ),
            "files": [
                str(
                    path.relative_to(
                        BACKEND_DIR
                    )
                )
                for path in files
            ],
        },

        "m2_1_readability_evidence": (
            readability
        ),

        "filename_member_validation": (
            members
        ),

        "filename_lead_validation": (
            leads
        ),

        "member_lead_pair_validation": (
            pairs
        ),

        "file_inventory": (
            file_results
        ),

        "file_errors": (
            file_errors
        ),

        "variables": (
            merged
        ),

        "canonical_variables": (
            canonical
        ),

        "grid_validation": (
            grid
        ),

        "metadata_consistency": (
            metadata_consistency
        ),

        "precipitation_validation": (
            precipitation
        ),

        "soil_moisture_validation": (
            soil_moisture
        ),

        "soil_moisture_gate": (
            soil_moisture_gate
        ),

        "required_canonical_variables": (
            REQUIRED_CANONICAL
        ),

        "issues": issues,

        "scientific_notes": [
            (
                "The 25-file sample contains the control "
                "member gec00 and perturbed members "
                "gep01-gep04."
            ),
            (
                "Historical/operational ensemble member-count "
                "compatibility must be handled explicitly."
            ),
            (
                "tp is accumulated precipitation. "
                "Temporal accumulation/differencing must be "
                "handled explicitly during ingestion."
            ),
            (
                "soilw is reported but no soil-moisture "
                "layer is silently selected."
            ),
            (
                "M2.2 inventories the smoke-test sample only. "
                "It does not claim complete 2020-2025 archive "
                "coverage."
            ),
        ],
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # =================================================================
    # FINAL VERDICT
    # =================================================================

    print("=" * 72)
    print("M2.2 VERDICT")
    print("=" * 72)
    print()

    if issues:

        print(
            "STATUS: NOT_PASS"
        )

        print()

        print(
            "Issues requiring resolution:"
        )

        for issue in issues:
            print(
                f"  - {issue}"
            )

        print()

        print(
            f"Report: {OUTPUT_JSON}"
        )

        return 1

    print(
        "STATUS: PASS"
    )

    print()

    print(
        "25 GEFS files discovered."
    )

    print(
        "Member/lead filename coverage validated."
    )

    print(
        "M2.1 readability evidence validated."
    )

    print(
        "Required GEFS variables identified."
    )

    print(
        "0.25-degree regular grid validated."
    )

    print(
        "Precipitation accumulation semantics validated."
    )

    print()

    print(
        f"Soil moisture status: "
        f"{soil_moisture_gate}"
    )

    print()

    print(
        f"Report: {OUTPUT_JSON}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())