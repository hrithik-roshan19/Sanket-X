from __future__ import annotations

import json
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

OUTPUT = (
    BACKEND_DIR
    / "data"
    / "analysis"
    / "gefs_archive_m2_3_availability.json"
)


# ============================================================
# NOAA GEFS AWS
# ============================================================

AWS_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"

# Verified GEFS 0.25-degree product directory.
PRODUCT_DIR = "pgrb2sp25"

# Filename product identifier.
FILENAME_PRODUCT = "pgrb2s.0p25"


# ============================================================
# COMPATIBLE 5-MEMBER CONTRACT
# ============================================================

MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]


# ============================================================
# LEADS USED BY CURRENT SANKET-X CONTRACT
# ============================================================

LEADS = [
    24,
    48,
    72,
    96,
    120,
]


# ============================================================
# GEFS OPERATIONAL CYCLES
# ============================================================

CYCLES = [
    0,
    6,
    12,
    18,
]


# ============================================================
# REPRESENTATIVE DATES
#
# One date per year.
#
# These are availability probes only.
# They are NOT being claimed as complete
# year-wide archive coverage.
# ============================================================

SAMPLE_DATES = {
    2020: [
        "20200924",
    ],
    2021: [
        "20210105",
    ],
    2022: [
        "20220712",
    ],
    2023: [
        "20230103",
    ],
    2024: [
        "20240611",
    ],
    2025: [
        "20250102",
    ],
}


# ============================================================
# NETWORK
# ============================================================

TIMEOUT_SECONDS = 30

MAX_WORKERS = 8


# ============================================================
# URL BUILDER
# ============================================================

def build_url(
    day: str,
    cycle: int,
    member: str,
    lead: int,
) -> str:

    return (
        f"{AWS_BASE}/"
        f"gefs.{day}/"
        f"{cycle:02d}/"
        f"atmos/"
        f"{PRODUCT_DIR}/"
        f"{member}.t{cycle:02d}z."
        f"{FILENAME_PRODUCT}."
        f"f{lead:03d}"
    )


# ============================================================
# SINGLE OBJECT PROBE
# ============================================================

def probe_url(
    url: str,
) -> dict:

    try:

        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={
                "User-Agent": (
                    "Sanket-X-M2.3/1.0"
                )
            },
        )

        context = (
            ssl.create_default_context()
        )

        with urllib.request.urlopen(
            request,
            timeout=TIMEOUT_SECONDS,
            context=context,
        ) as response:

            status = response.status

            return {
                "url": url,
                "status": status,
                "available": (
                    status == 200
                ),
                "content_length": (
                    response.headers.get(
                        "Content-Length"
                    )
                ),
                "error": None,
            }

    except Exception as exc:

        return {
            "url": url,
            "status": None,
            "available": False,
            "content_length": None,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        }


# ============================================================
# BUILD PROBE CASES
# ============================================================

def build_cases() -> list[dict]:

    cases = []

    for year, dates in (
        SAMPLE_DATES.items()
    ):

        for day in dates:

            for cycle in CYCLES:

                for member in MEMBERS:

                    for lead in LEADS:

                        cases.append(
                            {
                                "year": year,
                                "date": day,
                                "cycle": cycle,
                                "member": member,
                                "lead": lead,
                                "url": build_url(
                                    day,
                                    cycle,
                                    member,
                                    lead,
                                ),
                            }
                        )

    return cases


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 72)
    print(
        "SANKET-X M2.3 — GEFS ARCHIVE AVAILABILITY"
    )
    print("=" * 72)

    print()

    print(
        "Source       : NOAA GEFS AWS Open Data"
    )

    print(
        f"Product dir  : {PRODUCT_DIR}"
    )

    print(
        f"File product : {FILENAME_PRODUCT}"
    )

    print(
        f"Members      : {MEMBERS}"
    )

    print(
        f"Cycles       : {CYCLES}"
    )

    print(
        f"Leads        : {LEADS}"
    )

    print()

    cases = build_cases()

    expected_files = len(cases)

    files_per_cycle = (
        len(MEMBERS)
        * len(LEADS)
    )

    print(
        f"Files per cycle : "
        f"{files_per_cycle}"
    )

    print(
        f"Total probes    : "
        f"{expected_files}"
    )

    print()

    # ========================================================
    # PARALLEL PROBES
    # ========================================================

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                probe_url,
                case["url"],
            ): case
            for case in cases
        }

        completed = 0

        for future in as_completed(
            futures
        ):

            case = futures[future]

            result = future.result()

            results.append(
                {
                    **case,
                    **result,
                }
            )

            completed += 1

            if (
                completed % 25 == 0
                or completed == expected_files
            ):

                print(
                    f"Probed "
                    f"{completed}/"
                    f"{expected_files}"
                )

    # ========================================================
    # SORT RESULTS
    # ========================================================

    results.sort(
        key=lambda row: (
            row["year"],
            row["date"],
            row["cycle"],
            row["member"],
            row["lead"],
        )
    )

    # ========================================================
    # YEAR SUMMARY
    # ========================================================

    year_summary = {}

    for year in SAMPLE_DATES:

        rows = [
            row
            for row in results
            if row["year"] == year
        ]

        available = [
            row
            for row in rows
            if row["available"]
        ]

        expected = len(rows)

        available_count = len(
            available
        )

        missing_count = (
            expected
            - available_count
        )

        rate = (
            available_count / expected
            if expected
            else 0.0
        )

        year_summary[
            str(year)
        ] = {
            "dates": SAMPLE_DATES[
                year
            ],
            "expected_files": expected,
            "available_files": (
                available_count
            ),
            "missing_files": (
                missing_count
            ),
            "availability_rate": rate,
            "complete": (
                missing_count == 0
            ),
        }

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    available_count = sum(
        1
        for row in results
        if row["available"]
    )

    missing_count = (
        expected_files
        - available_count
    )

    availability_rate = (
        available_count
        / expected_files
        if expected_files
        else 0.0
    )

    # ========================================================
    # MISSING FILES
    # ========================================================

    missing = [
        {
            "year": row["year"],
            "date": row["date"],
            "cycle": row["cycle"],
            "member": row["member"],
            "lead": row["lead"],
            "url": row["url"],
            "error": row["error"],
        }
        for row in results
        if not row["available"]
    ]

    # ========================================================
    # MEMBER COVERAGE
    # ========================================================

    member_coverage = {}

    for member in MEMBERS:

        member_rows = [
            row
            for row in results
            if row["member"] == member
        ]

        member_available = sum(
            row["available"]
            for row in member_rows
        )

        member_expected = len(
            member_rows
        )

        member_coverage[
            member
        ] = {
            "expected": member_expected,
            "available": member_available,
            "missing": (
                member_expected
                - member_available
            ),
            "complete": (
                member_available
                == member_expected
            ),
        }

    # ========================================================
    # LEAD COVERAGE
    # ========================================================

    lead_coverage = {}

    for lead in LEADS:

        lead_rows = [
            row
            for row in results
            if row["lead"] == lead
        ]

        lead_available = sum(
            row["available"]
            for row in lead_rows
        )

        lead_expected = len(
            lead_rows
        )

        lead_coverage[
            str(lead)
        ] = {
            "expected": lead_expected,
            "available": lead_available,
            "missing": (
                lead_expected
                - lead_available
            ),
            "complete": (
                lead_available
                == lead_expected
            ),
        }

    # ========================================================
    # FINAL STATUS
    #
    # PASS means:
    # Every representative object tested
    # was available.
    #
    # It DOES NOT mean complete 2020-2025
    # archive coverage.
    # ========================================================

    status = (
        "PASS"
        if available_count
        == expected_files
        else "NOT_PASS"
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "milestone": "M2.3",

        "title": (
            "GEFS Operational Archive "
            "Availability Audit"
        ),

        "status": status,

        "source": {
            "provider": "NOAA",
            "dataset": "GEFS",
            "bucket": "noaa-gefs-pds",
            "base_url": AWS_BASE,
            "product_directory": PRODUCT_DIR,
            "filename_product": FILENAME_PRODUCT,
        },

        "sample_design": {
            "years": sorted(
                SAMPLE_DATES.keys()
            ),
            "dates": SAMPLE_DATES,
            "cycles": CYCLES,
            "members": MEMBERS,
            "leads": LEADS,
            "files_per_cycle": (
                files_per_cycle
            ),
        },

        "summary": {
            "expected_files": (
                expected_files
            ),
            "available_files": (
                available_count
            ),
            "missing_files": (
                missing_count
            ),
            "availability_rate": (
                availability_rate
            ),
        },

        "year_summary": year_summary,

        "member_coverage": (
            member_coverage
        ),

        "lead_coverage": (
            lead_coverage
        ),

        "missing": missing,

        "results": results,

        "interpretation": (
            "This is a representative "
            "availability audit. A PASS does "
            "not establish complete daily "
            "archive coverage for 2020-2025."
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print("=" * 72)
    print(
        "YEAR-BY-YEAR AVAILABILITY"
    )
    print("=" * 72)

    for year, info in (
        year_summary.items()
    ):

        print(
            f"{year}: "
            f"{info['available_files']}/"
            f"{info['expected_files']} "
            f"("
            f"{info['availability_rate']:.1%}"
            f")"
        )

    print()

    print("=" * 72)
    print("M2.3 VERDICT")
    print("=" * 72)

    print()

    print(
        f"Expected files : "
        f"{expected_files}"
    )

    print(
        f"Available      : "
        f"{available_count}"
    )

    print(
        f"Missing        : "
        f"{missing_count}"
    )

    print(
        f"Availability   : "
        f"{availability_rate:.2%}"
    )

    print()

    print(
        f"STATUS: {status}"
    )

    print()

    print(
        f"Report: {OUTPUT}"
    )

    return (
        0
        if status == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )