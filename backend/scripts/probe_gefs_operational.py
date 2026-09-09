from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ============================================================
# Sanket-X M1 — GEFS Availability Audit
#
# Purpose:
#   1. Historical operational GEFS archive: AWS
#   2. Current operational GEFS: NOMADS
#
# This script ONLY probes availability.
# It does NOT download large GRIB files.
# ============================================================


AWS_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"

NOMADS_BASE = (
    "https://nomads.ncep.noaa.gov/"
    "pub/data/nccf/com/gens/prod"
)

# GEFSv12 became operational on 2020-09-23.
DEFAULT_DATES = [
    "2020-09-24",
    "2023-01-03",
    "2025-01-02",
    "2026-09-05",
]

CYCLES = ["00", "06", "12", "18"]

# First 5 members — compatible with current Sanket-X model.
MEMBERS_5 = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]

# Full operational GEFSv12 catalog.
MEMBERS_31 = ["gec00"] + [
    f"gep{i:02d}" for i in range(1, 31)
]

# NOAA GEFS 0.25 degree select-parameter product.
# NCEP inventory documents this product as pgrb2s.0p25.
PRODUCT = "pgrb2sp25"

# Representative forecast hours.
FORECAST_HOURS = [
    "f000",
    "f024",
    "f048",
    "f072",
    "f096",
    "f120",
]

TIMEOUT = 20


# ============================================================
# HTTP helpers
# ============================================================

def probe_url(url: str) -> dict:
    """
    Lightweight HTTP availability probe.

    HEAD is attempted first.
    If HEAD is rejected, a tiny Range GET is attempted.
    """

    result = {
        "url": url,
        "available": False,
        "status": None,
        "method": None,
        "error": None,
    }

    headers = {
        "User-Agent": "Sanket-X-M1-GEFS-Probe/1.0"
    }

    # -----------------------------
    # HEAD
    # -----------------------------

    try:
        request = Request(
            url,
            method="HEAD",
            headers=headers,
        )

        with urlopen(request, timeout=TIMEOUT) as response:
            result["status"] = response.status
            result["available"] = 200 <= response.status < 400
            result["method"] = "HEAD"
            return result

    except HTTPError as exc:
        result["status"] = exc.code

        # Some S3/NOMADS endpoints reject HEAD.
        if exc.code not in (400, 403, 405, 501):
            result["error"] = str(exc)
            return result

    except (URLError, TimeoutError, OSError) as exc:
        result["error"] = str(exc)

    # -----------------------------
    # Range GET fallback
    # -----------------------------

    try:
        headers["Range"] = "bytes=0-1023"

        request = Request(
            url,
            headers=headers,
        )

        with urlopen(request, timeout=TIMEOUT) as response:
            result["status"] = response.status
            result["available"] = response.status in (200, 206)
            result["method"] = "RANGE_GET"
            return result

    except HTTPError as exc:
        result["status"] = exc.code
        result["error"] = str(exc)

    except (URLError, TimeoutError, OSError) as exc:
        result["error"] = str(exc)

    return result


def normalise_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date: {value}. Use YYYY-MM-DD."
        ) from exc

    return parsed.strftime("%Y%m%d")


# ============================================================
# AWS historical archive
# ============================================================

def aws_object_url(
    yyyymmdd: str,
    cycle: str,
    member: str,
    forecast_hour: str,
) -> str:
    """
    GEFS PDS object layout.

    We use the operational-style GEFS directory structure.
    """

    return (
        f"{AWS_BUCKET}/"
        f"gefs.{yyyymmdd}/"
        f"{cycle}/"
        f"atmos/"
        f"{PRODUCT}/"
        f"{member}.t{cycle}z.pgrb2s.0p25.{forecast_hour}"
    )


def probe_aws_cycle(
    yyyymmdd: str,
    cycle: str,
) -> dict:

    print()
    print("-" * 72)
    print(f"AWS historical archive: {yyyymmdd} {cycle}Z")
    print("-" * 72)

    result = {
        "source": "AWS_GEFS_PDS",
        "date": yyyymmdd,
        "cycle": cycle,
        "members_5": {},
        "members_31": {},
        "forecast_hours": {},
    }

    # --------------------------------------------------------
    # Probe first 5 compatible members
    # --------------------------------------------------------

    available_5 = []

    print("5-member compatibility check:")

    for member in MEMBERS_5:

        url = aws_object_url(
            yyyymmdd,
            cycle,
            member,
            "f024",
        )

        probe = probe_url(url)

        result["members_5"][member] = probe

        if probe["available"]:
            available_5.append(member)
            print(f"  {member}: AVAILABLE")
        else:
            print(f"  {member}: NOT CONFIRMED")

    # --------------------------------------------------------
    # Probe full 31-member catalog
    #
    # Only f024 is checked, so this remains lightweight.
    # --------------------------------------------------------

    available_31 = []

    print()
    print("31-member operational catalog check:")

    for member in MEMBERS_31:

        url = aws_object_url(
            yyyymmdd,
            cycle,
            member,
            "f024",
        )

        probe = probe_url(url)

        result["members_31"][member] = probe

        if probe["available"]:
            available_31.append(member)

    print(
        f"  Available members: "
        f"{len(available_31)}/31"
    )

    # --------------------------------------------------------
    # Forecast-hour check
    # --------------------------------------------------------

    print()
    print("Representative forecast-hour check:")

    for fh in FORECAST_HOURS:

        url = aws_object_url(
            yyyymmdd,
            cycle,
            "gec00",
            fh,
        )

        probe = probe_url(url)

        result["forecast_hours"][fh] = probe

        print(
            f"  gec00 {fh}: "
            f"{'AVAILABLE' if probe['available'] else 'NOT CONFIRMED'}"
        )

    result["available_5_count"] = len(available_5)
    result["available_31_count"] = len(available_31)

    result["five_member_compatible"] = (
        len(available_5) == 5
    )

    result["full_31_member_confirmed"] = (
        len(available_31) == 31
    )

    return result


# ============================================================
# Current NOMADS
# ============================================================

def nomads_object_url(
    yyyymmdd: str,
    cycle: str,
    member: str,
    forecast_hour: str,
) -> str:

    return (
        f"{NOMADS_BASE}/"
        f"gefs.{yyyymmdd}/"
        f"{cycle}/"
        f"atmos/"
        f"{PRODUCT}/"
        f"{member}.t{cycle}z.pgrb2s.0p25.{forecast_hour}"
    )


def probe_nomads_cycle(
    yyyymmdd: str,
    cycle: str,
) -> dict:

    print()
    print("-" * 72)
    print(f"NOMADS current/live source: {yyyymmdd} {cycle}Z")
    print("-" * 72)

    result = {
        "source": "NOMADS",
        "date": yyyymmdd,
        "cycle": cycle,
        "members_5": {},
        "members_31": {},
        "forecast_hours": {},
    }

    # --------------------------------------------------------
    # 5-member compatibility
    # --------------------------------------------------------

    available_5 = []

    print("5-member compatibility check:")

    for member in MEMBERS_5:

        url = nomads_object_url(
            yyyymmdd,
            cycle,
            member,
            "f024",
        )

        probe = probe_url(url)

        result["members_5"][member] = probe

        if probe["available"]:
            available_5.append(member)
            print(f"  {member}: AVAILABLE")
        else:
            print(f"  {member}: NOT CONFIRMED")

    # --------------------------------------------------------
    # Full 31 members
    # --------------------------------------------------------

    available_31 = []

    print()
    print("31-member operational catalog check:")

    for member in MEMBERS_31:

        url = nomads_object_url(
            yyyymmdd,
            cycle,
            member,
            "f024",
        )

        probe = probe_url(url)

        result["members_31"][member] = probe

        if probe["available"]:
            available_31.append(member)

    print(
        f"  Available members: "
        f"{len(available_31)}/31"
    )

    # --------------------------------------------------------
    # Forecast hours
    # --------------------------------------------------------

    print()
    print("Representative forecast-hour check:")

    for fh in FORECAST_HOURS:

        url = nomads_object_url(
            yyyymmdd,
            cycle,
            "gec00",
            fh,
        )

        probe = probe_url(url)

        result["forecast_hours"][fh] = probe

        print(
            f"  gec00 {fh}: "
            f"{'AVAILABLE' if probe['available'] else 'NOT CONFIRMED'}"
        )

    result["available_5_count"] = len(available_5)
    result["available_31_count"] = len(available_31)

    result["five_member_compatible"] = (
        len(available_5) == 5
    )

    result["full_31_member_confirmed"] = (
        len(available_31) == 31
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Sanket-X M1 GEFS availability audit. "
            "No large data download."
        )
    )

    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="Dates in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--cycles",
        nargs="+",
        default=CYCLES,
        choices=CYCLES,
    )

    args = parser.parse_args()

    dates = [
        normalise_date(x)
        for x in args.dates
    ]

    print("=" * 72)
    print("Sanket-X M1 — GEFS Availability Audit")
    print("=" * 72)

    print()
    print("Historical source : AWS noaa-gefs-pds")
    print("Current source    : NOAA NOMADS")
    print("Grid              : 0.25 degree")
    print("Product           : pgrb2s.0p25")
    print()
    print("5-member contract:")
    print(MEMBERS_5)

    print()
    print("Full operational catalog:")
    print("gec00 + gep01 ... gep30")

    print()
    print("Dates:")
    for d in dates:
        print(f"  {d}")

    print()
    print("Cycles:")
    for c in args.cycles:
        print(f"  {c}Z")

    print()
    print(
        "IMPORTANT: This script performs lightweight HTTP "
        "availability checks only."
    )
    print(
        "No large GRIB archive will be downloaded."
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    results = []

    for yyyymmdd in dates:

        for cycle in args.cycles:

            # 2026 is current/live → NOMADS.
            if yyyymmdd.startswith("2026"):

                result = probe_nomads_cycle(
                    yyyymmdd,
                    cycle,
                )

            else:

                result = probe_aws_cycle(
                    yyyymmdd,
                    cycle,
                )

            results.append(result)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print()
    print("#" * 72)
    print("M1 GEFS AVAILABILITY SUMMARY")
    print("#" * 72)

    historical = [
        x for x in results
        if x["source"] == "AWS_GEFS_PDS"
    ]

    current = [
        x for x in results
        if x["source"] == "NOMADS"
    ]

    historical_5_pass = [
        x for x in historical
        if x["five_member_compatible"]
    ]

    historical_31_pass = [
        x for x in historical
        if x["full_31_member_confirmed"]
    ]

    current_5_pass = [
        x for x in current
        if x["five_member_compatible"]
    ]

    current_31_pass = [
        x for x in current
        if x["full_31_member_confirmed"]
    ]

    print()
    print("Historical AWS:")
    print(
        f"  Cycles tested              : {len(historical)}"
    )
    print(
        f"  5-member compatible        : "
        f"{len(historical_5_pass)}"
    )
    print(
        f"  Full 31-member confirmed   : "
        f"{len(historical_31_pass)}"
    )

    print()
    print("Current NOMADS:")
    print(
        f"  Cycles tested              : {len(current)}"
    )
    print(
        f"  5-member compatible        : "
        f"{len(current_5_pass)}"
    )
    print(
        f"  Full 31-member confirmed   : "
        f"{len(current_31_pass)}"
    )

    print()
    print("Detailed results:")

    for result in results:

        print(
            f"  {result['source']:14s} "
            f"{result['date']} "
            f"{result['cycle']}Z "
            f"| 5-member "
            f"{result['available_5_count']}/5 "
            f"| 31-member "
            f"{result['available_31_count']}/31"
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    output_dir = (
        Path("data")
        / "analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "gefs_operational_availability_m1.json"
    )

    report = {
        "stage": "M1",
        "purpose": (
            "GEFS operational availability audit"
        ),
        "historical_source": (
            "NOAA GEFS Public Dataset on AWS"
        ),
        "current_source": (
            "NOAA NOMADS"
        ),
        "product": "pgrb2s.0p25",
        "grid_resolution": "0.25 degree",
        "five_member_contract": MEMBERS_5,
        "operational_member_count": 31,
        "dates": dates,
        "cycles": args.cycles,
        "large_download_performed": False,
        "summary": {
            "historical_cycles": len(historical),
            "historical_5_member_pass": len(
                historical_5_pass
            ),
            "historical_31_member_pass": len(
                historical_31_pass
            ),
            "current_cycles": len(current),
            "current_5_member_pass": len(
                current_5_pass
            ),
            "current_31_member_pass": len(
                current_31_pass
            ),
        },
        "results": results,
    }

    output_file.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Report saved: {output_file}"
    )

    print()
    print("=" * 72)
    print("M1 PROBE COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()