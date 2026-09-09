from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


S3_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"

DEFAULT_DATE = "20200924"
DEFAULT_CYCLE = "00"

MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]

FORECAST_HOURS = [
    "f024",
    "f048",
    "f072",
    "f096",
    "f120",
]

TIMEOUT = 30


def s3_list(prefix: str, max_keys: int = 100) -> dict:
    """
    List a SMALL number of S3 objects under a narrow prefix.

    Pagination is intentionally not followed automatically.
    We only need enough objects to discover the real naming pattern.
    """

    url = (
        f"{S3_BUCKET}/"
        f"?list-type=2"
        f"&prefix={quote(prefix, safe='')}"
        f"&max-keys={max_keys}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Sanket-X-M2-GEFS-Discovery/1.0"
        },
    )

    try:
        with urlopen(
            request,
            timeout=TIMEOUT,
        ) as response:

            body = response.read()

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
    ) as exc:

        return {
            "success": False,
            "keys": [],
            "truncated": False,
            "error": str(exc),
        }

    except Exception as exc:

        return {
            "success": False,
            "keys": [],
            "truncated": False,
            "error": repr(exc),
        }

    try:

        root = ET.fromstring(body)

    except ET.ParseError as exc:

        return {
            "success": False,
            "keys": [],
            "truncated": False,
            "error": f"S3 XML parse error: {exc}",
        }

    namespace = {
        "s3": "http://s3.amazonaws.com/doc/2006-03-01/"
    }

    keys = []

    for node in root.findall(
        "s3:Contents/s3:Key",
        namespace,
    ):

        if node.text:
            keys.append(node.text)

    truncated_node = root.find(
        "s3:IsTruncated",
        namespace,
    )

    truncated = (
        truncated_node is not None
        and truncated_node.text == "true"
    )

    return {
        "success": True,
        "keys": keys,
        "truncated": truncated,
        "error": None,
    }


def discover_exact(
    yyyymmdd: str,
    cycle: str,
) -> dict:

    print()
    print("=" * 72)
    print("GEFS S3 EXACT OBJECT DISCOVERY")
    print("=" * 72)

    print()
    print(f"Date  : {yyyymmdd}")
    print(f"Cycle : {cycle}Z")
    print(f"Bucket: {S3_BUCKET}")

    results = []

    # --------------------------------------------------------
    # Narrow prefixes.
    #
    # We deliberately search by member/product instead of
    # listing the entire cycle directory.
    # --------------------------------------------------------

    prefixes = [
        f"gefs.{yyyymmdd}/{cycle}/atmos/",
        f"gefs.{yyyymmdd}/{cycle}/",
    ]

    discovered = []

    for prefix in prefixes:

        print()
        print(f"Trying prefix:")
        print(f"  {prefix}")

        response = s3_list(
            prefix,
            max_keys=100,
        )

        if not response["success"]:

            print(
                f"  ERROR: {response['error']}"
            )

            results.append(
                {
                    "prefix": prefix,
                    **response,
                }
            )

            continue

        print(
            f"  Objects returned: "
            f"{len(response['keys'])}"
        )

        print(
            f"  Truncated: "
            f"{response['truncated']}"
        )

        discovered.extend(
            response["keys"]
        )

        results.append(
            {
                "prefix": prefix,
                **response,
            }
        )

        if response["keys"]:
            break

    # Remove duplicates.
    discovered = list(
        dict.fromkeys(discovered)
    )

    # --------------------------------------------------------
    # Show only useful GEFS keys.
    # --------------------------------------------------------

    useful = []

    for key in discovered:

        lower = key.lower()

        if (
            "gec00" in lower
            or "gep01" in lower
            or "pgrb" in lower
            or "0p25" in lower
        ):
            useful.append(key)

    print()
    print("-" * 72)
    print("DISCOVERED GEFS KEYS")
    print("-" * 72)

    if useful:

        for key in useful[:100]:
            print(
                f"  {key}"
            )

    else:

        print(
            "  No matching GEFS object keys found."
        )

    # --------------------------------------------------------
    # Determine actual patterns.
    # --------------------------------------------------------

    member_matches = {
        member: []
        for member in MEMBERS
    }

    forecast_matches = {
        fh: []
        for fh in FORECAST_HOURS
    }

    for key in discovered:

        lower = key.lower()

        for member in MEMBERS:

            if member.lower() in lower:

                member_matches[
                    member
                ].append(key)

        for fh in FORECAST_HOURS:

            if fh in lower:

                forecast_matches[
                    fh
                ].append(key)

    print()
    print("-" * 72)
    print("MEMBER DISCOVERY")
    print("-" * 72)

    for member in MEMBERS:

        print(
            f"  {member}: "
            f"{len(member_matches[member])} matches"
        )

    print()
    print("-" * 72)
    print("FORECAST-HOUR DISCOVERY")
    print("-" * 72)

    for fh in FORECAST_HOURS:

        print(
            f"  {fh}: "
            f"{len(forecast_matches[fh])} matches"
        )

    # --------------------------------------------------------
    # Try direct candidate patterns.
    #
    # These are checked individually and do NOT download files.
    # --------------------------------------------------------

    candidate_patterns = [
        "{member}.t{cycle}z.pgrb2s.0p25.{fh}",
        "{member}.t{cycle}z.pgrb2a.0p25.{fh}",
        "{member}.t{cycle}z.pgrb2sp25.{fh}",
    ]

    direct_checks = []

    print()
    print("-" * 72)
    print("DIRECT OBJECT CHECKS")
    print("-" * 72)

    for member in MEMBERS:

        for fh in FORECAST_HOURS:

            for pattern in candidate_patterns:

                filename = pattern.format(
                    member=member,
                    cycle=cycle,
                    fh=fh,
                )

                possible_paths = [
                    f"gefs.{yyyymmdd}/{cycle}/"
                    f"atmos/pgrb2sp25/{filename}",

                    f"gefs.{yyyymmdd}/{cycle}/"
                    f"atmos/pgrb2s.0p25/{filename}",

                    f"gefs.{yyyymmdd}/{cycle}/"
                    f"atmos/pgrb2ap5/{filename}",

                    f"gefs.{yyyymmdd}/{cycle}/"
                    f"atmos/{filename}",
                ]

                for key in possible_paths:

                    url = (
                        f"{S3_BUCKET}/"
                        f"{quote(key, safe='/')}"
                    )

                    # HEAD only.
                    request = Request(
                        url,
                        method="HEAD",
                        headers={
                            "User-Agent":
                            "Sanket-X-M2-GEFS-Discovery/1.0"
                        },
                    )

                    available = False
                    status = None
                    error = None

                    try:

                        with urlopen(
                            request,
                            timeout=TIMEOUT,
                        ) as response:

                            status = response.status

                            available = (
                                200
                                <= response.status
                                < 400
                            )

                    except HTTPError as exc:

                        status = exc.code

                    except (
                        URLError,
                        TimeoutError,
                        OSError,
                    ) as exc:

                        error = str(exc)

                    direct_checks.append(
                        {
                            "member": member,
                            "forecast_hour": fh,
                            "key": key,
                            "url": url,
                            "available": available,
                            "status": status,
                            "error": error,
                        }
                    )

                    if available:

                        print(
                            f"  FOUND: "
                            f"{member} "
                            f"{fh}"
                        )

                        print(
                            f"         {key}"
                        )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    found = [
        x
        for x in direct_checks
        if x["available"]
    ]

    print()
    print("=" * 72)
    print("DISCOVERY SUMMARY")
    print("=" * 72)

    print()
    print(
        f"Listed objects discovered: "
        f"{len(discovered)}"
    )

    print(
        f"Direct candidate matches: "
        f"{len(found)}"
    )

    if found:

        print()
        print(
            "At least one exact object path "
            "has been confirmed."
        )

    else:

        print()
        print(
            "No candidate object path confirmed."
        )

    return {
        "stage": "M2.1",
        "purpose": (
            "Discover exact GEFS S3 object paths"
        ),
        "bucket": S3_BUCKET,
        "date": yyyymmdd,
        "cycle": cycle,
        "members": MEMBERS,
        "forecast_hours": FORECAST_HOURS,
        "listed_prefix_results": results,
        "discovered_keys": discovered,
        "useful_keys": useful,
        "member_matches": member_matches,
        "forecast_matches": forecast_matches,
        "direct_checks": direct_checks,
        "confirmed_objects": found,
        "confirmed_object_count": len(found),
        "large_download_performed": False,
    }


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Discover actual NOAA GEFS S3 object paths "
            "without downloading GRIB files."
        )
    )

    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="YYYYMMDD",
    )

    parser.add_argument(
        "--cycle",
        default=DEFAULT_CYCLE,
        choices=[
            "00",
            "06",
            "12",
            "18",
        ],
    )

    args = parser.parse_args()

    try:

        parsed = datetime.strptime(
            args.date,
            "%Y%m%d",
        )

        yyyymmdd = parsed.strftime(
            "%Y%m%d"
        )

    except ValueError:

        print(
            f"Invalid date: {args.date}"
        )

        print(
            "Use YYYYMMDD."
        )

        return 1

    report = discover_exact(
        yyyymmdd,
        args.cycle,
    )

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
        / "gefs_archive_discovery_m2.json"
    )

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
    print(
        "No large GRIB file was downloaded."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )