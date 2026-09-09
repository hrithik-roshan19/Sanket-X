from __future__ import annotations

import json
import statistics
from datetime import date
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]

INPUT = (
    BACKEND_DIR
    / "data"
    / "analysis"
    / "gefs_archive_m2_3_availability.json"
)

OUTPUT = (
    BACKEND_DIR
    / "data"
    / "analysis"
    / "gefs_archive_m2_4_plan.json"
)


START_DATE = date(2020, 9, 23)
END_DATE = date(2025, 12, 31)

MEMBERS = [
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
]

LEADS = [
    24,
    48,
    72,
    96,
    120,
]

CYCLES = [
    0,
    6,
    12,
    18,
]


def scenario(cycles: list[int], days: int, sizes: list[int]) -> dict:

    file_count = (
        days
        * len(cycles)
        * len(MEMBERS)
        * len(LEADS)
    )

    mean_bytes = statistics.mean(sizes)
    median_bytes = statistics.median(sizes)

    return {
        "cycles": cycles,
        "days": days,
        "files": file_count,
        "estimated_mean_gb": (
            file_count * mean_bytes / 1_000_000_000
        ),
        "estimated_median_gb": (
            file_count * median_bytes / 1_000_000_000
        ),
    }


def main() -> int:

    print("=" * 72)
    print("SANKET-X M2.4 — GEFS ARCHIVE PLAN")
    print("=" * 72)
    print()

    if not INPUT.exists():
        raise SystemExit(
            f"M2.3 report not found:\n{INPUT}"
        )

    report = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    rows = report.get(
        "results",
        []
    )

    sizes = []

    for row in rows:

        if not row.get("available"):
            continue

        value = row.get(
            "content_length"
        )

        if value in (
            None,
            "",
            0,
        ):
            continue

        try:
            sizes.append(
                int(value)
            )
        except (TypeError, ValueError):
            continue

    if not sizes:
        raise SystemExit(
            "No valid Content-Length values "
            "found in M2.3 report."
        )

    days = (
        END_DATE - START_DATE
    ).days + 1

    mean_bytes = statistics.mean(
        sizes
    )

    median_bytes = statistics.median(
        sizes
    )

    print(
        f"Training window : "
        f"{START_DATE} → {END_DATE}"
    )

    print(
        f"Days            : {days}"
    )

    print(
        f"Members         : {MEMBERS}"
    )

    print(
        f"Leads           : {LEADS}"
    )

    print()

    print("-" * 72)
    print("FILE-SIZE SAMPLE")
    print("-" * 72)

    print(
        f"Objects sampled : {len(sizes)}"
    )

    print(
        f"Mean size       : "
        f"{mean_bytes / 1_000_000:.2f} MB"
    )

    print(
        f"Median size     : "
        f"{median_bytes / 1_000_000:.2f} MB"
    )

    print(
        f"Minimum         : "
        f"{min(sizes) / 1_000_000:.2f} MB"
    )

    print(
        f"Maximum         : "
        f"{max(sizes) / 1_000_000:.2f} MB"
    )

    print()

    plan_00z = scenario(
        [0],
        days,
        sizes,
    )

    plan_all_cycles = scenario(
        CYCLES,
        days,
        sizes,
    )

    print("-" * 72)
    print("00Z-ONLY TRAINING PLAN")
    print("-" * 72)

    print(
        f"Files : "
        f"{plan_00z['files']:,}"
    )

    print(
        f"Estimated storage "
        f"(mean) : "
        f"{plan_00z['estimated_mean_gb']:.2f} GB"
    )

    print(
        f"Estimated storage "
        f"(median) : "
        f"{plan_00z['estimated_median_gb']:.2f} GB"
    )

    print()

    print("-" * 72)
    print("ALL-FOUR-CYCLE PLAN")
    print("-" * 72)

    print(
        f"Files : "
        f"{plan_all_cycles['files']:,}"
    )

    print(
        f"Estimated storage "
        f"(mean) : "
        f"{plan_all_cycles['estimated_mean_gb']:.2f} GB"
    )

    print(
        f"Estimated storage "
        f"(median) : "
        f"{plan_all_cycles['estimated_median_gb']:.2f} GB"
    )

    print()

    report_out = {
        "milestone": "M2.4",

        "status": "PASS",

        "training_contract": {
            "start_date": START_DATE.isoformat(),
            "end_date": END_DATE.isoformat(),
            "cycles": [0],
            "members": MEMBERS,
            "leads_hours": LEADS,

            "reason": (
                "00Z-only keeps the operational training "
                "contract closer to the one-cycle-per-day "
                "GEFSv12 reforecast design and avoids "
                "unnecessary four-cycle duplication."
            ),
        },

        "archive_window_note": (
            "GEFSv12 became operational on "
            "2020-09-23. January through "
            "2020-09-22 is therefore excluded "
            "from the GEFSv12 operational "
            "training window."
        ),

        "source_size_sample": {
            "objects": len(sizes),
            "mean_bytes": mean_bytes,
            "median_bytes": median_bytes,
            "min_bytes": min(sizes),
            "max_bytes": max(sizes),
        },

        "scenarios": {
            "00z_only": plan_00z,
            "all_four_cycles": plan_all_cycles,
        },

        "download_policy": {
            "resume": True,
            "retry": True,
            "checksum": True,
            "partial_files": ".part",
            "never_overwrite_valid_file": True,
            "download_order": (
                "date/cycle/member/lead"
            ),
        },

        "scientific_guardrails": [
            (
                "Historical GEFSv12 reforecast uses "
                "a different ensemble design from "
                "the operational 31-member system."
            ),
            (
                "The current production training "
                "contract remains the compatible "
                "5-member subset."
            ),
            (
                "This planning stage performs no "
                "large archive download."
            ),
            (
                "Storage estimates are based on "
                "Content-Length values from the "
                "M2.3 representative availability "
                "sample."
            ),
        ],
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report_out,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("M2.4 VERDICT")
    print("=" * 72)
    print()
    print("STATUS: PASS")
    print()
    print(
        "Archive window, training contract, "
        "file count and storage scenarios calculated."
    )
    print()
    print(
        f"Report: {OUTPUT}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )