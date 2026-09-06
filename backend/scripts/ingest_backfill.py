"""Load backfilled reforecast years and their ERA5 labels into the canonical store.

The historical training base was 17 initialisation dates from 2019, which left the
held-out test split at 3 cycles - so "ROC-AUC 0.756" was measured on three weather
situations in one year. `fetch_gefs_reforecast_sample.py --years 2010-2019` and
`fetch_era5_observations.py` (one year per run) produce the files; this puts them in the
store so a retrain can use them.

Mappings are imported from `app.live.orchestrator` rather than re-derived, deliberately.
The schema mapper proposes `pressure_hpa` for BOTH `mslp_hpa` and `psfc_hpa`, and the
production decision - MSLP is canonical pressure, surface pressure is unmapped - lives in
FORECAST_MAPPINGS. Re-deriving it here would risk backfilled years landing in a different
canonical shape from the live feed, which is the one thing that must not happen.

Observations are ingested as `final`: ERA5 is the reanalysis the models verify against,
not a near-real-time product, so nothing here is provisional.

Usage
-----
    python -m scripts.ingest_backfill --years 2010-2018
    python -m scripts.ingest_backfill --years 2010-2019 --dry-run   # report, ingest nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
SAMPLES = BACKEND_DIR / "data" / "samples"


def parse_years(spec: str) -> list:
    out: set = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", required=True, metavar="SPEC",
                    help="2010-2018, or 2011,2014")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be ingested and stop")
    ap.add_argument("--skip-forecasts", action="store_true")
    ap.add_argument("--skip-observations", action="store_true")
    args = ap.parse_args()

    years = parse_years(args.years)

    # Check every file exists before touching the store, so a missing year fails the run
    # up front rather than half way through with a partly-loaded store behind it.
    def pick(stem: str):
        """Prefer .parquet: identical content, ~18x smaller (1.6 MB vs 29.2 MB for a year
        of reforecast), which matters when these have to be shipped to a CI runner. The
        parser handles both."""
        pq, csv = SAMPLES / f"{stem}.parquet", SAMPLES / f"{stem}.csv"
        return pq if pq.exists() else csv

    plan = []
    missing = []
    for y in years:
        if not args.skip_forecasts:
            f = pick(f"gefs_reforecast_india_{y}")
            (plan if f.exists() else missing).append(("forecast", y, f))
        if not args.skip_observations:
            o = pick(f"era5_observations_india_{y}")
            (plan if o.exists() else missing).append(("observed", y, o))
    if missing:
        for kind, y, f in missing:
            print(f"MISSING {kind} {y}: {f.name}", file=sys.stderr)
        return 1

    print(f"{len(plan)} files across {len(years)} years: {years[0]}..{years[-1]}")
    for kind, y, f in plan:
        print(f"  {kind:9s} {y}  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    if args.dry_run:
        print("\n--dry-run: nothing ingested")
        return 0

    from app.db.base import init_db, SessionLocal
    from app.ingestion.pipeline import ingest_upload
    from app.live.orchestrator import FORECAST_MAPPINGS, OBSERVATION_MAPPINGS

    init_db()
    total = 0
    print()
    for kind, y, f in plan:
        session = SessionLocal()
        try:
            res = ingest_upload(
                session, f, f.name,
                confirmed_mappings=FORECAST_MAPPINGS if kind == "forecast" else OBSERVATION_MAPPINGS,
                verification_status=None if kind == "forecast" else "final",
            )
            session.commit()
        finally:
            session.close()
        # ingest_upload returns pending_confirmation | ingested | failed. "ingested" is
        # the success case; anything else means the confirmed mappings did not apply and
        # continuing would leave a half-loaded store behind.
        if res.status != "ingested":
            print(f"  {kind:9s} {y}  status={res.status} - stopping", file=sys.stderr)
            return 1
        total += res.row_count_ingested
        print(f"  {kind:9s} {y}  {res.row_count_ingested:>7,} rows  "
              f"[{', '.join(sorted(res.canonical_variables_found))}]")

    print(f"\ningested {total:,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
