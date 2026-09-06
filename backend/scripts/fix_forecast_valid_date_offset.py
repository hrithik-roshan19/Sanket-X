"""One-off migration: correct the one-day offset on already-ingested forecast rows.

Background
----------
``fetch_gefs_reforecast_sample.py`` aggregated forecast hours ((k-1)*24, k*24] for lead
day k but labelled the result ``valid_date = init + k days``. For a 00 UTC cycle those
hours are calendar day ``init + (k-1)``, so every forecast row was stamped one day later
than its own contents and was therefore verified against the following day's observation.

Measured over the 2019 sample, pairing one day earlier lowers mean absolute error on 7 of
8 variables - 16.3% on precipitable water, 15.2% on pressure, 8.1% on average. Soil
moisture is unchanged, which is the expected control: it varies slowly enough that a
one-day shift should not matter.

The script fixes the labelling in place. No data is re-downloaded: only ``valid_date`` on
forecast rows moves back one day. Observed rows are untouched - they were always correct.

    python backend/scripts/fix_forecast_valid_date_offset.py --dry-run
    python backend/scripts/fix_forecast_valid_date_offset.py

A timestamped copy of every partition it rewrites is kept next to the store, and the
before/after pairing error is printed so the improvement is verifiable rather than assumed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.parquet_store import ARROW_SCHEMA, CANONICAL_DIR  # noqa: E402


def _partitions() -> list:
    return sorted(CANONICAL_DIR.glob("batch_id=*/part-*.parquet"))


def _pairing_error(label: str) -> None:
    """Mean absolute forecast-observation error per variable, as the store currently reads."""
    from app.storage import parquet_store as ps

    df = ps.read_dataset()
    if df.empty:
        print(f"  ({label}: store empty)")
        return
    ob = (df[df.value_type == "observed"][["region_id", "valid_date", "variable", "value"]]
          .rename(columns={"value": "obs"}))
    fc = df[df.value_type == "forecast"]
    print(f"  {label}")
    for var in sorted(fc.variable.unique()):
        f = (fc[fc.variable == var]
             .groupby(["region_id", "valid_date", "lead_time_days"], as_index=False)["value"]
             .mean().rename(columns={"value": "fcst"}))
        m = f.merge(ob[ob.variable == var], on=["region_id", "valid_date"], how="inner")
        if len(m):
            print(f"    {var:<26} n={len(m):>5}  MAE={(m.fcst - m.obs).abs().mean():8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    parts = _partitions()
    if not parts:
        sys.exit(f"no canonical partitions under {CANONICAL_DIR}")

    print(f"canonical store: {CANONICAL_DIR}")
    print(f"partitions     : {len(parts)}\n")
    print("BEFORE")
    _pairing_error("current pairing")

    planned = []
    for p in parts:
        df = pd.read_parquet(p)
        n_fc = int((df["value_type"] == "forecast").sum())
        if n_fc:
            planned.append((p, len(df), n_fc))

    print(f"\nforecast rows to relabel: {sum(n for _, _, n in planned):,} "
          f"across {len(planned)} partition(s)")
    for p, total, n_fc in planned:
        print(f"  {p.parent.name}  {n_fc:>7,} / {total:>7,} rows")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = CANONICAL_DIR.parent / f"canonical_backup_{stamp}"
    shutil.copytree(CANONICAL_DIR, backup)
    print(f"\nbackup written to {backup}")

    changed = 0
    for p, _, _ in planned:
        df = pd.read_parquet(p)
        mask = df["value_type"] == "forecast"
        vd = pd.to_datetime(df.loc[mask, "valid_date"]) - pd.Timedelta(days=1)
        df.loc[mask, "valid_date"] = vd.dt.date
        for col in ("init_date", "valid_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
        df["lead_time_days"] = df["lead_time_days"].astype("Int16")
        # The partition column is implied by the directory name, never stored in the file.
        df = df[[c for c in ARROW_SCHEMA.names if c in df.columns]]
        for col in ARROW_SCHEMA.names:
            if col not in df.columns:
                df[col] = None
        df = df[list(ARROW_SCHEMA.names)]
        pq.write_table(pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False), p)
        changed += int(mask.sum())

    print(f"relabelled {changed:,} forecast rows\n")
    print("AFTER")
    _pairing_error("corrected pairing")
    print(f"\nIf anything looks wrong, restore with:\n  rm -rf {CANONICAL_DIR} && mv {backup} {CANONICAL_DIR}")
    print("\nThe models are now stale with respect to the data - retrain before serving:")
    print("  python -m app.ml.train_pipeline")


if __name__ == "__main__":
    main()
