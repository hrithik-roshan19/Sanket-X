from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DUPLICATE_BATCH_ID = "bda4827b-44e7-46ab-9999-ed26c5ecf407"
KEEP_BATCH_ID = "40c5d957-7acb-437f-832c-85033b79e7e1"
TARGET_INIT_DATE = "2026-09-07"

LOGICAL_KEY = [
    "region_id",
    "init_date",
    "valid_date",
    "lead_time_days",
    "ensemble_member_id",
    "variable",
]


def _load_forecast():
    from app.storage import parquet_store

    return parquet_store.read_dataset(
        value_types=["forecast"],
        init_dates=[pd.Timestamp(TARGET_INIT_DATE).date()],
        dedupe=False,
    )


def _batch_counts(df):
    if df.empty:
        return {}
    if "batch_id" not in df.columns:
        raise RuntimeError("Canonical forecast data has no batch_id column.")
    return {
        str(batch_id): int(count)
        for batch_id, count in df.groupby("batch_id", dropna=False).size().items()
    }


def _logical_duplicate_count(df):
    if df.empty:
        return 0
    missing = [c for c in LOGICAL_KEY if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing logical-key columns: {missing}")
    return int(df.duplicated(LOGICAL_KEY).sum())


def _verify_partial_apply_state():
    df = _load_forecast()

    if df.empty:
        raise RuntimeError("No forecast rows remain for the target init date.")

    counts = _batch_counts(df)

    if set(counts) != {KEEP_BATCH_ID}:
        raise RuntimeError(
            "Safety check failed: expected only the retained batch after the "
            f"successful parquet deletion. Found: {sorted(counts)}"
        )

    if counts[KEEP_BATCH_ID] != 7100:
        raise RuntimeError(
            f"Safety check failed: retained batch has {counts[KEEP_BATCH_ID]} rows; "
            "expected 7100."
        )

    duplicate_rows = _logical_duplicate_count(df)
    if duplicate_rows != 0:
        raise RuntimeError(
            f"Safety check failed: logical duplicates remain: {duplicate_rows}."
        )

    return df, counts


def _mark_db_batch_superseded():
    from app.db.base import SessionLocal
    from app.db.models import UploadBatch

    session = SessionLocal()
    try:
        batch = session.get(UploadBatch, DUPLICATE_BATCH_ID)
        if batch is None:
            raise RuntimeError(
                f"UploadBatch {DUPLICATE_BATCH_ID} was not found in the database."
            )

        # UploadBatch.notes_json is a JSON list in this project.
        existing = batch.notes_json
        if isinstance(existing, list):
            notes = list(existing)
        elif existing is None:
            notes = []
        else:
            notes = [existing]

        note = {
            "reason": "Exact duplicate of retained live 2026 forecast batch.",
            "superseded_by_batch_id": KEEP_BATCH_ID,
            "target_init_date": TARGET_INIT_DATE,
            "cleanup": "duplicate parquet partition removed",
        }

        # Avoid adding the same cleanup note twice if this repair is rerun.
        if note not in notes:
            notes.append(note)

        batch.status = "superseded"
        batch.notes_json = notes
        session.add(batch)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _verify_database_state():
    from app.db.base import SessionLocal
    from app.db.models import UploadBatch

    session = SessionLocal()
    try:
        batch = session.get(UploadBatch, DUPLICATE_BATCH_ID)
        if batch is None:
            raise RuntimeError(
                f"Post-repair verification failed: UploadBatch "
                f"{DUPLICATE_BATCH_ID} not found."
            )

        if batch.status != "superseded":
            raise RuntimeError(
                "Post-repair verification failed: duplicate UploadBatch status is "
                f"{batch.status!r}, expected 'superseded'."
            )

        return batch.status
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repair the database state after the duplicate 2026-09-07 "
            "parquet partition was successfully removed but DB marking failed."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Mark the already-removed duplicate UploadBatch as superseded.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("SANKET-X — REPAIR PARTIAL DUPLICATE CLEANUP")
    print("=" * 72)
    print(f"Target init date : {TARGET_INIT_DATE}")
    print(f"Keep batch       : {KEEP_BATCH_ID}")
    print(f"Removed batch    : {DUPLICATE_BATCH_ID}")
    print()

    _, counts = _verify_partial_apply_state()

    print("PARQUET STATE VERIFICATION: PASS")
    print(f"Rows present     : {sum(counts.values())}")
    print(f"Retained rows    : {counts[KEEP_BATCH_ID]}")
    print("Duplicate batch  : absent")
    print("Logical duplicates: 0")
    print()

    if not args.apply:
        print("DRY RUN ONLY — database was not changed.")
        print("The parquet cleanup already succeeded; only DB status repair remains.")
        return

    print("APPLY MODE")
    print("Marking removed duplicate UploadBatch as superseded...")
    _mark_db_batch_superseded()

    status = _verify_database_state()

    print()
    print("DATABASE REPAIR VERIFICATION: PASS")
    print(f"Duplicate UploadBatch status : {status}")
    print()
    print("PARTIAL CLEANUP REPAIR: PASS")
    print("Canonical parquet state and database state are consistent.")


if __name__ == "__main__":
    main()
