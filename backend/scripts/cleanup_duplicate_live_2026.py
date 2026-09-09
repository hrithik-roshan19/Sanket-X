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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safely remove the confirmed duplicate 2026-09-07 live forecast batch."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove the duplicate batch. Without this flag, only a dry-run is performed.",
    )
    return parser.parse_args()


def _load_forecast(init_date):
    from app.storage import parquet_store

    return parquet_store.read_dataset(
        value_types=["forecast"],
        init_dates=[pd.Timestamp(init_date).date()],
        dedupe=False,
    )


def _logical_duplicate_count(df):
    if df.empty:
        return 0
    missing = [column for column in LOGICAL_KEY if column not in df.columns]
    if missing:
        raise RuntimeError(
            f"Canonical forecast data is missing logical-key columns: {missing}"
        )
    return int(df.duplicated(LOGICAL_KEY).sum())


def _batch_counts(df):
    if df.empty:
        return {}
    if "batch_id" not in df.columns:
        raise RuntimeError("Canonical forecast data has no batch_id column.")
    return {
        str(batch_id): int(count)
        for batch_id, count in df.groupby("batch_id", dropna=False).size().items()
    }


def _assert_expected_pre_state(df):
    if df.empty:
        raise RuntimeError("No forecast rows found for the target init date.")

    counts = _batch_counts(df)
    expected_batches = {KEEP_BATCH_ID, DUPLICATE_BATCH_ID}
    actual_batches = set(counts)

    if actual_batches != expected_batches:
        raise RuntimeError(
            "Safety check failed: expected exactly the two confirmed live batches "
            f"{sorted(expected_batches)}, found {sorted(actual_batches)}."
        )

    if counts.get(KEEP_BATCH_ID) != 7100:
        raise RuntimeError(
            f"Safety check failed: keep batch has {counts.get(KEEP_BATCH_ID)} rows; "
            "expected 7100."
        )

    if counts.get(DUPLICATE_BATCH_ID) != 7100:
        raise RuntimeError(
            f"Safety check failed: duplicate batch has "
            f"{counts.get(DUPLICATE_BATCH_ID)} rows; expected 7100."
        )

    duplicate_rows = _logical_duplicate_count(df)
    if duplicate_rows != 7100:
        raise RuntimeError(
            f"Safety check failed: expected 7100 duplicate logical rows, "
            f"found {duplicate_rows}."
        )


def _mark_db_batch_superseded(batch_id):
    from app.db.base import SessionLocal
    from app.db.models import UploadBatch

    session = SessionLocal()
    try:
        batch = session.get(UploadBatch, batch_id)
        if batch is None:
            raise RuntimeError(
                f"UploadBatch {batch_id} was not found in the database."
            )

        batch.status = "superseded"
        notes = dict(batch.notes_json or {})
        notes.update(
            {
                "reason": "Exact duplicate of a retained live 2026 forecast batch.",
                "superseded_by_batch_id": KEEP_BATCH_ID,
                "target_init_date": TARGET_INIT_DATE,
            }
        )
        batch.notes_json = notes
        session.add(batch)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _delete_duplicate_partition(batch_id):
    from app.storage import parquet_store

    parquet_store.drop_batch(batch_id)


def _post_verify():
    df = _load_forecast(TARGET_INIT_DATE)

    if df.empty:
        raise RuntimeError(
            "Post-cleanup verification failed: no forecast rows remain."
        )

    counts = _batch_counts(df)

    if set(counts) != {KEEP_BATCH_ID}:
        raise RuntimeError(
            "Post-cleanup verification failed: unexpected batches remain: "
            f"{sorted(counts)}"
        )

    if counts.get(KEEP_BATCH_ID) != 7100:
        raise RuntimeError(
            "Post-cleanup verification failed: retained batch has "
            f"{counts.get(KEEP_BATCH_ID)} rows; expected 7100."
        )

    unique_keys = int(df[LOGICAL_KEY].drop_duplicates().shape[0])
    duplicate_rows = _logical_duplicate_count(df)

    if unique_keys != 7100:
        raise RuntimeError(
            "Post-cleanup verification failed: "
            f"unique logical keys={unique_keys}, expected 7100."
        )

    if duplicate_rows != 0:
        raise RuntimeError(
            "Post-cleanup verification failed: "
            f"duplicate logical rows={duplicate_rows}, expected 0."
        )

    return {
        "rows_after": int(len(df)),
        "batches_after": counts,
        "unique_logical_keys_after": unique_keys,
        "duplicate_logical_rows_after": duplicate_rows,
    }


def main():
    args = parse_args()

    print("=" * 72)
    print("SANKET-X — DUPLICATE LIVE 2026 CLEANUP")
    print("=" * 72)
    print(f"Target init date : {TARGET_INIT_DATE}")
    print(f"Keep batch       : {KEEP_BATCH_ID}")
    print(f"Remove batch     : {DUPLICATE_BATCH_ID}")
    print()

    before = _load_forecast(TARGET_INIT_DATE)
    _assert_expected_pre_state(before)

    counts = _batch_counts(before)
    unique_keys = int(before[LOGICAL_KEY].drop_duplicates().shape[0])
    duplicate_rows = _logical_duplicate_count(before)

    print("PRE-CLEANUP VERIFICATION: PASS")
    print(f"Rows before         : {len(before)}")
    print(f"Keep batch rows     : {counts[KEEP_BATCH_ID]}")
    print(f"Remove batch rows   : {counts[DUPLICATE_BATCH_ID]}")
    print(f"Unique logical keys : {unique_keys}")
    print(f"Duplicate rows      : {duplicate_rows}")
    print()

    if not args.apply:
        print("DRY RUN ONLY — no data changed.")
        return

    print("APPLY MODE")
    print("Removing only the confirmed duplicate partition...")
    _delete_duplicate_partition(DUPLICATE_BATCH_ID)

    print("Parquet duplicate partition removed.")
    print("Marking duplicate UploadBatch as superseded...")
    _mark_db_batch_superseded(DUPLICATE_BATCH_ID)

    after = _post_verify()

    print()
    print("POST-CLEANUP VERIFICATION: PASS")
    print(f"Rows after                : {after['rows_after']}")
    print(f"Unique logical keys after : {after['unique_logical_keys_after']}")
    print(f"Duplicate logical rows    : {after['duplicate_logical_rows_after']}")
    print(f"Remaining batch           : {KEEP_BATCH_ID}")
    print()
    print("LIVE 2026 DUPLICATE CLEANUP: PASS")


if __name__ == "__main__":
    main()
