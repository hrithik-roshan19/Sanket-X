"""
Shared pytest fixtures for Sanket-X.

The tests must never use the developer's real data/model store.

Windows note:
pytest's default temporary directory and SQLite cleanup can race with
file handles left by SQLAlchemy/TestClient. We therefore use a dedicated
project-local temporary root and explicitly dispose SQLAlchemy engines
before destructive cleanup.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Stable test runtime root
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

TEST_RUNTIME_ROOT = (
    BACKEND_DIR / ".pytest-runtime"
)

TEST_RUNTIME_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

# Force Python tempfile APIs to use the same stable location.
# This affects:
#   - tempfile.mkdtemp()
#   - API test temporary uploads
#   - other application code using tempfile
os.environ["TMP"] = str(TEST_RUNTIME_ROOT)
os.environ["TEMP"] = str(TEST_RUNTIME_ROOT)
os.environ["TMPDIR"] = str(TEST_RUNTIME_ROOT)

# pytest's tmp_path_factory uses this when --basetemp is not supplied.
os.environ["PYTEST_DEBUG_TEMPROOT"] = str(
    TEST_RUNTIME_ROOT
)


# ---------------------------------------------------------------------------
# One isolated data directory for the complete pytest session
# ---------------------------------------------------------------------------

_TMP = Path(
    tempfile.mkdtemp(
        prefix="forecastguard-test-",
        dir=TEST_RUNTIME_ROOT,
    )
)

os.environ["DATA_DIR"] = str(_TMP)
os.environ["DB_PATH"] = str(
    _TMP / "metadata.db"
)
os.environ["CANONICAL_DIR"] = str(
    _TMP / "canonical"
)
os.environ["RAW_UPLOAD_DIR"] = str(
    _TMP / "raw_uploads"
)
os.environ["MODEL_DIR"] = str(
    _TMP / "models"
)

# Local tests are explicitly allowed to retrain.
os.environ["ALLOW_LOCAL_RETRAIN"] = "true"


# ---------------------------------------------------------------------------
# Python import path
# ---------------------------------------------------------------------------

import sys

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR),
    )


# ---------------------------------------------------------------------------
# Real sample data
# ---------------------------------------------------------------------------

GENERATED_SAMPLES = (
    BACKEND_DIR
    / "data"
    / "samples"
)

GEFS_CSV = (
    GENERATED_SAMPLES
    / "gefs_reforecast_india_2019.csv"
)

ERA5_CSV = (
    GENERATED_SAMPLES
    / "era5_observations_india_2019.csv"
)


def _ensure_sample_csvs() -> None:
    """
    The repository keeps parquet samples as the compact source of truth.
    Tests consume CSV, so create CSV twins when necessary.
    """

    import pandas as pd

    for csv_path in (
        GEFS_CSV,
        ERA5_CSV,
    ):
        parquet_path = (
            csv_path.with_suffix(".parquet")
        )

        if (
            csv_path.exists()
            or not parquet_path.exists()
        ):
            continue

        try:
            pd.read_parquet(
                parquet_path
            ).to_csv(
                csv_path,
                index=False,
            )
        except ImportError:
            return


_ensure_sample_csvs()


# ---------------------------------------------------------------------------
# Sample discovery helpers
# ---------------------------------------------------------------------------

def _resolve_sample_dir() -> Path:
    candidates = (
        os.environ.get(
            "FORECASTGUARD_SAMPLE_DIR"
        ),
        str(
            Path.home()
            / "Desktop"
            / "data"
        ),
        str(GENERATED_SAMPLES),
    )

    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)

    return GENERATED_SAMPLES


SAMPLE_DIR = _resolve_sample_dir()


def iter_sample_files():
    extensions = {
        ".csv",
        ".tsv",
        ".txt",
        ".xlsx",
        ".xls",
        ".json",
        ".parquet",
    }

    roots = {
        SAMPLE_DIR,
        GENERATED_SAMPLES,
    }

    for root in roots:
        if not root.exists():
            continue

        for path in sorted(
            Path(root).rglob("*")
        ):
            if (
                path.is_file()
                and path.suffix.lower()
                in extensions
                and "_gefs_parts"
                not in path.parts
            ):
                yield path


def find_sample(
    *name_fragments: str,
) -> Path | None:
    for path in iter_sample_files():
        name = path.name.lower()

        if all(
            fragment.lower()
            in name
            for fragment in name_fragments
        ):
            return path

    return None


# ---------------------------------------------------------------------------
# Database/store cleanup helpers
# ---------------------------------------------------------------------------

def _dispose_engine() -> None:
    """
    Dispose every SQLAlchemy connection held by the application.

    Import lazily because conftest must configure environment variables
    before app.db.base is imported.
    """

    try:
        from app.db.base import engine

        engine.dispose()
    except Exception:
        pass


def _clean_store() -> None:
    """
    Remove test-owned data while keeping the parent test directory alive.
    """

    try:
        from app.storage import parquet_store

        shutil.rmtree(
            parquet_store.CANONICAL_DIR,
            ignore_errors=True,
        )
    except Exception:
        pass

    shutil.rmtree(
        _TMP / "raw_uploads",
        ignore_errors=True,
    )

    shutil.rmtree(
        _TMP / "models",
        ignore_errors=True,
    )


def _reset_database() -> None:
    """
    Drop and recreate the test database.

    The engine is disposed before and after DDL to avoid Windows
    file-lock problems.
    """

    from app.db.base import (
        Base,
        engine,
        init_db,
    )

    _dispose_engine()

    try:
        Base.metadata.drop_all(
            engine
        )
    finally:
        engine.dispose()

    init_db()

    engine.dispose()


# ---------------------------------------------------------------------------
# Per-test clean store
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_store():
    """
    Clean metadata DB and canonical/model/upload stores for one test.
    """

    from app.db.base import init_db

    _dispose_engine()
    _clean_store()
    _reset_database()

    init_db()

    try:
        yield
    finally:
        _dispose_engine()
        _clean_store()

        # Keep the SQLite DB available for subsequent tests.
        try:
            init_db()
        except Exception:
            pass

        _dispose_engine()


@pytest.fixture
def session(fresh_store):
    from app.db.base import SessionLocal

    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Shared real-data ingestion helper
# ---------------------------------------------------------------------------

def _confirm_all_mappings(
    result,
    session,
):
    """
    Confirm all non-conflicting measurement proposals exactly once.
    """

    from app.ingestion.pipeline import (
        confirm_mapping,
    )

    if result.status != "pending_confirmation":
        return result

    seen = set()
    mappings = []

    for proposal in sorted(
        result.mapping_proposals,
        key=lambda item: item[
            "source_column"
        ],
    ):
        if (
            proposal["role"]
            != "measurement"
        ):
            continue

        if (
            proposal["decision"]
            != "needs_confirmation"
        ):
            continue

        if not proposal[
            "suggested_variable"
        ]:
            continue

        key = (
            proposal[
                "suggested_variable"
            ],
            proposal[
                "suggested_value_type"
            ],
        )

        if key in seen:
            continue

        seen.add(key)

        mappings.append(
            {
                "source_column": proposal[
                    "source_column"
                ],
                "variable": proposal[
                    "suggested_variable"
                ],
                "value_type": proposal[
                    "suggested_value_type"
                ],
                "unit_conversion": proposal[
                    "unit_conversion"
                ],
            }
        )

    return confirm_mapping(
        session,
        result.batch_id,
        mappings,
    )


# ---------------------------------------------------------------------------
# Shared reduced real-data slice
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _ingested_slice():
    """
    Ingest the first eight real GEFS cycles plus the ERA5 sample.

    This is intentionally separate from pytest's tmp_path_factory so the
    test suite does not depend on Windows' global pytest temp directory.
    """

    import pandas as pd

    from app.db.base import (
        SessionLocal,
        engine,
        init_db,
    )
    from app.db.models import Base
    from app.ingestion.pipeline import (
        ingest_upload,
    )
    from app.storage import parquet_store

    _dispose_engine()

    Base.metadata.drop_all(
        engine
    )

    shutil.rmtree(
        parquet_store.CANONICAL_DIR,
        ignore_errors=True,
    )

    shutil.rmtree(
        _TMP / "models",
        ignore_errors=True,
    )

    init_db()

    work_dir = (
        _TMP
        / "fixture-inputs"
        / f"mlslice-{uuid.uuid4().hex}"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ge_fs = pd.read_csv(
        GEFS_CSV
    )

    keep_cycles = sorted(
        ge_fs["init_date"]
        .dropna()
        .unique()
    )[:8]

    ge_fs = ge_fs[
        ge_fs["init_date"].isin(
            keep_cycles
        )
    ].copy()

    gefs_path = (
        work_dir / "gefs.csv"
    )

    ge_fs.to_csv(
        gefs_path,
        index=False,
    )

    era5 = pd.read_csv(
        ERA5_CSV
    )

    era5_path = (
        work_dir / "era5.csv"
    )

    era5.to_csv(
        era5_path,
        index=False,
    )

    session = SessionLocal()

    try:
        gefs_result = ingest_upload(
            session,
            gefs_path,
            gefs_path.name,
        )

        _confirm_all_mappings(
            gefs_result,
            session,
        )

        era5_result = ingest_upload(
            session,
            era5_path,
            era5_path.name,
        )

        _confirm_all_mappings(
            era5_result,
            session,
        )

        session.commit()

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()

    try:
        yield
    finally:
        _dispose_engine()

        shutil.rmtree(
            parquet_store.CANONICAL_DIR,
            ignore_errors=True,
        )

        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )


# ---------------------------------------------------------------------------
# Shared retraining fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _retrain(_ingested_slice):
    """
    Perform one genuine local retrain against the real eight-cycle slice.

    No mocked model or fabricated metrics are used.
    """

    from app.ml.train_pipeline import (
        full_retrain,
    )

    report = full_retrain(
        make_current=True
    )

    if report.status != "success":
        raise AssertionError(
            f"Local retrain failed: "
            f"{report.error}"
        )

    return report


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------

def pytest_sessionfinish(
    session,
    exitstatus,
):
    """
    Final cleanup after pytest has finished all fixtures.
    """

    _dispose_engine()

    try:
        from app.ml import inference

        inference.invalidate_caches()
    except Exception:
        pass

    try:
        from app.services import replay_service

        replay_service.invalidate()
    except Exception:
        pass

    _dispose_engine()

    # Remove only our test session directory.
    # Do not delete TEST_RUNTIME_ROOT itself because pytest may still have
    # its own temporary bookkeeping there.
    shutil.rmtree(
        _TMP,
        ignore_errors=True,
    )