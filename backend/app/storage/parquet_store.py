from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from app.db.base import resolve_path
from app.config import settings
from app.ingestion.canonical_schema import CANONICAL_COLUMNS

log = logging.getLogger(__name__)

CANONICAL_DIR = resolve_path(settings.canonical_dir)

_ROW_GROUP_SIZE = 16_384

ARROW_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("upload_batch_id", pa.string()),
        ("source_file", pa.string()),
        ("source_column", pa.string()),
        ("variable", pa.string()),
        ("value_type", pa.string()),
        ("value", pa.float64()),
        ("region_id", pa.string()),
        ("region_name", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("init_date", pa.date32()),
        ("valid_date", pa.date32()),
        ("lead_time_days", pa.int16()),
        ("ensemble_member_id", pa.string()),
        ("mapping_confidence", pa.float64()),
        ("ingested_at", pa.timestamp("us")),
        ("grain", pa.string()),
        ("region_resolution_method", pa.string()),
        ("verification_status", pa.string()),
    ]
)

assert [f.name for f in ARROW_SCHEMA] == list(CANONICAL_COLUMNS), (
    "ARROW_SCHEMA drifted from CanonicalRow field order"
)


def _partition_dir(batch_id: str) -> Path:
    return CANONICAL_DIR / f"batch_id={batch_id}"


def append_batch(batch_id: str, rows: Iterable[dict] | pd.DataFrame) -> int:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if df.empty:
        return 0

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(CANONICAL_COLUMNS)]

    for col in ("init_date", "valid_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
    df["lead_time_days"] = df["lead_time_days"].astype("Int16")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.sort_values(
        ["value_type", "init_date", "valid_date", "variable"],
        kind="mergesort", na_position="first",
    ).reset_index(drop=True)

    table = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)

    part_dir = _partition_dir(batch_id)
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part_dir / "part-0.parquet", row_group_size=_ROW_GROUP_SIZE)
    return len(df)


def compact_store() -> dict:
    before = _row_count_on_disk()
    df = read_dataset()
    if df.empty:
        return {"rows_before": before, "rows_after": before, "removed": 0, "batches": 0}

    kept: set = set()
    for batch_id, group in df.groupby("upload_batch_id", observed=True, dropna=False):
        if pd.isna(batch_id):
            continue
        append_batch(str(batch_id), group.copy())
        kept.add(str(batch_id))

    for part in CANONICAL_DIR.glob("batch_id=*"):
        if part.is_dir() and part.name.split("=", 1)[1] not in kept:
            shutil.rmtree(part, ignore_errors=True)

    after = _row_count_on_disk()
    log.info("compacted store: %s -> %s rows across %s batches", f"{before:,}",
             f"{after:,}", len(kept))
    return {"rows_before": before, "rows_after": after,
            "removed": before - after, "batches": len(kept)}


def _row_count_on_disk() -> int:
    try:
        return ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive").count_rows()
    except Exception:  # noqa: BLE001 - an absent store is zero rows, not a crash
        return 0


def drop_batch(batch_id: str) -> None:
    part_dir = _partition_dir(batch_id)
    if part_dir.exists():
        shutil.rmtree(part_dir)


def store_fingerprint() -> str:
    if not CANONICAL_DIR.exists():
        return "empty"
    parts = sorted(
        f"{p.name}:{int(p.stat().st_mtime_ns)}"
        for p in CANONICAL_DIR.glob("batch_id=*")
        if p.is_dir()
    )
    return "|".join(parts) if parts else "empty"


def _dataset() -> ds.Dataset | None:
    if not CANONICAL_DIR.exists() or not any(CANONICAL_DIR.glob("batch_id=*/*.parquet")):
        return None
    schema = ARROW_SCHEMA.append(pa.field("batch_id", pa.string()))
    return ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive", schema=schema)


_DEDUPE_KEY = [
    "region_id", "valid_date", "variable", "value_type",
    "init_date", "lead_time_days", "ensemble_member_id", "source_column",
]


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not set(_DEDUPE_KEY).issubset(df.columns):
        return df

    rank = (
        (df["verification_status"] != "provisional").astype(int)
        if "verification_status" in df.columns
        else pd.Series(1, index=df.index)
    )
    order = pd.to_datetime(df.get("ingested_at"), errors="coerce")

    tmp = df.assign(_rank=rank, _order=order).sort_values(
        ["_rank", "_order"], kind="mergesort"
    )

    keys = []
    for col in _DEDUPE_KEY:
        s = tmp[col]
        if col == "region_id":
            s = s.astype(object).where(
                s.notna(),
                tmp["lat"].astype(str) + "," + tmp["lon"].astype(str),
            )
        keys.append(s.astype(object).where(s.notna(), "\x00"))

    deduped = tmp.groupby(keys, sort=False, dropna=False).tail(1)
    return deduped.drop(columns=["_rank", "_order"]).sort_index()


def read_dataset(
    variables: Sequence[str] | None = None,
    value_types: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    init_dates: Sequence | None = None,
    valid_date_min: date | None = None,
    valid_date_max: date | None = None,
    exclude_provisional: bool = False,
    dedupe: bool = True,
) -> pd.DataFrame:
    dataset = _dataset()
    if dataset is None:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    filt = None

    def _and(expr):
        nonlocal filt
        filt = expr if filt is None else (filt & expr)

    if variables:
        _and(ds.field("variable").isin(list(variables)))
    if value_types:
        _and(ds.field("value_type").isin(list(value_types)))
    if init_dates:
        _and(ds.field("init_date").isin([_as_date(d) for d in init_dates]))
    if valid_date_min is not None:
        _and(ds.field("valid_date") >= _as_date(valid_date_min))
    if valid_date_max is not None:
        _and(ds.field("valid_date") <= _as_date(valid_date_max))
    if exclude_provisional:
        _and(ds.field("verification_status").is_null()
             | (ds.field("verification_status") != "provisional"))

    want = list(columns) if columns else None
    read_cols = want
    if want and dedupe:
        read_cols = list(dict.fromkeys(
            want + _DEDUPE_KEY + ["lat", "lon", "verification_status", "ingested_at"]
        ))
        read_cols = [c for c in read_cols if c in ARROW_SCHEMA.names]

    df = dataset.to_table(filter=filt, columns=read_cols).to_pandas()
    if dedupe:
        df = _dedupe(df)
    return df[want] if want else df


def _as_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    return pd.to_datetime(v).date()


def has_forecast_cycle(init_date: date, min_rows: int = 1) -> bool:
    dataset = _dataset()
    if dataset is None:
        return False
    n = dataset.count_rows(
        filter=(ds.field("value_type") == "forecast")
        & (ds.field("init_date") == _as_date(init_date))
    )
    return n >= min_rows


def latest_forecast_init_date() -> date | None:
    dataset = _dataset()
    if dataset is None:
        return None
    tbl = dataset.to_table(
        filter=(ds.field("value_type") == "forecast"), columns=["init_date"]
    )
    if tbl.num_rows == 0:
        return None
    s = tbl.to_pandas()["init_date"].dropna()
    return None if s.empty else _as_date(s.max())


_summary_lock = threading.Lock()
_summary_memo: "dict[str, dict]" = {}

SUMMARY_CACHE_PATH = CANONICAL_DIR.parent / "summary.json"


def store_signature() -> str:
    if not CANONICAL_DIR.exists():
        return "empty"
    parts = []
    for f in sorted(CANONICAL_DIR.glob("batch_id=*/*.parquet")):
        try:
            rows = pq.ParquetFile(f).metadata.num_rows
        except Exception:  # noqa: BLE001 - an unreadable partition invalidates the cache
            return f"unreadable:{f.name}"
        parts.append(f"{f.parent.name}/{f.name}:{f.stat().st_size}:{rows}")
    return "|".join(parts) if parts else "empty"


def write_summary_cache() -> dict:
    summary = _compute_summary()
    SUMMARY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CACHE_PATH.write_text(
        json.dumps({"signature": store_signature(), "summary": summary}, indent=2)
    )
    return summary


def _read_summary_cache() -> dict | None:
    if not SUMMARY_CACHE_PATH.is_file():
        return None
    try:
        blob = json.loads(SUMMARY_CACHE_PATH.read_text())
    except Exception:  # noqa: BLE001 - a corrupt sidecar just means recompute
        return None
    if blob.get("signature") != store_signature():
        return None
    got = blob.get("summary")
    return got if isinstance(got, dict) else None

_SUMMARY_COLUMNS = [
    "variable", "value_type", "upload_batch_id", "region_id", "valid_date", "grain",
    "init_date",
]


def dataset_summary() -> dict:
    fp = store_fingerprint()
    with _summary_lock:
        hit = _summary_memo.get(fp)
    if hit is not None:
        return dict(hit)

    summary = _read_summary_cache()
    if summary is None:
        summary = _summary_without_cache()
    with _summary_lock:
        _summary_memo.clear()
        _summary_memo[fp] = summary
    return dict(summary)


# Past this the summary peaks ~918 MB and the 512 MB container is killed; refuse instead.
_SUMMARY_COMPUTE_MAX_ROWS = 800_000


def _summary_without_cache() -> dict:
    try:
        n = ds.dataset(CANONICAL_DIR, format="parquet", partitioning="hive").count_rows()
    except Exception:  # noqa: BLE001 - an unreadable store is not a reason to crash
        n = 0
    if n > _SUMMARY_COMPUTE_MAX_ROWS:
        log.error(
            "summary sidecar missing or stale for a %s-row store; refusing to compute it "
            "here (needs ~300 MB and this process has far less). Republish so "
            "package_for_deploy regenerates data/summary.json.", f"{n:,}")
        return {
            "total_rows": None, "batches": None, "by_variable": {}, "regions": None,
            "valid_date_min": None, "valid_date_max": None,
            "forecast_cycles": None, "init_date_min": None, "init_date_max": None,
            "unavailable_reason": (
                "The store summary is computed during packaging and shipped alongside the "
                "data. This deployment's copy is missing or out of date, and recomputing "
                "it here would exceed the memory this instance has. Everything else on "
                "this page is unaffected."
            ),
        }
    return _compute_summary()


def _compute_summary() -> dict:
    df = read_dataset(columns=_SUMMARY_COLUMNS)
    if df.empty:
        return {"total_rows": 0, "batches": 0, "by_variable": {}, "regions": 0,
                "valid_date_min": None, "valid_date_max": None,
                "forecast_cycles": 0, "init_date_min": None, "init_date_max": None}

    by_var = (
        df.groupby(["variable", "value_type"]).size()
        .unstack(fill_value=0).to_dict(orient="index")
    )
    return {
        "total_rows": int(len(df)),
        "batches": int(df["upload_batch_id"].nunique()),
        "by_variable": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in by_var.items()},
        "regions": int(df["region_id"].nunique(dropna=True)),
        "valid_date_min": str(df["valid_date"].min()),
        "valid_date_max": str(df["valid_date"].max()),
        "grain_counts": {k: int(v) for k, v in df["grain"].value_counts().items()},
        **_cycle_coverage(df),
    }


def _cycle_coverage(df) -> dict:
    inits = df.loc[df["value_type"] == "forecast", "init_date"].dropna()
    if inits.empty:
        return {"forecast_cycles": 0, "init_date_min": None, "init_date_max": None}
    return {
        "forecast_cycles": int(inits.nunique()),
        "init_date_min": str(inits.min()),
        "init_date_max": str(inits.max()),
    }
