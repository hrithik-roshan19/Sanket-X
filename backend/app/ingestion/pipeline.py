from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud
from app.db.base import get_session, init_db, resolve_path
from app.ingestion.canonical_schema import CanonicalVariable, ValueType
from app.ingestion.parsers import ParsedTable, parse_upload
from app.ingestion import schema_mapper as sm
from app.storage import parquet_store
from app.utils.geo import get_resolver

RAW_DIR = resolve_path(settings.raw_upload_dir)

_KMH_TO_MS = 1.0 / 3.6


@dataclass
class IngestResult:
    batch_id: str
    status: str
    layout: Optional[str] = None
    detected_format: Optional[str] = None
    row_count_raw: int = 0
    row_count_ingested: int = 0
    skipped_rows: int = 0
    canonical_variables_found: list = field(default_factory=list)
    region_resolution_rate: float = 0.0
    profile_match: str = "none"
    mapping_proposals: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    should_retrain: bool = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _proposal_to_dict(p: sm.ColumnProposal) -> dict:
    return {
        "source_column": p.source_column,
        "normalized": p.normalized,
        "role": p.role,
        "sample_values": p.sample_values,
        "suggested_variable": p.suggested_variable,
        "suggested_value_type": p.suggested_value_type,
        "confidence": p.confidence,
        "ambiguity_gap": p.ambiguity_gap,
        "method": p.method,
        "unit_conversion": p.unit_conversion,
        "decision": p.decision,
        "alternatives": p.alternatives,
    }


def _mapping_rows_for_db(result: sm.MappingResult) -> list[dict]:
    rows = []
    for p in result.proposals:
        rows.append(
            dict(
                source_column=p.source_column,
                source_header_normalized=p.normalized,
                role=p.role,
                mapped_variable=p.suggested_variable if p.role in ("measurement",) else None,
                mapped_value_type=p.suggested_value_type,
                decision=p.decision,
                method=p.method,
                confidence=float(p.confidence or 0.0),
                ambiguity_gap=float(p.ambiguity_gap or 0.0),
                unit_conversion=p.unit_conversion,
                sample_values_json=p.sample_values,
            )
        )
    return rows


def _dimension_columns(result: sm.MappingResult) -> dict:
    out = {}
    for p in result.proposals:
        if p.role == sm.ROLE_DIMENSION and p.suggested_variable:
            out.setdefault(p.suggested_variable, p.source_column)
    return out


def _apply_unit(value: float, conversion: Optional[str]) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if conversion == "kmh_to_ms":
        return value * _KMH_TO_MS
    if conversion == "K_to_C":
        return value - 273.15
    if conversion == "Pa_to_hPa":
        return value / 100.0
    if conversion == "frac_to_pct":
        return value * 100.0
    return value


def _coerce_date(v) -> Optional[date]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    ts = pd.to_datetime(v, errors="coerce", utc=False)
    return None if pd.isna(ts) else ts.date()


def _reconcile_time(init_d, valid_d, lead_days) -> tuple[Optional[date], Optional[date], Optional[int]]:
    have = sum(x is not None for x in (init_d, valid_d, lead_days))
    if have < 2:
        return init_d, valid_d, lead_days
    if valid_d is None:
        valid_d = init_d + pd.Timedelta(days=int(lead_days))
        valid_d = valid_d.date() if hasattr(valid_d, "date") else valid_d
    elif init_d is None:
        init_d = valid_d - pd.Timedelta(days=int(lead_days))
        init_d = init_d.date() if hasattr(init_d, "date") else init_d
    elif lead_days is None:
        lead_days = (valid_d - init_d).days
    return init_d, valid_d, lead_days


def to_canonical_rows(
    df: pd.DataFrame,
    result: sm.MappingResult,
    *,
    batch_id: str,
    source_file: str,
    grain: str,
    filename_hint: str = "",
) -> tuple[pd.DataFrame, int, list]:
    notes: list = []
    resolver = get_resolver()
    ingested_at = datetime.now(timezone.utc)
    dims = _dimension_columns(result)

    region_col = dims.get("region")
    lat_col, lon_col = dims.get("lat"), dims.get("lon")
    valid_col, init_col, lead_col = dims.get("valid_date"), dims.get("init_date"), dims.get("lead_time_days")
    member_col = dims.get("ensemble_member_id")
    vt_col = result.value_type_column

    plan: list[tuple] = []
    if result.layout == "long":
        var_name_col = next((p.source_column for p in result.proposals
                             if p.role == sm.ROLE_VARIABLE_NAME), None)
        value_cols = [(p.source_column, p.suggested_value_type)
                      for p in result.proposals if p.role == sm.ROLE_VALUE]
        if not var_name_col or not value_cols:
            notes.append("long layout but no variable-name/value columns resolved; nothing ingested")
            return pd.DataFrame(), len(df), notes
    else:
        for scol, spec in result.measurement_map().items():
            plan.append((scol, spec["variable"], spec["value_type"], spec.get("unit_conversion")))
        if not plan:
            notes.append("no accepted measurement columns; nothing ingested")
            return pd.DataFrame(), len(df), notes

    _region_cache: dict = {}

    def _resolve_region(name, lat, lon):
        key = (name, None if lat is None else round(lat, 3), None if lon is None else round(lon, 3))
        if key not in _region_cache:
            _region_cache[key] = resolver.resolve(name=name, lat=lat, lon=lon)
        return _region_cache[key]

    def _num(row, col):
        if not col or col not in row:
            return None
        v = pd.to_numeric(row[col], errors="coerce")
        return None if pd.isna(v) else float(v)

    def _norm_value_type(raw) -> Optional[str]:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        n = sm.normalize_header(str(raw))
        for vt, syns in sm.VALUE_TYPE_SYNONYMS.items():
            if any(s in n or n in s for s in syns):
                return vt.value
        return None

    out_rows: list[dict] = []
    skipped = 0
    resolved_regions = 0
    total_region_attempts = 0

    records = df.to_dict("records")
    for row in records:
        name = str(row[region_col]).strip() if region_col and pd.notna(row.get(region_col)) else None
        lat, lon = _num(row, lat_col), _num(row, lon_col)
        init_d = _coerce_date(row.get(init_col)) if init_col else None
        valid_d = _coerce_date(row.get(valid_col)) if valid_col else None
        lead = _num(row, lead_col)
        if lead is not None:
            lead = int(round(lead / 24)) if lead >= 24 else int(round(lead))
        init_d, valid_d, lead = _reconcile_time(init_d, valid_d, lead)

        if valid_d is None:
            skipped += 1
            continue

        member = str(row[member_col]).strip() if member_col and pd.notna(row.get(member_col)) else None
        row_vt = _norm_value_type(row.get(vt_col)) if vt_col else None

        rm = _resolve_region(name, lat, lon)
        total_region_attempts += 1
        if rm.region_id is not None:
            resolved_regions += 1

        emissions: list[tuple] = []
        if result.layout == "long":
            var_label = row.get(var_name_col)
            variable = _match_variable(var_label)
            if variable is None:
                skipped += 1
                continue
            for vcol, vcol_vt in value_cols:
                raw = _num(row, vcol)
                if raw is None:
                    continue
                vt = row_vt or vcol_vt
                emissions.append((variable, vt, raw, None, vcol))
        else:
            for scol, variable, col_vt, unit in plan:
                raw = _num(row, scol)
                if raw is None:
                    continue
                vt = row_vt or col_vt
                emissions.append((variable, vt, raw, unit, scol))

        for variable, vt, raw, unit, scol in emissions:
            if vt is None:
                skipped += 1
                continue
            value = _apply_unit(raw, unit)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            lt = lead if (vt == ValueType.FORECAST.value and lead and 1 <= lead <= 10) else None
            out_rows.append(
                dict(
                    record_id=str(uuid.uuid4()),
                    upload_batch_id=batch_id,
                    source_file=source_file,
                    source_column=scol,
                    variable=variable,
                    value_type=vt,
                    value=float(value),
                    region_id=rm.region_id,
                    region_name=rm.region_name or name,
                    lat=lat,
                    lon=lon,
                    init_date=init_d if vt == ValueType.FORECAST.value else None,
                    valid_date=valid_d,
                    lead_time_days=lt,
                    ensemble_member_id=member,
                    mapping_confidence=_confidence_for(result, scol),
                    ingested_at=ingested_at,
                    grain=grain,
                    region_resolution_method=rm.method,
                )
            )

    if total_region_attempts:
        rate = resolved_regions / total_region_attempts
        notes.append(f"region resolution: {resolved_regions}/{total_region_attempts} "
                     f"({rate:.0%}) via name/point-in-polygon")
    if skipped:
        notes.append(f"skipped {skipped} row-values with no valid_date or no value_type signal")

    return pd.DataFrame(out_rows), skipped, notes


def _match_variable(label) -> Optional[str]:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    n = sm.normalize_header(str(label))
    for var, syns in sm.VARIABLE_SYNONYMS.items():
        if n in syns or any(sm._tok(s) == sm._tok(n) or sm._tok(s) <= sm._tok(n) for s in syns):
            return var.value
    return None


def _confidence_for(result: sm.MappingResult, source_column: str) -> float:
    for p in result.proposals:
        if p.source_column == source_column:
            return float(p.confidence or 0.0)
    return 0.0


def _load_profiles(session: Session):
    return crud.all_source_profiles(session)


def _store_raw(path: Path, batch_id: str, original_filename: str) -> Path:
    dest_dir = RAW_DIR / batch_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / original_filename
    shutil.copy2(path, dest)
    return dest


def _finish_canonicalization(
    session: Session,
    batch,
    parsed: ParsedTable,
    result: sm.MappingResult,
    verification_status: Optional[str] = None,
) -> IngestResult:
    canon_df, skipped, notes = to_canonical_rows(
        parsed.df,
        result,
        batch_id=batch.id,
        source_file=batch.original_filename,
        grain=parsed.grain,
    )
    if verification_status and not canon_df.empty:
        canon_df["verification_status"] = canon_df["value_type"].map(
            lambda vt: verification_status if vt == "observed" else None
        )
    n = parquet_store.append_batch(batch.id, canon_df)
    variables = sorted(canon_df["variable"].unique().tolist()) if not canon_df.empty else []

    crud.replace_column_mappings(session, batch.id, _mapping_rows_for_db(result))
    confirmed_map = {
        p.source_column: {
            "variable": p.suggested_variable,
            "value_type": p.suggested_value_type,
            "role": p.role,
            "unit_conversion": p.unit_conversion,
        }
        for p in result.proposals
        if p.decision in ("auto_accept", "confirmed") and p.role != sm.ROLE_UNMAPPED
    }
    profile = crud.upsert_source_profile(
        session,
        fingerprint=result.fingerprint,
        headers=parsed.headers,
        confirmed_mapping=confirmed_map,
        file_format=parsed.detected_format,
        layout=result.layout,
    )

    rate = 0.0
    if not canon_df.empty:
        rate = float((canon_df["region_id"].notna()).mean())

    all_notes = parsed.parse_notes + result.notes + notes
    crud.update_batch(
        session,
        batch,
        status="ingested",
        detected_format=parsed.detected_format,
        layout=result.layout,
        grain=parsed.grain,
        canonical_row_count=n,
        skipped_row_count=skipped,
        source_profile_id=profile.id,
        notes_json=all_notes,
    )
    return IngestResult(
        batch_id=batch.id,
        status="ingested",
        layout=result.layout,
        detected_format=parsed.detected_format,
        row_count_raw=batch.row_count_raw or 0,
        row_count_ingested=n,
        skipped_rows=skipped,
        canonical_variables_found=variables,
        region_resolution_rate=rate,
        profile_match=result.profile_match,
        notes=all_notes,
        should_retrain=n > 0,
    )


def ingest_upload(
    session: Session,
    path: Path | str,
    original_filename: Optional[str] = None,
    confirmed_mappings: Optional[list] = None,
    verification_status: Optional[str] = None,
) -> IngestResult:
    path = Path(path)
    original_filename = original_filename or path.name

    batch = crud.create_upload_batch(
        session,
        original_filename=original_filename,
        stored_path="",
        content_sha256=_sha256(path),
    )
    try:
        stored = _store_raw(path, batch.id, original_filename)
        crud.update_batch(session, batch, stored_path=str(stored))

        parsed = parse_upload(stored, original_filename)
        crud.update_batch(session, batch, row_count_raw=int(len(parsed.df)),
                          detected_format=parsed.detected_format, grain=parsed.grain)

        mapper = sm.SchemaMapper(filename_hint=original_filename)
        result = mapper.map_table(parsed.df, existing_profiles=_load_profiles(session))

        if confirmed_mappings is not None:
            _apply_confirmations(result, confirmed_mappings)

        if confirmed_mappings is None and not result.auto_accepted and result.profile_match != "exact":
            crud.replace_column_mappings(session, batch.id, _mapping_rows_for_db(result))
            crud.update_batch(session, batch, status="pending_confirmation",
                              layout=result.layout,
                              notes_json=parsed.parse_notes + result.notes)
            return IngestResult(
                batch_id=batch.id,
                status="pending_confirmation",
                layout=result.layout,
                detected_format=parsed.detected_format,
                row_count_raw=int(len(parsed.df)),
                profile_match=result.profile_match,
                mapping_proposals=[_proposal_to_dict(p) for p in result.proposals],
                notes=parsed.parse_notes + result.notes,
            )

        return _finish_canonicalization(session, batch, parsed, result, verification_status)

    except Exception as exc:  # noqa: BLE001 - record then re-raise
        crud.update_batch(session, batch, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


def confirm_mapping(session: Session, batch_id: str, mappings: list) -> IngestResult:
    batch = crud.get_upload_batch(session, batch_id)
    if batch is None:
        raise ValueError(f"unknown batch {batch_id}")

    parsed = parse_upload(Path(batch.stored_path), batch.original_filename)
    result = sm.SchemaMapper(filename_hint=batch.original_filename).map_table(
        parsed.df, existing_profiles=_load_profiles(session)
    )
    _apply_confirmations(result, mappings)
    crud.update_batch(session, batch, status="canonicalizing")
    return _finish_canonicalization(session, batch, parsed, result)


def _apply_confirmations(result: sm.MappingResult, mappings: list) -> None:
    by_col = {m["source_column"]: m for m in mappings}

    vt_col = result.value_type_column
    if vt_col is not None:
        m = by_col.get(vt_col)
        if m is not None and (m.get("role") == sm.ROLE_UNMAPPED
                              or m.get("variable") in (None, "", "unmapped")):
            result.value_type_column = None
            result.notes.append(
                f"value_type column '{vt_col}' was excluded by confirmation; "
                "value_type now comes from each column's confirmed mapping"
            )

    for p in result.proposals:
        m = by_col.get(p.source_column)
        if not m:
            if p.decision == "needs_confirmation":
                p.decision = "unmapped"
                p.role = sm.ROLE_UNMAPPED
            continue
        if m.get("role") == sm.ROLE_UNMAPPED or m.get("variable") in (None, "", "unmapped"):
            p.decision, p.role = "unmapped", sm.ROLE_UNMAPPED
            continue
        p.role = m.get("role", sm.ROLE_MEASUREMENT if not m.get("role") else p.role)
        if p.role == sm.ROLE_UNMAPPED:
            p.role = sm.ROLE_MEASUREMENT
        p.suggested_variable = m.get("variable", p.suggested_variable)
        p.suggested_value_type = m.get("value_type", p.suggested_value_type)
        p.unit_conversion = m.get("unit_conversion", p.unit_conversion)
        p.decision, p.method = "confirmed", "manual"

    accepted: dict = {}
    for p in result.proposals:
        if p.role == sm.ROLE_MEASUREMENT and p.decision in ("auto_accept", "confirmed") \
                and p.suggested_variable:
            accepted.setdefault(
                (p.suggested_variable, p.suggested_value_type), []
            ).append(p.source_column)
    for (variable, _vt), cols in accepted.items():
        if len(cols) > 1:
            result.notes.append(
                f"{len(cols)} columns ingested as {variable} ({', '.join(cols)}) - "
                "kept as distinct series (dedupe keys on source column)"
            )


def _main() -> None:
    ap = argparse.ArgumentParser(description="ingest one file into the canonical store")
    ap.add_argument("--file", required=True)
    ap.add_argument("--confirm-all", action="store_true",
                    help="accept every needs_confirmation proposal as-is (demo shortcut)")
    ap.add_argument("--reset", action="store_true", help="wipe metadata DB + canonical store first")
    args = ap.parse_args()

    if args.reset:
        from app.db.base import engine
        from app.db.models import Base
        Base.metadata.drop_all(engine)
        if parquet_store.CANONICAL_DIR.exists():
            shutil.rmtree(parquet_store.CANONICAL_DIR)

    init_db()
    src = Path(args.file)

    with get_session() as session:
        res = ingest_upload(session, src, src.name)
        if res.status == "pending_confirmation" and args.confirm_all:
            seen: dict = {}
            confirmations = []
            cands = sorted(
                (p for p in res.mapping_proposals
                 if p["role"] == "measurement" and p["decision"] == "needs_confirmation"
                 and p["suggested_variable"]),
                key=lambda p: p["confidence"], reverse=True,
            )
            for p in cands:
                key = (p["suggested_variable"], p["suggested_value_type"])
                if key in seen:
                    continue
                seen[key] = p["source_column"]
                confirmations.append({
                    "source_column": p["source_column"],
                    "variable": p["suggested_variable"],
                    "value_type": p["suggested_value_type"],
                    "unit_conversion": p["unit_conversion"],
                })
            res = confirm_mapping(session, res.batch_id, confirmations)

    print(f"\nbatch {res.batch_id}  status={res.status}  layout={res.layout}  format={res.detected_format}")
    print(f"raw rows={res.row_count_raw}  canonical rows={res.row_count_ingested}  skipped={res.skipped_rows}")
    print(f"variables: {res.canonical_variables_found}")
    print(f"region resolution rate: {res.region_resolution_rate:.0%}   profile match: {res.profile_match}")
    for n in res.notes:
        print("  -", n)
    if res.status == "pending_confirmation":
        print("\nPROPOSALS NEEDING CONFIRMATION:")
        for p in res.mapping_proposals:
            if p["decision"] == "needs_confirmation":
                print(f"  {p['source_column']:22} -> {p['suggested_variable']} "
                      f"(vt={p['suggested_value_type']}, conf={p['confidence']:.2f})  "
                      f"samples={p['sample_values'][:3]}")
    else:
        print("\ncanonical store summary:")
        for k, v in parquet_store.dataset_summary().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(_main())
