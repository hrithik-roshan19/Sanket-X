from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ColumnMapping, SourceProfile, UploadBatch


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_upload_batch(
    session: Session,
    *,
    original_filename: str,
    stored_path: str,
    content_sha256: str,
) -> UploadBatch:
    batch = UploadBatch(
        original_filename=original_filename,
        stored_path=stored_path,
        content_sha256=content_sha256,
        status="received",
    )
    session.add(batch)
    session.flush()
    return batch


def get_upload_batch(session: Session, batch_id: str) -> UploadBatch | None:
    return session.get(UploadBatch, batch_id)


def update_batch(session: Session, batch: UploadBatch, **fields: Any) -> UploadBatch:
    for key, value in fields.items():
        setattr(batch, key, value)
    batch.updated_at = _now()
    session.add(batch)
    session.flush()
    return batch


def replace_column_mappings(
    session: Session, batch_id: str, rows: Iterable[dict]
) -> list[ColumnMapping]:
    session.query(ColumnMapping).filter(ColumnMapping.upload_batch_id == batch_id).delete()
    created: list[ColumnMapping] = []
    for row in rows:
        cm = ColumnMapping(upload_batch_id=batch_id, **row)
        session.add(cm)
        created.append(cm)
    session.flush()
    return created


def get_column_mappings(session: Session, batch_id: str) -> list[ColumnMapping]:
    stmt = select(ColumnMapping).where(ColumnMapping.upload_batch_id == batch_id)
    return list(session.scalars(stmt))


def find_source_profile(session: Session, fingerprint: str) -> SourceProfile | None:
    stmt = select(SourceProfile).where(SourceProfile.fingerprint == fingerprint)
    return session.scalars(stmt).first()


def all_source_profiles(session: Session) -> list[SourceProfile]:
    return list(session.scalars(select(SourceProfile)))


def find_similar_source_profiles(
    session: Session, header_set: Sequence[str], min_jaccard: float = 0.6
) -> list[tuple[SourceProfile, float]]:
    target = set(header_set)
    out: list[tuple[SourceProfile, float]] = []
    for profile in all_source_profiles(session):
        stored = set(profile.header_list_json or [])
        if not stored:
            continue
        union = stored | target
        j = len(stored & target) / len(union) if union else 0.0
        if j >= min_jaccard:
            out.append((profile, j))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def upsert_source_profile(
    session: Session,
    *,
    fingerprint: str,
    headers: Sequence[str],
    confirmed_mapping: dict,
    file_format: str | None,
    layout: str | None,
) -> SourceProfile:
    profile = find_source_profile(session, fingerprint)
    if profile is None:
        profile = SourceProfile(
            fingerprint=fingerprint,
            header_list_json=list(headers),
            confirmed_mapping_json=confirmed_mapping,
            file_format=file_format,
            layout=layout,
        )
        session.add(profile)
    else:
        profile.confirmed_mapping_json = confirmed_mapping
        profile.header_list_json = list(headers)
        profile.file_format = file_format
        profile.layout = layout
        profile.times_seen += 1
        profile.last_seen_at = _now()
    session.flush()
    return profile
