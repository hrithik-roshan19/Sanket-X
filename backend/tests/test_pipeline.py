"""End-to-end ingestion against the real samples."""

from __future__ import annotations

import pandas as pd
import pytest

from app.db import crud
from app.ingestion.pipeline import confirm_mapping, ingest_upload
from app.storage import parquet_store
from tests.conftest import ERA5_CSV, GEFS_CSV, find_sample


def _slice_csv(src, tmp_path, n):
    df = pd.read_csv(src, nrows=n)
    dst = tmp_path / src.name
    df.to_csv(dst, index=False)
    return dst


def _confirm_all(res):
    return [
        {"source_column": p["source_column"], "variable": p["suggested_variable"],
         "value_type": p["suggested_value_type"], "unit_conversion": p["unit_conversion"]}
        for p in res.mapping_proposals
        if p["role"] == "measurement" and p["decision"] == "needs_confirmation"
        and p["suggested_variable"]
    ]


def test_gefs_then_era5_ingest_and_join(session, tmp_path):
    gefs = _slice_csv(GEFS_CSV, tmp_path, 4000)
    era5 = _slice_csv(ERA5_CSV, tmp_path, 4000)

    r1 = ingest_upload(session, gefs, gefs.name)
    if r1.status == "pending_confirmation":
        # dedupe collisions: keep first per (variable, value_type)
        seen, conf = set(), []
        for c in sorted(_confirm_all(r1), key=lambda x: x["source_column"]):
            key = (c["variable"], c["value_type"])
            if key not in seen:
                seen.add(key); conf.append(c)
        r1 = confirm_mapping(session, r1.batch_id, conf)
    assert r1.status == "ingested"
    assert r1.row_count_ingested > 0
    assert "forecast" not in r1.canonical_variables_found  # it's a list of variables
    assert r1.region_resolution_rate > 0.8

    r2 = ingest_upload(session, era5, era5.name)
    if r2.status == "pending_confirmation":
        r2 = confirm_mapping(session, r2.batch_id, _confirm_all(r2))
    assert r2.status == "ingested"

    df = parquet_store.read_dataset(variables=["temperature_c"])
    f = df[df.value_type == "forecast"]
    o = df[df.value_type == "observed"][["region_id", "valid_date", "value"]]
    joined = f.merge(o, on=["region_id", "valid_date"], suffixes=("_fc", "_obs"))
    assert len(joined) > 100
    err = (joined["value_fc"] - joined["value_obs"]).abs()
    assert 0.2 < err.mean() < 8.0  # sane GEFS-vs-ERA5 2m-temp error band


def test_forecast_rows_have_lead_and_init(session, tmp_path):
    gefs = _slice_csv(GEFS_CSV, tmp_path, 2000)
    r = ingest_upload(session, gefs, gefs.name)
    if r.status == "pending_confirmation":
        seen, conf = set(), []
        for c in _confirm_all(r):
            key = (c["variable"], c["value_type"])
            if key not in seen:
                seen.add(key); conf.append(c)
        r = confirm_mapping(session, r.batch_id, conf)

    df = parquet_store.read_dataset()
    fc = df[df.value_type == "forecast"]
    assert fc["lead_time_days"].between(1, 10).all()
    assert fc["init_date"].notna().all()
    # valid_date = init_date + (lead - 1): lead day 1 is the init day itself (equal, not
    # strictly after), lead day k is k-1 days later. See test_live.py's dedicated check
    # and scripts/fix_forecast_valid_date_offset.py for why the -1 is load-bearing.
    offset_days = (pd.to_datetime(fc["valid_date"]) - pd.to_datetime(fc["init_date"])).dt.days
    assert (offset_days == fc["lead_time_days"] - 1).all()


def test_reupload_hits_source_profile(session, tmp_path):
    era5 = _slice_csv(ERA5_CSV, tmp_path, 1500)
    r1 = ingest_upload(session, era5, era5.name)
    if r1.status == "pending_confirmation":
        r1 = confirm_mapping(session, r1.batch_id, _confirm_all(r1))
    assert r1.status == "ingested"
    assert len(crud.all_source_profiles(session)) == 1

    r2 = ingest_upload(session, era5, era5.name)
    assert r2.status == "ingested"          # auto-applied, no confirmation needed
    assert r2.profile_match == "exact"

    # Each ingest writes its own partition: re-ingesting never overwrites the first.
    assert len(list(parquet_store.CANONICAL_DIR.glob("batch_id=*/part-*.parquet"))) == 2
    assert parquet_store.read_dataset(dedupe=False)["upload_batch_id"].nunique() == 2

    # ...but the same file twice is the same readings twice, and a deduplicated read must
    # not double-count them: a duplicated observation would double every forecast row it
    # merges with in feature engineering, silently reweighting training.
    deduped = parquet_store.read_dataset()
    raw = parquet_store.read_dataset(dedupe=False)
    assert len(deduped) == len(raw) // 2
    assert not deduped.duplicated(subset=parquet_store._DEDUPE_KEY).any()

    # Two source columns mapping onto one canonical variable is NOT duplication: this file
    # carries both mslp_hpa and psfc_hpa and _confirm_all accepts both as pressure_hpa.
    # Deduplication must leave those alone - picking one silently would discard a real
    # measurement. Disambiguating them is the schema mapper's job (it flags the collision).
    pressure = deduped[deduped["variable"] == "pressure_hpa"]
    assert set(pressure["source_column"]) == {"mslp_hpa", "psfc_hpa"}


def test_row_without_valid_date_is_skipped_not_dropped(session, tmp_path):
    df = pd.DataFrame({
        "region": ["Bihar", "Bihar", "Odisha", "Kerala"],
        "valid_date": ["2019-06-01", None, "2019-06-03", "2019-06-04"],
        "temp_c": [30.1, 29.4, 31.0, 28.7],
        "data_type": ["observed", "observed", "observed", "observed"],
    })
    src = tmp_path / "tiny.csv"
    df.to_csv(src, index=False)
    r = ingest_upload(session, src, src.name)
    if r.status == "pending_confirmation":
        r = confirm_mapping(session, r.batch_id, _confirm_all(r))
    assert r.status == "ingested"
    assert r.row_count_ingested == 3           # the dateless row is not canonicalised
    assert r.skipped_rows == 1
    assert any("skipped" in n.lower() for n in r.notes)


def test_avg_max_min_collision_is_surfaced_not_silent(session, tmp_path):
    # avg/max/min all claim temperature_c. The mapper routes this to confirmation (the UI
    # is where an accidental 3-onto-1 gets pared to one column). If a client confirms all
    # three anyway they are kept as distinct series - the store dedupes on source_column -
    # but the pipeline must say so in the notes rather than swallow it silently.
    df = pd.DataFrame({
        "region": ["Bihar", "Bihar", "Odisha", "Odisha"],
        "valid_date": ["2019-06-01", "2019-06-02", "2019-06-01", "2019-06-02"],
        "Temp_Avg_C": [30.1, 29.4, 31.0, 30.2],
        "Temp_Max_C": [36.2, 35.1, 37.4, 36.0],
        "Temp_Min_C": [24.0, 23.6, 25.1, 24.4],
        "data_type": ["observed"] * 4,
    })
    src = tmp_path / "triple_temp.csv"
    df.to_csv(src, index=False)

    r = ingest_upload(session, src, src.name)
    assert r.status == "pending_confirmation"            # collision -> confirmation
    colliding = [
        {"source_column": c, "variable": "temperature_c", "value_type": "observed",
         "unit_conversion": None}
        for c in ("Temp_Avg_C", "Temp_Max_C", "Temp_Min_C")
    ]
    r = confirm_mapping(session, r.batch_id, colliding)
    assert r.status == "ingested"

    temp = parquet_store.read_dataset(variables=["temperature_c"])
    assert set(temp["source_column"]) == {"Temp_Avg_C", "Temp_Max_C", "Temp_Min_C"}
    assert any("temperature_c" in n and "distinct series" in n for n in r.notes)


def test_power_monthly_ingests_as_monthly_grain(session):
    path = find_sample("POWER_Regional_Monthly")
    if path is None:
        pytest.skip("no NASA POWER monthly file available")
    r = ingest_upload(session, path, path.name)
    if r.status == "pending_confirmation":
        r = confirm_mapping(session, r.batch_id, _confirm_all(r))
    assert r.status == "ingested"
    df = parquet_store.read_dataset()
    assert (df["grain"] == "monthly").all()
    assert (pd.to_datetime(df["valid_date"]).dt.day == 1).all()


def test_power_daily_doy_dates_and_gridded_resolution(session):
    path = find_sample("POWER_Regional_Daily")
    if path is None:
        pytest.skip("no NASA POWER daily file available")
    r = ingest_upload(session, path, path.name)
    if r.status == "pending_confirmation":
        r = confirm_mapping(session, r.batch_id, _confirm_all(r))
    assert r.status == "ingested"
    df = parquet_store.read_dataset(variables=["temperature_c"])
    assert df["value_type"].eq("observed").all()
    assert pd.to_datetime(df["valid_date"]).dt.year.between(2000, 2035).all()


def test_summary_refuses_to_deduplicate_a_large_store_on_the_serving_box(session, tmp_path, monkeypatch):
    """The store summary is computed during packaging and shipped beside the data. If that
    sidecar is ever missing on the serving box, the fallback used to deduplicate the whole
    store to answer /api/model/status - ~300 MB on the current store, and measured at
    918 MB peak on the backfilled one, inside a 512 MB container that gets killed rather
    than throttled.

    Past a generous row count the fallback now refuses and says why. Asserted both ways:
    that it returns the explanation, and that it never reaches the expensive path -
    checking the return value alone would pass even if the computation still ran.
    """
    from app.storage import parquet_store as ps

    src = _slice_csv(ERA5_CSV, tmp_path, 200)
    r = ingest_upload(session, src, src.name)
    if r.status == "pending_confirmation":
        r = confirm_mapping(session, r.batch_id, _confirm_all(r))
    assert r.status == "ingested" and r.row_count_ingested > 0

    called = []
    monkeypatch.setattr(ps, "_compute_summary", lambda: called.append(1) or {})
    monkeypatch.setattr(ps, "_SUMMARY_COMPUTE_MAX_ROWS", 1)

    out = ps._summary_without_cache()
    assert not called, "the guard must refuse before computing, not after"
    assert "memory" in out["unavailable_reason"]
    # The shape the API and the page expect still comes back, with honest blanks.
    for k in ("total_rows", "regions", "forecast_cycles", "valid_date_min"):
        assert k in out and out[k] is None

    # Under the threshold it computes as before - the guard is not a permanent downgrade.
    monkeypatch.setattr(ps, "_SUMMARY_COMPUTE_MAX_ROWS", 10_000_000)
    ps._summary_without_cache()
    assert called, "a small store must still be summarised normally"


def test_compaction_removes_superseded_rows_without_changing_what_is_read(session, tmp_path):
    """Re-ingesting an observation window writes a second copy rather than replacing the
    first, so the store accumulated ~25k dead rows a day. They were always collapsed at
    read time, so nothing was wrong - but they were paid for in every read, in an artifact
    pulled on every container start, and in the memory of a 512 MB box.

    Compaction drops them. The bar is that nothing a reader can see changes, so this
    asserts the read result is identical before and after - not merely that rows went
    down, which a function that deleted the wrong rows would also satisfy.
    """
    import pandas as pd

    from app.storage import parquet_store as ps

    src = _slice_csv(ERA5_CSV, tmp_path, 300)
    for name in ("first", "second"):
        r = ingest_upload(session, src, f"{name}-{src.name}")
        if r.status == "pending_confirmation":
            r = confirm_mapping(session, r.batch_id, _confirm_all(r))
        assert r.status == "ingested"

    on_disk = ps._row_count_on_disk()
    before = ps.read_dataset().sort_values(ps._DEDUPE_KEY).reset_index(drop=True)
    assert on_disk > len(before), "the same window ingested twice should leave dead rows"

    stats = ps.compact_store()
    after = ps.read_dataset().sort_values(ps._DEDUPE_KEY).reset_index(drop=True)

    assert stats["removed"] == on_disk - len(before)
    assert ps._row_count_on_disk() == len(before), "disk should now hold only live rows"
    pd.testing.assert_frame_equal(before, after, check_like=True, check_dtype=False)

    # Idempotent: a compacted store has nothing left to remove.
    assert ps.compact_store()["removed"] == 0
