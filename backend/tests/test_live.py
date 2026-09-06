"""Phase 6 (live ingestion) tests.

Deliberately offline: nothing here touches NOAA or Open-Meteo. The network-dependent
behaviour is exercised by the fetch scripts themselves; what matters to protect here is the
logic that decides *what* to fetch, *how* samples are grouped into days, and the guards
that keep live data consistent with what the models were trained on.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.config import settings
from app.ingestion.canonical_schema import CANONICAL_COLUMNS
from app.live import gefs
from app.live.orchestrator import FORECAST_MAPPINGS, OBSERVATION_MAPPINGS, cycle_target
from app.storage import parquet_store


# ---------------------------------------------------------------- cycle selection

def test_latest_expected_cycle_respects_publication_lag():
    """A cycle is only considered available once the publication lag has elapsed."""
    # 12:44 UTC with a 5.5 h lag: 06Z (issued 6.7 h ago) is ready, 12Z is not.
    now = datetime(2026, 8, 29, 12, 44, tzinfo=timezone.utc)
    assert gefs.latest_expected_cycle(now, lag_hours=5.5) == (date(2026, 8, 29), "06")

    # Just after midnight, the newest ready cycle is the previous day's 18Z.
    now = datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc)
    assert gefs.latest_expected_cycle(now, lag_hours=5.5) == (date(2026, 8, 28), "18")


def test_recent_cycles_walks_backwards_without_repeats():
    now = datetime(2026, 8, 29, 12, 44, tzinfo=timezone.utc)
    cycles = gefs.recent_cycles(5, now, lag_hours=5.5, stride_hours=24)
    assert len(cycles) == len(set(cycles)) == 5
    assert cycles[0] == (date(2026, 8, 29), "06")
    assert cycles[-1] == (date(2026, 8, 25), "06")


def test_cycle_target_is_hour_specific():
    """00Z and 06Z of one day are different cycles; a date alone cannot separate them."""
    assert cycle_target(date(2026, 8, 29), "00") != cycle_target(date(2026, 8, 29), "06")


# ---------------------------------------------------------------- day grouping

def test_00z_lead_days_reproduce_the_training_windows():
    """For a 00Z cycle, lead day k must be exactly forecast hours ((k-1)*24, k*24].

    This is the window the training data was built from, so any drift here would put live
    forecasts on a different footing from the data the regressors learned on.
    """
    init = datetime(2026, 8, 29, 0, tzinfo=timezone.utc)
    by_day = gefs._steps_by_day(init, gefs.STEP_HOURS)
    for k in range(1, 11):
        assert by_day[k] == list(range((k - 1) * 24 + 3, k * 24 + 1, 3)), f"lead {k}"


def test_valid_date_is_init_plus_lead_minus_one():
    """The convention fixed on 2026-08-29: lead day 1 is the init day itself for a 00Z run.

    Labelling it init+lead instead put every forecast a day after its own contents and
    verified it against the wrong day's observation.
    """
    init = datetime(2026, 8, 29, 0, tzinfo=timezone.utc)
    for fh in (3, 24):                     # first and last sample of lead day 1
        day, lead = gefs._lead_day_of(init, fh)
        assert lead == 1
        assert day == init.date() + timedelta(days=lead - 1) == date(2026, 8, 29)
    day, lead = gefs._lead_day_of(init, 240)
    assert (lead, day) == (10, date(2026, 9, 7))


def test_non_00z_cycles_group_by_calendar_day():
    """A 06Z run's lead days must still be whole calendar days.

    Observations are calendar-day aggregates, so a lead-hour window (which for a 06Z run
    straddles midnight) could never line up with the observation verifying it.
    """
    init = datetime(2026, 8, 29, 6, tzinfo=timezone.utc)
    by_day = gefs._steps_by_day(init, gefs.STEP_HOURS)
    complete = [k for k, hrs in by_day.items() if len(hrs) == 8 and 1 <= k <= 10]
    # Lead 1 is partial (the run starts 6 h into that day) and is dropped; 10 exceeds +240h.
    assert complete == list(range(2, 11))
    for k in complete:
        first, last = by_day[k][0], by_day[k][-1]
        assert (init + timedelta(hours=last)).date() - timedelta(days=0)
        # every sample of a lead day belongs to that same calendar day
        days = {gefs._lead_day_of(init, h)[0] for h in by_day[k]}
        assert len(days) == 1


def test_accumulation_uses_non_overlapping_buckets_only():
    """APCP buckets reset every 6 h, so the 3-hourly steps OVERLAP (0-3 and 0-6 both
    include hours 0-3). Summing every 3-hourly value would roughly double daily rainfall.
    A day must be tiled by exactly four 6-hourly buckets."""
    assert gefs.VAR_SPEC["apcp_mm"]["step_multiple"] == 6
    assert gefs.VAR_SPEC["apcp_mm"]["agg"] == "sum"
    init = datetime(2026, 8, 29, 0, tzinfo=timezone.utc)
    buckets = gefs._steps_by_day(init, 6)
    assert buckets[1] == [6, 12, 18, 24]
    for k in range(1, 11):
        assert len(buckets[k]) == gefs._expected_samples(6) == 4


# ---------------------------------------------------------------- consistency guards

def test_member_set_matches_what_the_models_were_trained_on():
    """Operational GEFS has 31 members; the reforecast the models learned from has 5.

    Ensemble spread grows with member count and spread_* are classifier inputs, so scoring
    a 31-member spread with a 5-member-trained model would bias its strongest features.
    """
    assert len(settings.gefs_member_list) == 5
    assert settings.gefs_member_list[0] == "gec00"


def test_only_mslp_becomes_canonical_pressure():
    """Both live files carry mslp_hpa and psfc_hpa; only mean-sea-level pressure is the
    canonical pressure variable. Surface pressure is dominated by elevation (Leh reads
    ~684 hPa), so mixing them would make the pressure model meaningless."""
    for mappings in (FORECAST_MAPPINGS, OBSERVATION_MAPPINGS):
        by_col = {m["source_column"]: m for m in mappings}
        assert by_col["mslp_hpa"]["variable"] == "pressure_hpa"
        assert by_col["psfc_hpa"].get("role") == "unmapped"
        assert "variable" not in by_col["psfc_hpa"]


def test_live_mappings_cover_every_measurement_column():
    """Live ingestion passes explicit mappings rather than trusting the schema mapper, so
    an unattended job cannot silently mis-map a column. Every canonical variable the
    models expect must therefore be present."""
    expected = {
        "temperature_c", "humidity_pct", "rainfall_mm", "pressure_hpa",
        "atmospheric_moisture_kgm2", "soil_moisture_pct",
        "wind_speed_ms", "wind_direction_deg",
    }
    for mappings, vt in ((FORECAST_MAPPINGS, "forecast"), (OBSERVATION_MAPPINGS, "observed")):
        mapped = {m["variable"] for m in mappings if m.get("variable")}
        assert mapped == expected
        assert all(m["value_type"] == vt for m in mappings if m.get("variable"))


# ---------------------------------------------------------------- storage behaviour

def test_verification_status_is_part_of_the_canonical_schema():
    assert "verification_status" in CANONICAL_COLUMNS
    assert "verification_status" in parquet_store.ARROW_SCHEMA.names


def _row(**kw):
    base = dict(
        record_id="r", upload_batch_id="b", source_file="f.csv", source_column="c",
        variable="temperature_c", value_type="observed", value=1.0,
        region_id="IN-MH", region_name="Maharashtra", lat=19.0, lon=72.0,
        init_date=None, valid_date=date(2026, 8, 1), lead_time_days=None,
        ensemble_member_id=None, mapping_confidence=1.0,
        ingested_at=pd.Timestamp("2026-08-29T00:00:00"),
        grain="native", region_resolution_method="name", verification_status=None,
    )
    base.update(kw)
    return base


def test_final_observation_supersedes_provisional(fresh_store):
    """The revision path: ERA5 landing later must replace the near-real-time value for the
    same reading, not sit alongside it - a duplicate observation would double every
    forecast row it merges with in feature engineering."""
    parquet_store.append_batch("prov", [_row(
        upload_batch_id="prov", value=30.0, verification_status="provisional",
        ingested_at=pd.Timestamp("2026-08-29T00:00:00"))])
    parquet_store.append_batch("fin", [_row(
        upload_batch_id="fin", value=28.5, verification_status="final",
        ingested_at=pd.Timestamp("2026-08-30T00:00:00"))])

    df = parquet_store.read_dataset()
    assert len(df) == 1
    assert df.iloc[0]["value"] == 28.5
    assert df.iloc[0]["verification_status"] == "final"
    # both partitions still exist on disk; deduplication happens on read
    assert len(parquet_store.read_dataset(dedupe=False)) == 2


def test_final_wins_even_when_it_arrives_first(fresh_store):
    """Precedence is by authority, not arrival order: a provisional refresh that runs
    after ERA5 has already landed must not overwrite the settled value."""
    parquet_store.append_batch("fin", [_row(
        upload_batch_id="fin", value=28.5, verification_status="final",
        ingested_at=pd.Timestamp("2026-08-29T00:00:00"))])
    parquet_store.append_batch("prov", [_row(
        upload_batch_id="prov", value=30.0, verification_status="provisional",
        ingested_at=pd.Timestamp("2026-08-30T00:00:00"))])

    df = parquet_store.read_dataset()
    assert len(df) == 1 and df.iloc[0]["value"] == 28.5


def test_exclude_provisional_keeps_legacy_null_rows(fresh_store):
    """Rows ingested before Phase 6 carry a NULL status and came from the ERA5 archive,
    so they are already final. Filtering with a bare `!= 'provisional'` would drop them:
    in Arrow, as in SQL, comparing a null yields null rather than true."""
    parquet_store.append_batch("legacy", [_row(
        upload_batch_id="legacy", verification_status=None,
        valid_date=date(2026, 8, 1))])
    parquet_store.append_batch("prov", [_row(
        upload_batch_id="prov", verification_status="provisional",
        valid_date=date(2026, 8, 2))])

    kept = parquet_store.read_dataset(exclude_provisional=True)
    assert len(kept) == 1
    assert kept.iloc[0]["upload_batch_id"] == "legacy"


def test_has_forecast_cycle_and_latest_init_date(fresh_store):
    assert parquet_store.latest_forecast_init_date() is None
    parquet_store.append_batch("fc", [
        _row(upload_batch_id="fc", value_type="forecast", init_date=date(2026, 8, 29),
             valid_date=date(2026, 8, 30), lead_time_days=2, ensemble_member_id="c00"),
    ])
    assert parquet_store.has_forecast_cycle(date(2026, 8, 29))
    assert not parquet_store.has_forecast_cycle(date(2026, 8, 28))
    assert parquet_store.latest_forecast_init_date() == date(2026, 8, 29)


def test_scheduler_is_off_unless_explicitly_enabled():
    """A fresh clone, a test run or CI must never start reaching out to NOAA on import."""
    from app.live.scheduler import scheduler
    assert settings.live_ingest_enabled is False
    scheduler.start()
    assert scheduler.running is False


# ---------------------------------------------------------------- feature engineering

def test_rate_of_change_survives_a_single_verified_lead_day():
    """A live cycle verifies one lead day at a time, so early on every group holds exactly
    one row. pandas' groupby.apply returns a DataFrame rather than a Series in that case,
    which used to crash feature engineering during precisely the window a fresh cycle is
    in for its first days."""
    from app.features import engineering as fe

    rows = []
    for region in ("IN-MH", "IN-KL"):
        for member in ("c00", "p01"):
            rows.append(dict(
                region_id=region, variable="pressure_hpa", value_type="forecast",
                value=1008.0, init_date=pd.Timestamp("2026-08-27"),
                valid_date=pd.Timestamp("2026-08-28"), lead_time_days=2,
                ensemble_member_id=member, verification_status=None,
            ))
        rows.append(dict(
            region_id=region, variable="pressure_hpa", value_type="observed",
            value=1007.0, init_date=pd.NaT, valid_date=pd.Timestamp("2026-08-28"),
            lead_time_days=None, ensemble_member_id=None,
            verification_status="provisional",
        ))
    frame = fe.build_training_frame(pd.DataFrame(rows), require_observed=True)

    assert not frame.empty
    assert sorted(frame["lead_time_days"].unique()) == [2]
    # No second lead to difference against, so the feature is absent - not an exception.
    assert frame["pressure_rate_of_change"].isna().all()


def test_verification_status_reaches_the_training_frame():
    """The UI badges provisional observations, so the status has to survive the
    forecast/observation merge. Both sides carry the column, and without dropping it from
    the forecast side pandas suffixes both to _x/_y and the real one disappears."""
    from app.features import engineering as fe

    rows = []
    for lead, status in ((2, "provisional"), (3, "final")):
        vd = pd.Timestamp("2026-08-26") + pd.Timedelta(days=lead)
        rows.append(dict(
            region_id="IN-MH", variable="temperature_c", value_type="forecast",
            value=30.0, init_date=pd.Timestamp("2026-08-26"), valid_date=vd,
            lead_time_days=lead, ensemble_member_id="c00", verification_status=None,
        ))
        rows.append(dict(
            region_id="IN-MH", variable="temperature_c", value_type="observed",
            value=29.0, init_date=pd.NaT, valid_date=vd, lead_time_days=None,
            ensemble_member_id=None, verification_status=status,
        ))
    frame = fe.build_training_frame(pd.DataFrame(rows), require_observed=True)

    assert "verification_status" in frame.columns
    got = dict(zip(frame["lead_time_days"], frame["verification_status"]))
    assert got == {2: "provisional", 3: "final"}


def test_legacy_null_status_is_treated_as_final():
    """Observations ingested before Phase 6 carry NULL and came from the ERA5 archive, so
    they are already settled. They must not be mistaken for provisional and withheld."""
    from app.features import engineering as fe

    rows = [
        dict(region_id="IN-MH", variable="temperature_c", value_type="forecast",
             value=30.0, init_date=pd.Timestamp("2019-12-11"),
             valid_date=pd.Timestamp("2019-12-12"), lead_time_days=2,
             ensemble_member_id="c00", verification_status=None),
        dict(region_id="IN-MH", variable="temperature_c", value_type="observed",
             value=29.0, init_date=pd.NaT, valid_date=pd.Timestamp("2019-12-12"),
             lead_time_days=None, ensemble_member_id=None, verification_status=None),
    ]
    frame = fe.build_training_frame(pd.DataFrame(rows), require_observed=True)
    assert frame["verification_status"].tolist() == ["final"]


# ------------------------------------------------- cycle completeness (publish guard)
# A slow NOAA pull returns a cycle that is real but thin: some GRIB steps time out and the
# day's aggregate is built from fewer samples. These guard the decision about whether such
# a cycle is fit to publish, because the failure is silent - the numbers look ordinary.

def _report(**kw) -> gefs.FetchReport:
    base = dict(init_date=date(2026, 8, 31), cycle_hour="00", members=["gec00"],
                steps_expected=400, steps_fetched=400)
    base.update(kw)
    return gefs.FetchReport(**base)


def test_only_accumulations_are_treated_as_sum_variables():
    # If another summed variable is ever added, the guard must widen with it - hence
    # deriving the set from VAR_SPEC rather than hard-coding "apcp_mm" in two places.
    assert gefs.SUM_VARIABLES == {"apcp_mm"}
    assert all(gefs.VAR_SPEC[v]["agg"] == "sum" for v in gefs.SUM_VARIABLES)


def test_step_completeness_reports_the_fetched_fraction():
    assert _report(steps_fetched=344).step_completeness == pytest.approx(0.86)
    assert _report(steps_fetched=400).step_completeness == 1.0
    # a cycle that never started must not divide by zero
    assert _report(steps_expected=0, steps_fetched=0).step_completeness == 0.0


def test_undersampled_sums_singles_out_short_accumulations():
    r = _report(undersampled=[
        "gep01/D1/t2m_c(7/8)",        # a mean: noisier, same quantity
        "gep01/D2/apcp_mm(2/4)",      # a sum: roughly half the real rainfall
        "gep02/D3/rh2m_pct(6/8)",
    ])
    assert r.undersampled_sums == ["gep01/D2/apcp_mm(2/4)"]
    assert len(r.undersampled) == 3


def test_a_member_named_after_a_variable_is_not_mistaken_for_one():
    # Matching on "/<var>(" keys off the variable segment; a bare substring search would
    # misread a member or day segment that happened to contain the name.
    r = _report(undersampled=["apcp_mm_member/D1/t2m_c(7/8)"])
    assert r.undersampled_sums == []


def test_complete_requires_full_steps_and_no_undersampling():
    assert _report().complete is True
    assert _report(steps_fetched=399).complete is False
    assert _report(undersampled=["gep01/D2/apcp_mm(2/4)"]).complete is False


# ------------------------------------------------------- the publish/refuse decision

class _Args:
    """Stand-in for the argparse namespace refresh_for_deploy builds."""
    def __init__(self, min_steps=0.98, allow_short_accumulations=False):
        self.min_steps = min_steps
        self.allow_short_accumulations = allow_short_accumulations


def _reject(result, **kw):
    from scripts.refresh_for_deploy import _reject_reason
    return _reject_reason(result, _Args(**kw))


# The shape run_forecast_cycle actually returns on a clean pull.
_GOOD = {"status": "complete", "target": "2026-08-30 18", "step_completeness": 1.0,
         "steps_fetched": 400, "steps_expected": 400,
         "undersampled_count": 0, "undersampled_sum_count": 0,
         "missing_steps": [], "undersampled": []}

# The real payload from the 2026-08-31 00Z run, which was published before this guard
# existed: 344/400 steps and rainfall summed from 2 of 4 samples on some members.
_THIN = {"status": "complete", "target": "2026-08-31 00", "step_completeness": 0.86,
         "steps_fetched": 344, "steps_expected": 400,
         "undersampled_count": 56, "undersampled_sum_count": 1,
         "missing_steps": ["gep01/f003"] * 56,
         "undersampled": ["gep01/D1/t2m_c(7/8)", "gep01/D2/apcp_mm(2/4)"]}


def test_a_complete_cycle_is_published():
    assert _reject(_GOOD) is None


def test_the_cycle_that_slipped_through_is_now_refused():
    reason = _reject(_THIN)
    assert reason is not None and "86.0%" in reason


def test_a_short_accumulation_alone_is_enough_to_refuse():
    # Full step count, but one day's rainfall summed from half its samples. Rainfall drives
    # most busts, so publishing this would understate risk on the variable that matters.
    nearly = {**_GOOD, "undersampled_count": 1, "undersampled_sum_count": 1,
              "undersampled": ["gep02/D4/apcp_mm(2/4)"]}
    reason = _reject(nearly)
    assert reason is not None and "understates rainfall" in reason
    assert "gep02/D4/apcp_mm(2/4)" in reason


def test_undersampled_means_alone_do_not_refuse():
    # A mean over 7 of 8 samples is the same quantity, measured a little more noisily.
    means_only = {**_GOOD, "undersampled_count": 3, "undersampled_sum_count": 0,
                  "undersampled": ["gep01/D1/t2m_c(7/8)"]}
    assert _reject(means_only) is None


def test_flags_can_force_a_thin_cycle_through():
    assert _reject(_THIN, min_steps=0, allow_short_accumulations=True) is None


def test_a_skipped_cycle_is_not_a_rejection():
    # Nothing was ingested, so there is nothing to refuse - the previous artifact stands.
    assert _reject({"status": "skipped", "target": "2026-08-31 06",
                    "reason": "already ingested"}) is None


def test_an_unknown_completeness_is_not_treated_as_a_bad_cycle():
    # Absent counts mean the orchestrator predates the guard, not that the cycle is thin.
    assert _reject({"status": "complete", "target": "x"}) is None


def test_completeness_is_derived_when_only_the_raw_counts_are_present():
    assert _reject({"status": "complete", "target": "x",
                    "steps_fetched": 344, "steps_expected": 400}) is not None


def test_the_example_is_honest_when_the_truncated_list_holds_none():
    # `undersampled` is truncated to 20 by the orchestrator, so a non-zero count can come
    # with no matching example. It must say so rather than print an empty "e.g.".
    truncated = {**_GOOD, "undersampled_count": 40, "undersampled_sum_count": 2,
                 "undersampled": ["gep01/D1/t2m_c(7/8)"] * 20}
    reason = _reject(truncated)
    assert reason is not None and "not in the truncated sample" in reason


# --------------------------------------------------- filling NOMADS gaps from S3
# NOMADS answers a burst it dislikes with a 302 to its throttle page, and the window
# outlasts the retry ladder - so a mid-pull throttle leaves contiguous holes that only a
# second transport can fill. These stay offline: `_fetch_step_s3` and `_extract_points`
# are stubbed, because what is being protected is the bookkeeping, not the download.

def _stub_s3(monkeypatch, recoverable):
    """S3 serves the steps in `recoverable`; everything else 404s as it would for real."""
    def fake_fetch(init, hh, member, fh):
        if (member, fh) not in recoverable:
            raise FileNotFoundError(f"404 for {member} f{fh:03d}")
        return b"GRIB-stub"

    monkeypatch.setattr(gefs, "_fetch_step_s3", fake_fetch)
    monkeypatch.setattr(gefs, "_extract_points",
                        lambda blob, cities, scratch: {"t2m_c": [1.0]})


def _recover(monkeypatch, failed, recoverable):
    report = _report(steps_fetched=400 - len(failed),
                     missing_steps=[f"{m}/f{fh:03d}" for m, fh in failed])
    _stub_s3(monkeypatch, recoverable)
    step_values: dict = {}
    got = gefs._recover_steps_from_s3(
        date(2026, 8, 31), "00", failed, cities=pd.DataFrame(), scratch=None,
        step_values=step_values, report=report, workers=4,
    )
    return report, step_values, got


def test_gaps_nomads_left_are_filled_from_s3(monkeypatch):
    # The 2026-08-31 10:39Z shape in miniature: NOMADS 302s a contiguous run of steps,
    # S3 has all of them, and the cycle ends complete instead of thin.
    failed = [("gep01", 3), ("gep01", 6), ("gep02", 9)]
    report, step_values, got = _recover(monkeypatch, failed, recoverable=set(failed))

    assert report.steps_fetched == 400
    assert report.missing_steps == []
    assert report.steps_recovered == 3
    assert sorted(got) == ["gep01/f003", "gep01/f006", "gep02/f009"]
    # The recovered values must reach the reduction, or the day is still short.
    assert set(step_values) == set(failed)


def test_a_partly_recoverable_cycle_keeps_the_rest_listed(monkeypatch):
    failed = [("gep01", 3), ("gep01", 6), ("gep02", 9)]
    report, step_values, got = _recover(monkeypatch, failed,
                                        recoverable={("gep01", 3)})

    assert report.steps_recovered == 1
    assert report.steps_fetched == 398
    # Still thin, and honest about exactly which steps are absent.
    assert report.missing_steps == ["gep01/f006", "gep02/f009"]
    assert got == ["gep01/f003"]


def test_s3_failing_too_leaves_the_report_untouched(monkeypatch):
    # The fallback must not turn a reportable gap into a raised exception: a thin cycle is
    # refused downstream by the publish guard, which is a decision, not a crash.
    failed = [("gep01", 3), ("gep02", 9)]
    report, step_values, got = _recover(monkeypatch, failed, recoverable=set())

    assert got == []
    assert report.steps_recovered == 0
    assert report.steps_fetched == 398
    assert report.missing_steps == ["gep01/f003", "gep02/f009"]
    assert step_values == {}


def test_recovered_bytes_are_counted_once(monkeypatch):
    failed = [("gep01", 3)]
    report, _, _ = _recover(monkeypatch, failed, recoverable=set(failed))
    assert report.bytes_downloaded == len(b"GRIB-stub")


def test_fetch_cycle_repairs_gaps_before_the_daily_reduction(monkeypatch, tmp_path):
    # The wiring, not the helper: a NOMADS pull that loses steps must hand the reduction a
    # repaired set, because `_reduce_to_daily` is where a short day becomes a short mean.
    steps_lost = {("gec00", 12), ("gec00", 15)}
    monkeypatch.setattr(gefs, "load_cities",
                        lambda: pd.DataFrame({"city": ["Pune"], "state": ["MH"],
                                              "region": ["West"], "lat": [18.5],
                                              "lon": [73.9]}))

    def fake_step(init, hh, member, fh, transport):
        if (member, fh) in steps_lost:
            raise RuntimeError("302 for NOMADS throttle page")
        return b"GRIB-stub"

    monkeypatch.setattr(gefs, "_fetch_step", fake_step)
    monkeypatch.setattr(gefs, "_fetch_step_s3", lambda i, h, m, f: b"GRIB-stub")
    monkeypatch.setattr(gefs, "_extract_points",
                        lambda blob, cities, scratch: {"t2m_c": [1.0]})

    seen: dict = {}
    real = gefs._recover_steps_from_s3

    def spy(*args, **kw):
        seen["workers"] = args[-1]
        return real(*args, **kw)

    monkeypatch.setattr(gefs, "_recover_steps_from_s3", spy)
    monkeypatch.setattr(gefs, "_reduce_to_daily",
                        lambda step_values, *a, **kw: pd.DataFrame(
                            {"n": [len(step_values)]}))

    frame, report = gefs.fetch_cycle(date(2026, 8, 31), "00", members=["gec00"],
                                     workers=20, scratch=tmp_path, transport="nomads")

    assert report.steps_fetched == report.steps_expected == 80
    assert report.missing_steps == [] and report.steps_recovered == 2
    assert int(frame["n"].iloc[0]) == 80          # the reduction saw a whole cycle
    # The 6-worker cap exists to keep NOMADS happy; S3 has no such limit and must not
    # inherit it, or repairing a badly throttled pull takes longer than the pull did.
    assert seen["workers"] == 20


def test_an_s3_pull_does_not_try_to_repair_itself(monkeypatch, tmp_path):
    # On S3 there is no second transport to fall back to: a failure there means the step
    # genuinely is not published, and re-requesting it just doubles the wait.
    monkeypatch.setattr(gefs, "load_cities",
                        lambda: pd.DataFrame({"city": ["Pune"], "state": ["MH"],
                                              "region": ["West"], "lat": [18.5],
                                              "lon": [73.9]}))

    def fake_step(init, hh, member, fh, transport):
        if fh == 12:
            raise FileNotFoundError("404")
        return b"GRIB-stub"

    monkeypatch.setattr(gefs, "_fetch_step", fake_step)
    monkeypatch.setattr(gefs, "_extract_points",
                        lambda blob, cities, scratch: {"t2m_c": [1.0]})
    monkeypatch.setattr(gefs, "_reduce_to_daily",
                        lambda step_values, *a, **kw: pd.DataFrame({"n": [1]}))

    def refuse(*a, **kw):
        raise AssertionError("an S3 pull must not re-fetch from S3")

    monkeypatch.setattr(gefs, "_recover_steps_from_s3", refuse)

    _, report = gefs.fetch_cycle(date(2019, 6, 1), "00", members=["gec00"],
                                 workers=20, scratch=tmp_path, transport="s3")
    assert report.missing_steps == ["gec00/f012"] and report.steps_recovered == 0


# ------------------------------------------- which cycles a refresh run should pull
# GitHub's scheduler is best-effort: this workflow has fired 3 h 25 m late and skipped a
# slot outright. A run that only ever pulled "the newest cycle" would leave a permanent
# hole in the store each time, so the run catches up instead. Offline: only the choice of
# cycles is under test, never the download.

class _CycleArgs:
    def __init__(self, cycle=None, catch_up=4):
        self.cycle = cycle
        self.catch_up = catch_up


def _to_pull(**kw):
    from scripts.refresh_for_deploy import _cycles_to_pull
    return _cycles_to_pull(_CycleArgs(**kw))


def test_a_run_catches_up_on_consecutive_cycles_oldest_first(monkeypatch):
    # Six-hourly, not the 24 h stride recent_cycles uses for the historical backfill:
    # catching up on missed slots means consecutive cycles, not one per day.
    monkeypatch.setattr(gefs, "latest_expected_cycle",
                        lambda *a, **kw: (date(2026, 8, 31), "06"))
    assert _to_pull() == [
        (date(2026, 8, 30), "12"),
        (date(2026, 8, 30), "18"),
        (date(2026, 8, 31), "00"),
        (date(2026, 8, 31), "06"),
    ]


def test_catch_up_of_one_is_the_old_newest_only_behaviour(monkeypatch):
    monkeypatch.setattr(gefs, "latest_expected_cycle",
                        lambda *a, **kw: (date(2026, 8, 31), "06"))
    assert _to_pull(catch_up=1) == [(date(2026, 8, 31), "06")]


def test_catching_up_walks_across_a_month_boundary(monkeypatch):
    monkeypatch.setattr(gefs, "latest_expected_cycle",
                        lambda *a, **kw: (date(2026, 9, 1), "00"))
    assert _to_pull(catch_up=3) == [
        (date(2026, 8, 31), "12"),
        (date(2026, 8, 31), "18"),
        (date(2026, 9, 1), "00"),
    ]


def test_an_explicit_cycle_overrides_the_catch_up():
    assert _to_pull(cycle="2026-08-31T00") == [(date(2026, 8, 31), "00")]


def test_a_cycle_can_be_written_the_obvious_wrong_ways():
    from scripts.refresh_for_deploy import _parse_cycle
    for text in ("2026-08-31T00", "2026-08-31 00", "2026-08-31/00", " 2026-08-31T0 "):
        assert _parse_cycle(text) == (date(2026, 8, 31), "00")


def test_a_non_cycle_hour_is_refused_rather_than_pulled():
    # 03Z is not a GEFS cycle. Fetching it would 404 every step and report a thin cycle,
    # which reads as "NOAA is broken" rather than "you asked for a cycle that never exists".
    from scripts.refresh_for_deploy import _parse_cycle
    with pytest.raises(ValueError, match="not a GEFS cycle hour"):
        _parse_cycle("2026-08-31T03")
    with pytest.raises(ValueError, match="expected YYYY-MM-DDTHH"):
        _parse_cycle("yesterday")
