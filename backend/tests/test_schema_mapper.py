"""schema_mapper.py against the real headers."""

from __future__ import annotations

import pandas as pd
import pytest

from app.ingestion.parsers import parse_upload
from app.ingestion.schema_mapper import SchemaMapper, fingerprint, jaccard
from tests.conftest import ERA5_CSV, GEFS_CSV, find_sample


def _map(path, filename=None, rows=3000):
    pt = parse_upload(path, (filename or path.name))
    return pt, SchemaMapper(filename_hint=(filename or path.name)).map_table(pt.df.head(rows))


def _prop(result, col):
    return next(p for p in result.proposals if p.source_column == col)


# --------------------------------------------------------------------------- GEFS

def test_gefs_variable_mapping():
    _, mr = _map(GEFS_CSV)
    assert mr.layout == "wide"
    expect = {
        "t2m_c": "temperature_c",
        "rh2m_pct": "humidity_pct",
        "apcp_mm": "rainfall_mm",
        "mslp_hpa": "pressure_hpa",
        "pwat_kgm2": "atmospheric_moisture_kgm2",
        "wspd10m_ms": "wind_speed_ms",
        "wdir10m_deg": "wind_direction_deg",
        "soilw_vol_pct": "soil_moisture_pct",
    }
    for col, var in expect.items():
        p = _prop(mr, col)
        assert p.role == "measurement"
        assert p.suggested_variable == var, (col, p.suggested_variable)
        assert p.confidence >= 0.85


def test_gefs_dimensions_and_noise():
    _, mr = _map(GEFS_CSV)
    for col in ("init_date", "valid_date", "lead_day", "member"):
        assert _prop(mr, col).role == "dimension", col
    # provenance columns must be excluded, not force-mapped
    for col in ("src_tmp_2m", "srcmsg_apcp_sfc", "src_ugrd_hgt"):
        assert _prop(mr, col).role == "unmapped", col


def test_gefs_value_type_is_forecast_from_filename():
    _, mr = _map(GEFS_CSV)
    assert _prop(mr, "t2m_c").suggested_value_type == "forecast"


# --------------------------------------------------------------------------- ERA5

def test_era5_value_type_column_detected():
    _, mr = _map(ERA5_CSV)
    assert mr.value_type_column == "source"
    # per-column value_type suppressed when a value_type column exists
    assert _prop(mr, "t2m_c").suggested_value_type is None
    assert _prop(mr, "precip_mm").suggested_variable == "rainfall_mm"
    assert _prop(mr, "soil_moisture_pct").suggested_variable == "soil_moisture_pct"


# --------------------------------------------------------- all_india_weather (mixed)

def test_all_india_weather_data_type_and_units():
    path = find_sample("all_india_weather", ".csv")
    if path is None:
        pytest.skip("all_india_weather sample not available")
    _, mr = _map(path)
    assert mr.value_type_column == "Data_Type"
    ws = _prop(mr, "Wind_Speed_kmh")
    assert ws.suggested_variable == "wind_speed_ms"
    assert ws.unit_conversion == "kmh_to_ms"
    # avg/max/min all claim temperature_c -> collision -> confirmation
    assert _prop(mr, "Temp_Avg_C").decision == "needs_confirmation"
    assert any("temperature_c" in n for n in mr.notes)


def test_solar_irradiance_is_unmapped():
    path = find_sample("all_india_nasa_power_weather", ".csv") or find_sample("nasa_power_weather")
    if path is None:
        pytest.skip("nasa power weather sample not available")
    _, mr = _map(path)
    assert _prop(mr, "Solar_Irradiance_MJ_m2").role == "unmapped"


# --------------------------------------------------------------------------- long layout

def test_long_layout_detected():
    path = find_sample("forecastguard_demo_training_data") or find_sample("demo_training")
    if path is None:
        pytest.skip("long-layout demo file not available")
    _, mr = _map(path)
    assert mr.layout == "long"
    assert _prop(mr, "forecast_variable").role == "variable_name"
    assert _prop(mr, "forecast_value").suggested_value_type == "forecast"
    assert _prop(mr, "actual_value").suggested_value_type == "observed"


# --------------------------------------------------------------------------- ambiguity

def test_bare_moisture_column_needs_confirmation():
    df = pd.DataFrame({
        "region": ["Bihar"] * 20,
        "date": pd.date_range("2019-06-01", periods=20).astype(str),
        "moisture": [35.0 + i for i in range(20)],  # 35..54: plausible for both TCWV and soil %
    })
    mr = SchemaMapper(filename_hint="x.csv").map_table(df)
    p = _prop(mr, "moisture")
    assert p.decision == "needs_confirmation"
    assert p.suggested_variable in {"atmospheric_moisture_kgm2", "soil_moisture_pct"}


# --------------------------------------------------------------------------- profiles

def test_fingerprint_and_jaccard():
    a = ["Temp_C", "region", "date"]
    b = ["temp c", "Region", "Date"]           # same after normalisation
    c = ["Temp_C", "region", "date", "extra"]
    assert fingerprint(a) == fingerprint(b)
    assert jaccard(a, b) == 1.0
    assert 0.7 < jaccard(a, c) < 1.0


def test_source_profile_exact_reuse(session):
    """Map -> persist a confirmed profile -> map same headers again -> auto-applied."""
    from app.db import crud

    _, mr = _map(ERA5_CSV)
    confirmed = {
        p.source_column: {"variable": p.suggested_variable,
                          "value_type": p.suggested_value_type,
                          "role": p.role, "unit_conversion": p.unit_conversion}
        for p in mr.proposals if p.role in ("measurement", "value_type")
    }
    crud.upsert_source_profile(
        session, fingerprint=mr.fingerprint,
        headers=[str(c) for c in parse_upload(ERA5_CSV, ERA5_CSV.name).df.columns],
        confirmed_mapping=confirmed, file_format="csv", layout="wide",
    )
    session.flush()

    pt2 = parse_upload(ERA5_CSV, ERA5_CSV.name)
    mr2 = SchemaMapper(filename_hint=ERA5_CSV.name).map_table(
        pt2.df.head(2000), existing_profiles=crud.all_source_profiles(session)
    )
    assert mr2.profile_match == "exact"
    assert _prop(mr2, "t2m_c").decision == "confirmed"
    assert mr2.auto_accepted
