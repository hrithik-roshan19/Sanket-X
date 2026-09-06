from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_samples import validate_gefs, validate_obs, validate_pairability


BASE = Path(__file__).resolve().parents[1]
GEFS = BASE / "data" / "samples" / "gefs_reforecast_india_2019.parquet"
OBS = BASE / "data" / "samples" / "era5_observations_india_2019.parquet"


def test_bundled_samples_exist():
    assert GEFS.exists()
    assert OBS.exists()


def test_bundled_samples_pass_structural_validation():
    pytest.importorskip("pyarrow")
    gefs = pd.read_parquet(GEFS)
    obs = pd.read_parquet(OBS)
    issues = []
    validate_gefs(gefs, issues)
    validate_obs(obs, issues)
    validate_pairability(gefs, obs, issues)
    errors = [x for x in issues if x["severity"] == "ERROR"]
    assert not errors, errors


def _minimal_gefs(**overrides):
    row = {
        "city": "Test", "state": "Test", "region": "Test", "latitude": 20.0, "longitude": 80.0,
        "init_date": "2019-01-01", "valid_date": "2019-01-01", "lead_day": 1, "member": "c00",
        "t2m_c": 25.0, "rh2m_pct": 50.0, "apcp_mm": 0.0, "mslp_hpa": 1010.0, "psfc_hpa": 1000.0,
        "pwat_kgm2": 20.0, "wspd10m_ms": 3.0, "wdir10m_deg": 180.0, "soilw_vol_pct": 30.0,
        "src_t2m_c": "test",
    }
    row.update(overrides)
    return pd.DataFrame([row])

def _minimal_obs():
    return pd.DataFrame([{
        "city": "Test", "state": "Test", "region": "Test", "latitude": 20.0, "longitude": 80.0,
        "date": "2019-01-01", "t2m_c": 25.0, "rh2m_pct": 50.0, "precip_mm": 0.0, "mslp_hpa": 1010.0,
        "psfc_hpa": 1000.0, "pwat_kgm2": 20.0, "wspd10m_ms": 3.0, "wdir10m_deg": 180.0,
        "soil_moisture_pct": 30.0,
    }])

def test_non_integer_lead_is_rejected():
    issues = []
    validate_gefs(_minimal_gefs(lead_day=1.5), issues)
    assert any(x["code"] == "BAD_LEAD" for x in issues)

def test_fractional_day_alignment_is_rejected():
    issues = []
    validate_gefs(_minimal_gefs(valid_date="2019-01-01T12:00:00"), issues)
    assert any(x["code"] == "TIME_ALIGNMENT" for x in issues)

def test_valid_pair_is_accepted():
    issues = []
    gefs = _minimal_gefs()
    obs = _minimal_obs()
    validate_obs(obs, issues)
    result = validate_pairability(gefs, obs, issues)
    assert result["pair_rate"] == 1.0
    assert not [x for x in issues if x["severity"] == "ERROR"]
