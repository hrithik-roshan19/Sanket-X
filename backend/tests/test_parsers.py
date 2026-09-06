"""parsers.py against the real sample files."""

from __future__ import annotations

import json

import pytest

from app.ingestion.parsers import ParseError, parse_upload
from tests.conftest import ERA5_CSV, GEFS_CSV, find_sample, iter_sample_files


@pytest.mark.parametrize("path", list(iter_sample_files()), ids=lambda p: p.name)
def test_every_sample_parses(path):
    """No real file in the sample folder should raise on parse."""
    pt = parse_upload(path, path.name)
    assert pt.df.shape[0] > 0
    assert pt.df.shape[1] > 0
    assert pt.detected_format in {"csv", "tsv", "xlsx", "json", "parquet"}


def test_nasa_power_daily_header_block_skipped():
    path = find_sample("POWER_Regional_Daily")
    if path is None:
        pytest.skip("no NASA POWER daily file available")
    pt = parse_upload(path, path.name)
    assert any("header" in n.lower() for n in pt.parse_notes)
    # header block gone: real columns are present, preamble text is not
    assert "valid_date" in pt.df.columns
    assert not any(str(c).startswith("-") for c in pt.df.columns)
    assert pt.reshape == "power_daily_doy"


def test_nasa_power_monthly_melt():
    path = find_sample("POWER_Regional_Monthly")
    if path is None:
        pytest.skip("no NASA POWER monthly file available")
    pt = parse_upload(path, path.name)
    assert pt.reshape == "power_monthly_melt"
    assert pt.grain == "monthly"
    assert "value" in pt.df.columns and "valid_date" in pt.df.columns
    # every month melted to first-of-month
    vd = pt.df["valid_date"].dropna()
    assert (vd.astype("datetime64[ns]").dt.day == 1).all()
    # no wide month columns left
    assert not ({"JAN", "FEB", "DEC"} & set(pt.df.columns))


def test_missing_sentinels_become_nan():
    path = find_sample("POWER_Regional_Daily")
    if path is None:
        pytest.skip("no NASA POWER daily file available")
    pt = parse_upload(path, path.name)
    # -999 sentinel must not survive as a literal value anywhere
    assert not (pt.df.astype(str) == "-999").any().any()


def test_xlsx_reads_first_sheet():
    path = find_sample("nasa_power_weather", ".xlsx") or find_sample(".xlsx")
    if path is None:
        pytest.skip("no xlsx sample available")
    pt = parse_upload(path, path.name)
    assert pt.detected_format == "xlsx"
    assert pt.df.shape[0] > 0


def test_generated_samples_have_expected_columns():
    g = parse_upload(GEFS_CSV, GEFS_CSV.name)
    assert {"init_date", "valid_date", "lead_day", "member", "t2m_c"} <= set(g.df.columns)
    e = parse_upload(ERA5_CSV, ERA5_CSV.name)
    assert {"date", "t2m_c", "precip_mm"} <= set(e.df.columns)


def test_geojson_json_is_rejected(tmp_path):
    p = tmp_path / "shape.json"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ParseError):
        parse_upload(p, p.name)


def test_records_json_ok(tmp_path):
    p = tmp_path / "rows.json"
    p.write_text(json.dumps([{"region": "Bihar", "date": "2019-06-01", "temp_c": 31.2}]))
    pt = parse_upload(p, p.name)
    assert list(pt.df.columns) == ["region", "date", "temp_c"]
