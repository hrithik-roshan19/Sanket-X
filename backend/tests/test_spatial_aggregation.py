import pandas as pd
import pytest
from shapely.geometry import box

from app.data_sources.spatial_aggregation import (
    aggregate_grid_to_subdivision,
    validate_forecast_time_alignment,
)


def test_area_weighted_subdivision_aggregation():
    # Two cells at different latitudes. Their values are deliberately different so
    # an unweighted mean would produce a different result.
    frame = pd.DataFrame({
        "lat": [10.0, 20.0],
        "lon": [75.0, 75.0],
        "value": [10.0, 20.0],
        "valid_date": ["2020-01-01", "2020-01-01"],
    })
    geom = [("imd-01", box(70, 5, 80, 25))]
    result = aggregate_grid_to_subdivision(frame, geom, "value", ["valid_date"])
    assert len(result) == 1
    assert result.loc[0, "grid_cell_count"] == 2
    assert 10 < result.loc[0, "value"] < 20


def test_unmatched_grid_cells_are_not_silently_assigned():
    frame = pd.DataFrame({"lat": [10.0], "lon": [75.0], "value": [10.0], "valid_date": ["2020-01-01"]})
    geom = [("imd-01", box(70, 20, 80, 30))]
    result = aggregate_grid_to_subdivision(frame, geom, "value", ["valid_date"])
    assert result.empty


def test_temporal_alignment_accepts_lead_one_same_day():
    frame = pd.DataFrame({
        "init_time": ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
        "valid_time": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
        "lead_day": [1, 2],
    })
    validate_forecast_time_alignment(frame)


def test_temporal_alignment_rejects_one_day_shift():
    frame = pd.DataFrame({
        "init_time": ["2020-01-01T00:00:00Z"],
        "valid_time": ["2020-01-02T00:00:00Z"],
        "lead_day": [1],
    })
    with pytest.raises(ValueError, match="misalignment"):
        validate_forecast_time_alignment(frame)


def test_weighted_coverage_exposes_missing_data():
    from app.data_sources.spatial_aggregation import weighted_coverage
    frame = pd.DataFrame({
        "lat": [10.0, 20.0], "lon": [75.0, 75.0],
        "value": [10.0, float("nan")], "valid_date": ["2020-01-01", "2020-01-01"],
    })
    geom = [("imd-01", box(70, 5, 80, 25))]
    result = weighted_coverage(frame, geom, "value", ["valid_date"])
    assert 0 < result.loc[0, "coverage_ratio"] < 1
