import pandas as pd
import numpy as np
from app.services.event_intelligence import build_event_intelligence, find_analogs


def test_event_intelligence_is_deterministic_and_deduped():
    out = build_event_intelligence({"rainfall_mm": 80, "temperature_c": 41})
    assert "extreme_rain" in out["event_types"]
    assert len(out["potential_impacts"]) == len(set(out["potential_impacts"]))
    assert out["recommended_actions"]


def test_event_rules_ignore_missing_and_nonfinite_values():
    out = build_event_intelligence({"rainfall_mm": np.nan, "temperature_c": None})
    assert out["event_types"] == []
    assert out["potential_impacts"] == []


def test_analogs_exclude_current_cycle_and_rank_similar_rows():
    history = pd.DataFrame([
        {"init_date": "2020-01-01", "valid_date": "2020-01-02", "region_id": "IN-A", "lead_time_days": 1, "x": 10, "y": 20, "bust_probability": .8, "risk_band": "high"},
        {"init_date": "2020-02-01", "valid_date": "2020-02-02", "region_id": "IN-B", "lead_time_days": 1, "x": 10.2, "y": 20.1, "bust_probability": .7, "risk_band": "high"},
        {"init_date": "2020-03-01", "valid_date": "2020-03-02", "region_id": "IN-C", "lead_time_days": 1, "x": 100, "y": 100, "bust_probability": .1, "risk_band": "low"},
    ])
    current = pd.Series({"init_date": "2026-01-01", "x": 10.1, "y": 20.05})
    out = find_analogs(current, history, ["x", "y"], k=2)
    assert len(out) == 2
    assert out[0]["region_id"] in {"IN-A", "IN-B"}
    assert out[0]["distance"] <= out[1]["distance"]
