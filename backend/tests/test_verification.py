import pandas as pd
import pytest

from app.verification import (
    event_level_labels,
    fit_bust_thresholds,
    label_busts,
    pair_forecasts_observations,
    validate_temporal_contract,
)


def _rows():
    return pd.DataFrame([
        {"region_id":"r1","variable":"temperature_c","value_type":"forecast","init_date":"2020-01-01","valid_date":"2020-01-01","lead_time_days":1,"ensemble_member_id":"c00","value":20},
        {"region_id":"r1","variable":"temperature_c","value_type":"forecast","init_date":"2020-01-01","valid_date":"2020-01-01","lead_time_days":1,"ensemble_member_id":"p01","value":22},
        {"region_id":"r1","variable":"temperature_c","value_type":"forecast","init_date":"2020-01-01","valid_date":"2020-01-02","lead_time_days":2,"ensemble_member_id":"c00","value":25},
        {"region_id":"r1","variable":"temperature_c","value_type":"observed","valid_date":"2020-01-01","value":21,"verification_status":"final"},
        {"region_id":"r1","variable":"temperature_c","value_type":"observed","valid_date":"2020-01-02","value":24,"verification_status":"final"},
    ])


def test_pairing_and_error_are_event_safe():
    paired, report = pair_forecasts_observations(_rows())
    assert report.status == "valid"
    assert len(paired) == 3
    assert paired["abs_error"].tolist() == [1, 1, 1]


def test_temporal_contract_rejects_day_shift():
    f = _rows().query("value_type == 'forecast'").copy()
    f.loc[f.index[0], "valid_date"] = "2020-01-02"
    bad = validate_temporal_contract(f)
    assert len(bad) == 1


def test_provisional_observation_is_excluded():
    df = _rows()
    df.loc[df.index[-1], "verification_status"] = "provisional"
    paired, report = pair_forecasts_observations(df)
    assert report.provisional_observations == 1
    assert len(paired) == 2


def test_conflicting_observation_is_fail_closed():
    df = pd.concat([_rows(), pd.DataFrame([{
        "region_id":"r1","variable":"temperature_c","value_type":"observed",
        "valid_date":"2020-01-01","value":99,"verification_status":"final"
    }])], ignore_index=True)
    _, report = pair_forecasts_observations(df)
    assert report.status == "invalid"
    assert any("conflicting" in e for e in report.errors)


def test_threshold_is_train_only_and_frozen():
    train = pd.DataFrame({"variable":["temperature_c"]*10,"abs_error":range(1,11)})
    thresholds = fit_bust_thresholds(train, 90)
    assert thresholds["temperature_c"] == pytest.approx(9.1)
    test = pd.DataFrame({"variable":["temperature_c"],"abs_error":[100]})
    labelled = label_busts(test, thresholds)
    assert int(labelled.iloc[0].y_bust) == 1


def test_event_label_uses_ensemble_mean_not_member_label():
    paired, _ = pair_forecasts_observations(_rows())
    thresholds = {"temperature_c": 1.0}
    events = event_level_labels(paired, thresholds)
    day1 = events[events.lead_time_days == 1].iloc[0]
    assert day1.fc_mean == pytest.approx(21)
    assert day1.actual_error == pytest.approx(0)
    assert int(day1.y_bust) == 0
