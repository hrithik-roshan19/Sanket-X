import numpy as np
import pandas as pd
import pytest

from app.ml.baselines import (
    ClimatologyBaseline,
    LeadDayBaseline,
    SpreadBaseline,
    LeadSpreadSeasonBaseline,
    brier,
    brier_skill_score,
    fit_all,
)


def _events(n=120):
    rng = np.random.default_rng(42)
    lead = np.tile(np.arange(1, 11), n // 10 + 1)[:n]
    spread = rng.uniform(0.1, 3.0, n)
    y = (spread > 2.0).astype(int)
    return pd.DataFrame({
        "y_bust": y,
        "lead_time_days": lead,
        "spread_mean": spread,
        "spread_max": spread + rng.uniform(0, 0.5, n),
        "season": np.where(np.arange(n) % 2, "monsoon", "winter"),
    })


def test_climatology_is_train_only_and_constant():
    train = pd.DataFrame({"y_bust": [0, 0, 1, 0]})
    test = pd.DataFrame({"y_bust": [1, 1, 1, 1]})
    m = ClimatologyBaseline().fit(train)
    p = m.predict_proba(test)
    assert np.allclose(p, 0.25)


def test_baselines_fit_and_predict_finite_probabilities():
    ev = _events()
    fitted = fit_all(ev.iloc[:100])
    for name, model in fitted.items():
        p = model.predict_proba(ev.iloc[100:])
        assert len(p) == 20, name
        assert np.isfinite(p).all(), name
        assert ((p > 0) & (p < 1)).all(), name


def test_brier_and_skill_score_reference_zero_edge():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    assert brier(y, p) < 0.05
    assert brier_skill_score(y, p, np.full(4, 0.5)) > 0
    assert np.isnan(brier_skill_score(np.zeros(4), np.zeros(4), np.zeros(4)))


def test_missing_spread_columns_degrade_to_base_rate_without_crashing():
    train = pd.DataFrame({"y_bust": [0, 1, 0, 1]})
    test = pd.DataFrame({"y_bust": [0, 1]})
    m = SpreadBaseline().fit(train)
    p = m.predict_proba(test)
    assert np.allclose(p, 0.5)


def test_unseen_season_is_supported_by_zero_indicator():
    train = pd.DataFrame({
        "y_bust": [0, 1, 0, 1],
        "lead_time_days": [1, 2, 1, 2],
        "spread_mean": [1, 2, 1, 2],
        "season": ["winter", "winter", "monsoon", "monsoon"],
    })
    test = pd.DataFrame({
        "y_bust": [0, 1],
        "lead_time_days": [3, 4],
        "spread_mean": [1.5, 2.5],
        "season": ["pre-monsoon", "pre-monsoon"],
    })
    m = LeadSpreadSeasonBaseline().fit(train)
    p = m.predict_proba(test)
    assert np.isfinite(p).all()


def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError, match="at least one labeled event"):
        ClimatologyBaseline().fit(pd.DataFrame({"y_bust": []}))
