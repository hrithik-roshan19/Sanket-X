import numpy as np
import pandas as pd

from app.ml.calibration import ProbabilityCalibrator, fit_calibrator


def test_identity_for_too_small_or_single_class():
    p = np.array([0.1, 0.8, 0.3])
    c = fit_calibrator([0, 1, 0], p)
    assert c.method == "identity"
    np.testing.assert_allclose(c.predict(p), p)


def test_platt_calibration_is_bounded_and_monotone():
    p = np.linspace(0.02, 0.98, 30)
    y = (p > 0.55).astype(int)
    c = fit_calibrator(y, p)
    assert c.method == "platt"
    out = c.predict(p)
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(np.diff(out) >= -1e-12)


def test_isotonic_calibration_roundtrip():
    p = np.linspace(0.01, 0.99, 80)
    y = (p > 0.65).astype(int)
    c = fit_calibrator(y, p)
    assert c.method == "isotonic"
    out = c.predict(p)
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(np.diff(out) >= -1e-12)


def test_nonfinite_probabilities_are_safe():
    c = ProbabilityCalibrator(method="platt", a=1.0, b=0.0)
    out = c.predict([np.nan, np.inf, -np.inf, 0.5])
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0) & (out <= 1))
