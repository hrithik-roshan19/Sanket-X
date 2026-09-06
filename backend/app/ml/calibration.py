from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    method: str = "identity"
    # Platt: p = sigmoid(a * logit(p_raw) + b)
    a: float = 1.0
    b: float = 0.0
    # Isotonic piecewise-linear mapping
    x_thresholds: list[float] | None = None
    y_thresholds: list[float] | None = None

    def predict(self, probabilities) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        p = np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 1e-7, 1 - 1e-7)
        if self.method == "platt":
            z = np.clip(np.log(p / (1.0 - p)), -30.0, 30.0)
            z = np.clip(self.a * z + self.b, -30.0, 30.0)
            return 1.0 / (1.0 + np.exp(-z))
        if self.method == "isotonic" and self.x_thresholds and self.y_thresholds:
            return np.interp(p, self.x_thresholds, self.y_thresholds)
        return p


def fit_calibrator(y_true, raw_probabilities) -> ProbabilityCalibrator:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(raw_probabilities, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    y, p = y[mask], np.clip(p[mask], 1e-7, 1 - 1e-7)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return ProbabilityCalibrator()

    # Isotonic needs enough calibration observations to avoid a highly jagged map.
    if len(y) >= 50:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        return ProbabilityCalibrator(
            method="isotonic",
            x_thresholds=[float(v) for v in iso.X_thresholds_],
            y_thresholds=[float(v) for v in iso.y_thresholds_],
        )

    # Platt scaling is more stable for small validation samples.
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(z, y)
    return ProbabilityCalibrator(method="platt", a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))


def save_calibrator(calibrator: ProbabilityCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "method": calibrator.method,
        "a": calibrator.a,
        "b": calibrator.b,
        "x_thresholds": calibrator.x_thresholds,
        "y_thresholds": calibrator.y_thresholds,
    }, indent=2))


def load_calibrator(path: Path) -> ProbabilityCalibrator:
    if not path.exists():
        return ProbabilityCalibrator()
    raw = json.loads(path.read_text())
    return ProbabilityCalibrator(
        method=raw.get("method", "identity"),
        a=float(raw.get("a", 1.0)),
        b=float(raw.get("b", 0.0)),
        x_thresholds=raw.get("x_thresholds"),
        y_thresholds=raw.get("y_thresholds"),
    )
