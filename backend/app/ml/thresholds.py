from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD_PCT = 90.0
DEFAULT_RISK_CUTS = (0.50, 0.80)


@dataclass
class Thresholds:
    bust_threshold: dict
    p90_error: dict
    risk_band_cuts: dict
    threshold_percentile: float = DEFAULT_THRESHOLD_PCT
    risk_cut_quantiles: tuple = DEFAULT_RISK_CUTS
    notes: list = field(default_factory=list)

    def band_for(self, p: float) -> str:
        if p is None or np.isnan(p):
            return "unknown"
        if p >= self.risk_band_cuts["high"]:
            return "high"
        if p >= self.risk_band_cuts["medium"]:
            return "medium"
        return "low"

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=_json_default))

    @classmethod
    def from_json(cls, path: Path) -> "Thresholds":
        d = json.loads(Path(path).read_text())
        d["risk_cut_quantiles"] = tuple(d.get("risk_cut_quantiles", DEFAULT_RISK_CUTS))
        return cls(**d)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(type(o))


def compute_error_thresholds(
    event_error: pd.DataFrame, percentile: float = DEFAULT_THRESHOLD_PCT
) -> dict:
    out = {}
    for var, g in event_error.groupby("variable"):
        vals = g["abs_error"].dropna().to_numpy()
        if vals.size:
            out[var] = float(np.percentile(vals, percentile))
    return out


def compute_member_p90(member_error: pd.DataFrame) -> dict:
    out = {}
    for var, g in member_error.groupby("variable"):
        vals = g["abs_error"].dropna().to_numpy()
        if vals.size:
            out[var] = float(np.percentile(vals, 90))
    return out


def compute_risk_bands(proba: np.ndarray, cuts: tuple = DEFAULT_RISK_CUTS) -> dict:
    p = np.asarray(proba, dtype=float)
    p = p[~np.isnan(p)]
    if p.size < 20 or np.allclose(p, p[0]):
        return {"medium": 0.33, "high": 0.66}
    med = float(np.quantile(p, cuts[0]))
    hi = float(np.quantile(p, cuts[1]))
    if hi <= med:
        hi = min(1.0, med + 1e-3)
    return {"medium": med, "high": hi}
