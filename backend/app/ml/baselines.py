from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LABEL = "y_bust"
_EPS = 1e-6

_LOGIT = dict(max_iter=1000, solver="lbfgs", random_state=42)


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)


@dataclass
class _Base:

    name: str = "baseline"
    features: list = field(default_factory=list)
    base_rate: float = float("nan")
    _model: LogisticRegression | None = None
    _medians: dict = field(default_factory=dict)
    _columns: list = field(default_factory=list)

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _prepare(self, ev: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        X = self._design(ev)
        if fitting:
            self._medians = {c: float(X[c].median()) for c in X.columns
                             if X[c].notna().any()}
        X = X.copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self._medians.get(c, 0.0))
        if fitting:
            self._columns = list(X.columns)
        return X.reindex(columns=self._columns, fill_value=0.0)

    def fit(self, train_events: pd.DataFrame) -> "_Base":
        if LABEL not in train_events.columns:
            raise ValueError(f"training events must contain {LABEL}")
        if train_events.empty:
            raise ValueError("at least one labeled event is required")
        y = np.asarray(train_events[LABEL], int)
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"{LABEL} must contain only 0/1 labels")
        self.base_rate = float(y.mean())
        X = self._prepare(train_events, fitting=True)
        self.features = list(X.columns)
        if len(np.unique(y)) < 2 or X.empty or X.shape[1] == 0:
            self._model = None
            return self
        self._model = LogisticRegression(**_LOGIT).fit(X.to_numpy(float), y)
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return _clip(np.full(len(events), self.base_rate))
        X = self._prepare(events, fitting=False)
        return _clip(self._model.predict_proba(X.to_numpy(float))[:, 1])


@dataclass
class ClimatologyBaseline(_Base):

    name: str = "climatology"

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=ev.index)

    def fit(self, train_events: pd.DataFrame) -> "ClimatologyBaseline":
        if LABEL not in train_events.columns:
            raise ValueError(f"training events must contain {LABEL}")
        if train_events.empty:
            raise ValueError("at least one labeled event is required")
        y = np.asarray(train_events[LABEL], int)
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"{LABEL} must contain only 0/1 labels")
        self.base_rate = float(y.mean())
        self._model, self.features = None, []
        return self

    def predict_proba(self, events: pd.DataFrame) -> np.ndarray:
        return _clip(np.full(len(events), self.base_rate))


@dataclass
class LeadDayBaseline(_Base):

    name: str = "lead_day"

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"lead_time_days": ev["lead_time_days"]}, index=ev.index)


@dataclass
class SpreadBaseline(_Base):

    name: str = "spread"

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in ("spread_mean", "spread_max"):
            if c in ev.columns:
                cols[c] = ev[c]
        if not cols:
            return pd.DataFrame(index=ev.index)
        return pd.DataFrame(cols, index=ev.index)


@dataclass
class LeadSpreadSeasonBaseline(_Base):

    name: str = "lead+spread+season"
    _seasons: list = field(default_factory=list)

    def _design(self, ev: pd.DataFrame) -> pd.DataFrame:
        cols = {"lead_time_days": ev["lead_time_days"]}
        for c in ("spread_mean", "spread_max"):
            if c in ev.columns:
                cols[c] = ev[c]
        X = pd.DataFrame(cols, index=ev.index)
        if "season" in ev.columns:
            s = ev["season"].astype(str)
            if not self._seasons:
                self._seasons = sorted(s.dropna().unique())
            for season in self._seasons:
                X[f"season_{season}"] = (s == season).astype(float)
        return X


ALL_BASELINES = (ClimatologyBaseline, LeadDayBaseline, SpreadBaseline,
                 LeadSpreadSeasonBaseline)


def brier(y_true, proba) -> float:
    y = np.asarray(y_true, float)
    p = np.asarray(proba, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def brier_skill_score(y_true, proba, reference_proba) -> float:
    bs_ref = brier(y_true, reference_proba)
    if not np.isfinite(bs_ref) or bs_ref <= 0:
        return float("nan")
    return float(1.0 - brier(y_true, proba) / bs_ref)


def fit_all(train_events: pd.DataFrame) -> dict:
    out = {}
    for cls in ALL_BASELINES:
        m = cls().fit(train_events)
        out[m.name] = m
    return out
