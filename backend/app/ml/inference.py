from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from app.features import engineering as fe
from app.features import pivot as pv
from app.ml import registry
from app.ml.calibration import ProbabilityCalibrator
from app.ml.thresholds import Thresholds
from app.services.regional_intelligence import reliability_score, confidence_label
from app.storage import parquet_store


@dataclass
class ModelState:

    run_id: str
    regressors: dict
    classifier: xgb.XGBClassifier
    classifier_columns: list
    thresholds: Thresholds
    historical_bust_freq: dict
    calibrator: ProbabilityCalibrator = field(default_factory=ProbabilityCalibrator)
    categorical_levels: dict = field(default_factory=dict)
    shap_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def variables(self) -> list:
        return sorted(self.regressors)


@dataclass
class ScoredCycle:

    run_id: str
    init_date: pd.Timestamp
    events: pd.DataFrame
    per_variable: pd.DataFrame
    n_rows_scored: int


_lock = threading.Lock()
_state_cache: tuple | None = None
_score_cache: "dict[tuple, ScoredCycle]" = {}
_SCORE_CACHE_MAX = 24


def load_model_state(run_id: Optional[str] = None) -> Optional[ModelState]:
    global _state_cache
    rid = run_id or registry.current_run_id()
    if rid is None:
        return None

    with _lock:
        if _state_cache and _state_cache[0] == rid:
            return _state_cache[1]

    regressors = registry.load_regressors(rid)
    clf, clf_cols = registry.load_classifier(rid)
    calibrator = registry.load_calibrator_for_run(rid)
    category_levels = registry.load_category_levels(rid)
    thr = registry.load_thresholds(rid)
    if not regressors or clf is None or thr is None:
        return None

    shap_path = registry.run_dir(rid) / "shap_summary.parquet"
    shap_summary = pd.read_parquet(shap_path) if shap_path.exists() else pd.DataFrame()

    import json
    def _read(name):
        p = registry.run_dir(rid) / name
        return json.loads(p.read_text()) if p.exists() else {}

    state = ModelState(
        run_id=rid,
        regressors=regressors,
        classifier=clf,
        classifier_columns=clf_cols,
        calibrator=calibrator,
        categorical_levels=category_levels,
        thresholds=thr,
        historical_bust_freq=registry.load_historical_bust_freq(rid),
        shap_summary=shap_summary,
        manifest=_read("manifest.json"),
        metrics=_read("metrics.json"),
    )
    with _lock:
        _state_cache = (rid, state)
    return state


def invalidate_caches() -> None:
    global _state_cache
    with _lock:
        _state_cache = None
        _score_cache.clear()
    try:
        from app.services import replay_service
        replay_service.invalidate()
        from app.services import ensemble_service
        ensemble_service.invalidate()
    except Exception:  # noqa: BLE001 - never let cache cleanup break an ingest
        pass


def score_latest_cycle(state: Optional[ModelState] = None) -> Optional[ScoredCycle]:
    return score_cycle(state, init_date=None)


def available_cycles() -> list[pd.Timestamp]:
    fc = parquet_store.read_dataset(
    # dedupe=False and the narrow projection are what keep this inside the memory budget.
        value_types=["forecast"], columns=["init_date"], dedupe=False,
    )
    if fc.empty:
        return []
    return sorted(pd.to_datetime(fc["init_date"]).dropna().unique(), reverse=True)


_MAX_LEAD_DAYS = 10
_OBS_PAD_DAYS = 3

_SCORING_COLUMNS = [
    "region_id", "variable", "valid_date", "value", "value_type",
    "init_date", "lead_time_days", "ensemble_member_id", "verification_status",
]


def score_cycle(
    state: Optional[ModelState] = None,
    init_date: "Optional[pd.Timestamp | str]" = None,
) -> Optional[ScoredCycle]:
    state = state or load_model_state()
    if state is None:
        return None

    fp = parquet_store.store_fingerprint()
    if init_date is None:
        latest = parquet_store.latest_forecast_init_date()
        if latest is None:
            return None
        target_init = pd.Timestamp(latest).normalize()
    else:
        target_init = pd.Timestamp(init_date).normalize()

    cache_key = (state.run_id, str(target_init), fp)
    with _lock:
        hit = _score_cache.get(cache_key)
        if hit is not None:
            return hit

    fc_rows = parquet_store.read_dataset(
        value_types=["forecast"], init_dates=[target_init.date()], columns=_SCORING_COLUMNS,
    )
    if fc_rows.empty:
        return None
    observed = parquet_store.read_dataset(
        value_types=["observed"],
        columns=_SCORING_COLUMNS,
        valid_date_min=(target_init - pd.Timedelta(days=_OBS_PAD_DAYS)).date(),
        valid_date_max=(target_init + pd.Timedelta(days=_MAX_LEAD_DAYS + _OBS_PAD_DAYS)).date(),
    )

    subset = pd.concat([fc_rows, observed], ignore_index=True)

    frame = fe.build_training_frame(
        subset,
        historical_bust_freq=state.historical_bust_freq or None,
        require_observed=False,
    )
    if frame.empty:
        return None

    pred = pd.Series(np.nan, index=frame.index, dtype=float)
    for var, (model, cols) in state.regressors.items():
        mask = frame["variable"] == var
        if not mask.any():
            continue
        X = _prep(frame.loc[mask], cols, fe_categorical=True, categorical_levels=state.categorical_levels.get(f"regressor::{var}", {}))
        pred.loc[mask] = model.predict(X)
    frame["pred_err"] = pred

    scored = frame.loc[frame["pred_err"].notna()].copy()
    if scored.empty:
        return None

    events = pv.build_event_frame(
        scored, scored["pred_err"], state.thresholds.p90_error,
        bust_threshold=None, historical_bust_freq=state.historical_bust_freq or None,
    )
    X_evt = _prep(events, state.classifier_columns, fe_categorical=False, categorical_levels=state.categorical_levels.get("classifier", {}))
    events["bust_probability"] = state.calibrator.predict(state.classifier.predict_proba(X_evt)[:, 1])
    events["risk_band"] = [state.thresholds.band_for(p) for p in events["bust_probability"]]
    # Reliability is a 0-100 trust score, not a probability, and never uses observed error.
    rel = []
    for row in events.itertuples():
        p = float(row.bust_probability)
        conf = float(getattr(row, "conf", np.nan)) if hasattr(row, "conf") else np.nan
        members = int(getattr(row, "ensemble_member_count", 31)) if hasattr(row, "ensemble_member_count") else 31
        base = 0.65 * (1.0 - np.clip(p, 0.0, 1.0)) + 0.35 * (0.0 if not np.isfinite(conf) else np.clip(conf, 0.0, 1.0))
        member_factor = np.clip(members / 31.0, 0.0, 1.0)
        rel.append(round(float(np.clip(100.0 * base * (0.85 + 0.15 * member_factor), 0.0, 100.0)), 2))
    events["reliability_score"] = rel
    events["confidence_label"] = [confidence_label(v) for v in rel]
    events["dominant_variable"] = _dominant_variable(events, state.thresholds.bust_threshold)

    per_variable = _per_variable_table(scored, state)

    result = ScoredCycle(
        run_id=state.run_id,
        init_date=target_init,
        events=events,
        per_variable=per_variable,
        n_rows_scored=len(scored),
    )
    with _lock:
        if len(_score_cache) >= _SCORE_CACHE_MAX:
            _score_cache.pop(next(iter(_score_cache)))
        _score_cache[cache_key] = result
    return result


def _prep(df: pd.DataFrame, cols: list, fe_categorical: bool, categorical_levels: dict | None = None) -> pd.DataFrame:
    categorical = {"region_id", "season"}
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = np.nan
    for c in cols:
        if c in categorical:
            levels = (categorical_levels or {}).get(c)
            X[c] = pd.Categorical(X[c], categories=levels) if levels is not None else X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _dominant_variable(events: pd.DataFrame, bust_threshold: dict) -> list:
    ratio_cols = {}
    for var, thr in bust_threshold.items():
        col = f"pred_err_{var}"
        if col in events.columns and thr:
            ratio_cols[var] = events[col] / thr
    if not ratio_cols:
        return [None] * len(events)
    ratios = pd.DataFrame(ratio_cols, index=events.index)
    if bool(ratios.isna().to_numpy().all()):
        return [None] * len(events)
    return ratios.idxmax(axis=1, skipna=True).where(ratios.notna().any(axis=1)).tolist()


def _per_variable_table(scored: pd.DataFrame, state: ModelState) -> pd.DataFrame:
    g = (
        scored.groupby(["region_id", "lead_time_days", "variable", "valid_date"],
                       observed=True)
        .agg(
            predicted_value=("forecast_value", "mean"),
            observed_value=("observed_value", "mean"),
            predicted_error=("pred_err", "mean"),
            ensemble_spread=("ensemble_spread", "mean"),
            ensemble_member_count=("ensemble_member_count", "max"),
        )
        .reset_index()
    )
    p90 = g["variable"].map(state.thresholds.p90_error)
    g["confidence"] = g["predicted_error"].div(p90).rsub(1.0).clip(0.0, 1.0)
    g["bust_threshold"] = g["variable"].map(state.thresholds.bust_threshold)
    return g


def model_validation_metrics(state: ModelState) -> dict:
    metrics = state.metrics or {}
    regs = {}
    for var, m in (metrics.get("regressors") or {}).items():
        chosen = m.get("test") or m.get("val") or m.get("train") or {}
        regs[var] = {
            "mae": chosen.get("mae"),
            "rmse": chosen.get("rmse"),
            "r2": chosen.get("r2"),
            "baseline_mae": chosen.get("baseline_mae_predict_mean"),
            "split": "test" if m.get("test") else ("val" if m.get("val") else "train"),
            "n": chosen.get("n"),
        }
    clf = metrics.get("classifier") or {}
    chosen_clf = clf.get("test") or clf.get("val") or clf.get("train") or {}
    return {
        "regressors": regs,
        "classifier": {
            **{k: chosen_clf.get(k) for k in
               ("n", "bust_rate", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier")},
            "split": "test" if clf.get("test") else ("val" if clf.get("val") else "train"),
        },
    }
