from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.features.pivot import classifier_feature_columns
from app.ml.calibration import ProbabilityCalibrator, fit_calibrator

CATEGORICAL = ["region_id", "season"]

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=15,
    reg_lambda=3.0,
    reg_alpha=1.0,
    gamma=0.5,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=0,
    random_state=42,
    early_stopping_rounds=40,
)


@dataclass
class ClassifierArtifact:
    model: xgb.XGBClassifier
    feature_columns: list
    metrics: dict
    n_train: int
    n_val: int
    train_bust_rate: float
    calibrator: ProbabilityCalibrator = field(default_factory=ProbabilityCalibrator)
    categorical_levels: dict[str, list] = field(default_factory=dict)


def _prep_X(df: pd.DataFrame, cols: list, categorical_levels: dict[str, list] | None = None) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if c in CATEGORICAL:
            levels = (categorical_levels or {}).get(c)
            X[c] = pd.Categorical(X[c], categories=levels) if levels is not None else X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    pred = (proba >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "bust_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)) if len(y_true) else float("nan"),
    }
    if 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    out["calibration"] = _reliability(y_true, proba)
    return out


def _reliability(y_true, proba, bins: int = 5) -> list:
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    if len(y_true) == 0:
        return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        out.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "predicted_mean": float(proba[m].mean()),
            "observed_rate": float(y_true[m].mean()),
            "n": n,
        })
    return out


def train_bust_classifier(
    event_train: pd.DataFrame, event_val: pd.DataFrame | None = None
) -> ClassifierArtifact:
    cols = classifier_feature_columns(event_train)
    y = event_train["y_bust"].astype(int).to_numpy()
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    spw = (neg / pos) if pos else 1.0

    params = dict(XGB_PARAMS)
    val_df: pd.DataFrame | None = (
        event_val
        if (event_val is not None and len(event_val) >= 10 and "y_bust" in event_val
            and 0 < int(event_val["y_bust"].sum()) < len(event_val))
        else None
    )
    categorical_levels = {c: [str(v) for v in event_train[c].astype(str).dropna().unique()] for c in CATEGORICAL if c in cols}
    Xtr = _prep_X(event_train, cols, categorical_levels)
    if val_df is not None:
        Xva = _prep_X(val_df, cols, categorical_levels)
        model = xgb.XGBClassifier(**params, scale_pos_weight=spw)
        model.fit(Xtr, y, eval_set=[(Xva, val_df["y_bust"].astype(int).to_numpy())], verbose=False)
    else:
        params.pop("early_stopping_rounds", None)
        model = xgb.XGBClassifier(**params, scale_pos_weight=spw)
        model.fit(Xtr, y)

    metrics: dict = {"train": _evaluate(y, model.predict_proba(Xtr)[:, 1])}
    n_val = 0
    if val_df is not None:
        pv = model.predict_proba(_prep_X(val_df, cols, categorical_levels))[:, 1]
        metrics["val"] = _evaluate(val_df["y_bust"], pv)
        n_val = len(val_df)
        metrics["best_iteration"] = int(getattr(model, "best_iteration", params["n_estimators"]) or 0)

    # Calibrate only on held-out validation predictions. If the validation sample is
    # too small or single-class, fit_calibrator returns an explicit identity calibrator.
    calibrator = ProbabilityCalibrator()
    if val_df is not None and len(val_df):
        raw_val = model.predict_proba(_prep_X(val_df, cols, categorical_levels))[:, 1]
        calibrator = fit_calibrator(val_df["y_bust"].astype(int), raw_val)
        metrics["val_raw_brier"] = metrics["val"].get("brier", float("nan"))
        cal_val = calibrator.predict(raw_val)
        metrics["val_calibrated"] = _evaluate(val_df["y_bust"], cal_val)

    return ClassifierArtifact(
        model=model,
        feature_columns=cols,
        metrics=metrics,
        n_train=len(event_train),
        n_val=n_val,
        train_bust_rate=float(y.mean()),
        calibrator=calibrator,
        categorical_levels=categorical_levels,
    )


def predict_bust_probability(artifact: ClassifierArtifact, event_df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
    raw = artifact.model.predict_proba(_prep_X(event_df, artifact.feature_columns, artifact.categorical_levels))[:, 1]
    return artifact.calibrator.predict(raw) if calibrated else raw
