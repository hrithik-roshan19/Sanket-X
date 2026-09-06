from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_ROWS = 30
MIN_VAL_ROWS_FOR_EARLY_STOPPING = 5
EARLY_STOPPING_ROUNDS = 30

NUMERIC_FEATURES = [
    "lead_time_days",
    "forecast_value",
    "month",
    "ensemble_spread",
    "ensemble_member_count",
    "forecast_value_lag",
    "forecast_delta_from_previous_lead",
    "pressure_rate_of_change",
    "moisture_rate_of_change",
]

CATEGORICAL_FEATURES = [
    "region_id",
    "season",
]

CONCURRENT_PREFIX = "fc_"

UNKNOWN_CATEGORY = "__UNK__"

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=2.0,
    reg_alpha=0.5,
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    n_jobs=0,
    random_state=42,
)


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

@dataclass
class RegressorArtifact:
    variable: str
    model: xgb.XGBRegressor
    feature_columns: list
    metrics: dict
    n_train: int
    n_val: int
    categorical_levels: dict[str, list] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def feature_columns(df: pd.DataFrame) -> list:
    """
    Return only forecast-available features.

    IMPORTANT:
    Target-derived fields such as abs_error, observed_value,
    actual_error, bust_ratio and y_bust must never enter the regressor.
    """

    cols = [
        c
        for c in NUMERIC_FEATURES
        if c in df.columns
    ]

    # Concurrent forecast variables are allowed because they are available
    # at forecast generation time.
    cols += [
        c
        for c in df.columns
        if c.startswith(CONCURRENT_PREFIX)
    ]

    cols += [
        c
        for c in CATEGORICAL_FEATURES
        if c in df.columns
    ]

    return cols


# ---------------------------------------------------------------------------
# Categorical vocabulary
# ---------------------------------------------------------------------------

def _fit_category_levels(
    df: pd.DataFrame,
    cols: list,
) -> dict[str, list]:
    """
    Build categorical vocabularies from TRAINING data only.

    An explicit __UNK__ category is added so inference can safely handle
    categories never seen during training.
    """

    levels: dict[str, list] = {}

    for c in CATEGORICAL_FEATURES:
        if c not in cols or c not in df.columns:
            continue

        values = (
            df[c]
            .astype("string")
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        values = sorted(
            set(values) - {UNKNOWN_CATEGORY}
        )

        values.append(UNKNOWN_CATEGORY)

        levels[c] = values

    return levels


def _normalise_categories(
    df: pd.DataFrame,
    cols: list,
    categorical_levels: dict[str, list] | None = None,
) -> pd.DataFrame:
    """
    Convert categorical columns using the training vocabulary.

    Unknown categories are explicitly mapped to __UNK__.
    """

    X = df[cols].copy()

    for c in CATEGORICAL_FEATURES:
        if c not in X.columns:
            continue

        levels = (
            categorical_levels or {}
        ).get(c)

        if levels is None:
            # Used only when no train vocabulary is available.
            values = (
                X[c]
                .astype("string")
                .fillna(UNKNOWN_CATEGORY)
                .astype(str)
            )

            unique = sorted(
                set(values.unique()) - {UNKNOWN_CATEGORY}
            )

            levels = unique + [UNKNOWN_CATEGORY]

        values = (
            X[c]
            .astype("string")
            .fillna(UNKNOWN_CATEGORY)
            .astype(str)
        )

        values = values.where(
            values.isin(levels),
            UNKNOWN_CATEGORY,
        )

        X[c] = pd.Categorical(
            values,
            categories=levels,
        )

    # All remaining features must be numeric.
    for c in X.columns:
        if c not in CATEGORICAL_FEATURES:
            X[c] = pd.to_numeric(
                X[c],
                errors="coerce",
            )

    return X


def _prep_X(
    df: pd.DataFrame,
    cols: list,
    categorical_levels: dict[str, list] | None = None,
) -> pd.DataFrame:
    """
    Prepare a dataframe using the same schema used during training.
    """

    # Defensive handling for missing columns.
    X = pd.DataFrame(index=df.index)

    for c in cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = np.nan

    return _normalise_categories(
        X,
        cols,
        categorical_levels,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _evaluate(
    y_true,
    y_pred,
) -> dict:
    """
    Evaluate error regression against the constant-mean baseline.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    finite = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) == 0:
        raise ValueError(
            "cannot evaluate an empty/non-finite target"
        )

    baseline_prediction = float(
        np.mean(y_true)
    )

    baseline_mae = float(
        np.mean(
            np.abs(
                y_true - baseline_prediction
            )
        )
    )

    model_mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    skill = np.nan

    if baseline_mae > 0:
        skill = float(
            1.0 - (
                model_mae / baseline_mae
            )
        )

    return {
        "mae": model_mae,
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": (
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )
            if len(y_true) > 2
            else float("nan")
        ),
        "n": int(len(y_true)),
        "baseline_mae_predict_mean": baseline_mae,
        "mae_skill_vs_mean": skill,
        "beats_mean_baseline": bool(
            model_mae < baseline_mae
        ),
    }


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

def _clean_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove rows with invalid regression targets.

    The target is absolute forecast error and therefore must be finite
    and non-negative.
    """

    if "abs_error" not in df.columns:
        raise ValueError(
            "training dataframe must contain abs_error"
        )

    out = df.copy()

    target = pd.to_numeric(
        out["abs_error"],
        errors="coerce",
    )

    valid = (
        np.isfinite(target)
        & (target >= 0)
    )

    out = out.loc[valid].copy()

    out["abs_error"] = target.loc[
        out.index
    ].astype(float)

    return out


# ---------------------------------------------------------------------------
# XGBoost model
# ---------------------------------------------------------------------------

def _new_model(
    *,
    use_early_stopping: bool,
) -> xgb.XGBRegressor:
    """
    Construct the XGBoost regressor.

    Early stopping is enabled only when a genuine validation set exists.
    """

    params = dict(XGB_PARAMS)

    if use_early_stopping:
        params["early_stopping_rounds"] = (
            EARLY_STOPPING_ROUNDS
        )

    return xgb.XGBRegressor(**params)


# ---------------------------------------------------------------------------
# Single-variable training
# ---------------------------------------------------------------------------

def train_variable_regressor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    variable: str,
) -> RegressorArtifact | None:
    """
    Train one XGBoost error regressor for one canonical variable.

    Training:
        train_df

    Model selection / early stopping:
        val_df

    Target:
        abs_error

    The validation set is never used for fitting the tree weights.
    """

    tr = train_df[
        train_df["variable"] == variable
    ].copy()

    va = val_df[
        val_df["variable"] == variable
    ].copy()

    tr = _clean_target(tr)
    va = _clean_target(va)

    if len(tr) < MIN_ROWS:
        return None

    cols = feature_columns(tr)

    if not cols:
        raise ValueError(
            f"no usable features for variable={variable}"
        )

    categorical_levels = _fit_category_levels(
        tr,
        cols,
    )

    X_train = _prep_X(
        tr,
        cols,
        categorical_levels,
    )

    y_train = tr["abs_error"].to_numpy(
        dtype=float
    )

    use_early_stopping = (
        len(va)
        >= MIN_VAL_ROWS_FOR_EARLY_STOPPING
    )

    model = _new_model(
        use_early_stopping=use_early_stopping,
    )

    if use_early_stopping:
        X_val = _prep_X(
            va,
            cols,
            categorical_levels,
        )

        y_val = va["abs_error"].to_numpy(
            dtype=float
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            verbose=False,
        )
    else:
        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

    train_pred = np.maximum(
        model.predict(X_train),
        0.0,
    )

    metrics = {
        "train": _evaluate(
            y_train,
            train_pred,
        )
    }

    if len(va) >= 5:
        val_pred = np.maximum(
            model.predict(
                _prep_X(
                    va,
                    cols,
                    categorical_levels,
                )
            ),
            0.0,
        )

        metrics["val"] = _evaluate(
            va["abs_error"],
            val_pred,
        )

    if use_early_stopping:
        best_iteration = getattr(
            model,
            "best_iteration",
            None,
        )

        best_score = getattr(
            model,
            "best_score",
            None,
        )

        metrics["early_stopping"] = {
            "enabled": True,
            "rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": (
                int(best_iteration)
                if best_iteration is not None
                else None
            ),
            "best_score": (
                float(best_score)
                if best_score is not None
                and np.isfinite(best_score)
                else None
            ),
        }
    else:
        metrics["early_stopping"] = {
            "enabled": False,
            "rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": None,
            "best_score": None,
        }

    return RegressorArtifact(
        variable=variable,
        model=model,
        feature_columns=cols,
        metrics=metrics,
        n_train=len(tr),
        n_val=len(va),
        categorical_levels=categorical_levels,
    )


# ---------------------------------------------------------------------------
# Grouped OOF predictions
# ---------------------------------------------------------------------------

def oof_predict(
    train_df: pd.DataFrame,
    variable: str,
    n_splits: int = 3,
) -> pd.Series:
    """
    Out-of-fold predictions grouped by init_date.

    No forecast initialization cycle is allowed to appear in both
    the fitting and held-out portions of an OOF fold.

    The classifier later consumes these predictions, so this is an
    important leakage barrier.
    """

    tr = train_df[
        train_df["variable"] == variable
    ].copy()

    tr = _clean_target(tr)

    if len(tr) < MIN_ROWS:
        return pd.Series(
            np.nan,
            index=tr.index,
            dtype=float,
        )

    cols = feature_columns(tr)

    groups = (
        tr["init_date"]
        .astype(str)
        .to_numpy()
    )

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    oof = pd.Series(
        np.nan,
        index=tr.index,
        dtype=float,
    )

    # ------------------------------------------------------------
    # Degenerate case: only one forecast cycle.
    #
    # There is no valid train/test OOF split in this situation.
    # Return predictions from a model fitted on the available cycle.
    # ------------------------------------------------------------

    if n_groups < 2:
        categorical_levels = _fit_category_levels(
            tr,
            cols,
        )

        model = _new_model(
            use_early_stopping=False,
        )

        X = _prep_X(
            tr,
            cols,
            categorical_levels,
        )

        y = tr["abs_error"].to_numpy(
            dtype=float
        )

        model.fit(
            X,
            y,
            verbose=False,
        )

        pred = np.maximum(
            model.predict(X),
            0.0,
        )

        oof.loc[tr.index] = pred

        return oof

    splits = min(
        n_splits,
        n_groups,
    )

    if splits < 2:
        raise ValueError(
            "OOF requires at least two init_date groups"
        )

    gkf = GroupKFold(
        n_splits=splits
    )

    for fit_idx, held_idx in gkf.split(
        tr,
        tr["abs_error"],
        groups,
    ):
        sub_tr = tr.iloc[
            fit_idx
        ].copy()

        sub_te = tr.iloc[
            held_idx
        ].copy()

        categorical_levels = (
            _fit_category_levels(
                sub_tr,
                cols,
            )
        )

        model = _new_model(
            use_early_stopping=False,
        )

        X_fit = _prep_X(
            sub_tr,
            cols,
            categorical_levels,
        )

        y_fit = sub_tr[
            "abs_error"
        ].to_numpy(dtype=float)

        model.fit(
            X_fit,
            y_fit,
            verbose=False,
        )

        X_held = _prep_X(
            sub_te,
            cols,
            categorical_levels,
        )

        pred = np.maximum(
            model.predict(X_held),
            0.0,
        )

        oof.loc[
            sub_te.index
        ] = pred

    return oof


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_variable_error(
    artifact: RegressorArtifact,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Predict non-negative absolute forecast error.
    """

    if df.empty:
        return np.array(
            [],
            dtype=float,
        )

    X = _prep_X(
        df,
        artifact.feature_columns,
        artifact.categorical_levels,
    )

    pred = artifact.model.predict(X)

    # Absolute error cannot physically be negative.
    return np.maximum(
        np.asarray(pred, dtype=float),
        0.0,
    )