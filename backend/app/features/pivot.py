from __future__ import annotations

import numpy as np
import pandas as pd


CONF_FLOOR = 0.0

EVENT_KEYS = [
    "region_id",
    "init_date",
    "valid_date",
    "lead_time_days",
]


def _season(month: pd.Series) -> pd.Series:
    """
    Convert calendar month into meteorological season.
    """
    return pd.Series(
        np.select(
            [
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8, 9]),
                month.isin([10, 11]),
            ],
            [
                "pre_monsoon",
                "monsoon",
                "post_monsoon",
            ],
            default="winter",
        ),
        index=month.index,
    )


def build_event_frame(
    paired: pd.DataFrame,
    pred_err: pd.Series,
    p90_error: dict,
    bust_threshold: dict | None,
    historical_bust_freq: dict | None = None,
) -> pd.DataFrame:
    """
    Build event-level prediction frame from paired
    forecast/observation data.

    Important:
    - Forecast values are preserved as fc_mean_<variable>.
    - Actual observations and actual errors remain available
      for verification/training use.
    - Forecast values are never replaced by actual errors.
    - Event intelligence can safely consume forecast-derived values.
    """

    if paired is None or paired.empty:
        return pd.DataFrame()

    required_columns = {
        "region_id",
        "init_date",
        "valid_date",
        "lead_time_days",
        "variable",
        "forecast_value",
        "observed_value",
        "ensemble_spread",
        "ensemble_member_count",
    }

    missing = required_columns - set(paired.columns)

    if missing:
        raise ValueError(
            "build_event_frame missing required columns: "
            f"{sorted(missing)}"
        )

    df = paired.copy()

    # ---------------------------------------------------------
    # Prediction error
    # ---------------------------------------------------------

    df["pred_err"] = (
        pred_err.reindex(df.index).to_numpy()
    )

    # P90 error threshold for each variable.
    df["p90"] = df["variable"].map(p90_error)

    # Confidence:
    # lower predicted error -> higher confidence.
    #
    # If p90 is missing/zero, confidence safely becomes NaN
    # instead of producing an invalid infinity.
    p90 = pd.to_numeric(
        df["p90"],
        errors="coerce",
    )

    pred = pd.to_numeric(
        df["pred_err"],
        errors="coerce",
    )

    df["conf"] = np.where(
        p90.gt(0),
        1.0 - pred / p90,
        np.nan,
    )

    df["conf"] = pd.Series(
        df["conf"],
        index=df.index,
    ).clip(
        CONF_FLOOR,
        1.0,
    )

    # ---------------------------------------------------------
    # Aggregate ensemble members to event/variable level
    # ---------------------------------------------------------

    em = (
        df.groupby(
            EVENT_KEYS + ["variable"],
            observed=True,
        )
        .agg(
            # Forecast mean is deliberately preserved.
            fc_mean=("forecast_value", "mean"),

            # Observation retained only for verification.
            obs=("observed_value", "mean"),

            # Ensemble spread.
            spread=("ensemble_spread", "mean"),

            # Model-predicted absolute error.
            pred_err=("pred_err", "mean"),

            # Confidence.
            conf=("conf", "mean"),

            # Number of ensemble members.
            ensemble_member_count=(
                "ensemble_member_count",
                "max",
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Actual verification error
    # ---------------------------------------------------------
    #
    # This is verification/training information.
    # It must NEVER become a live classifier feature.
    # ---------------------------------------------------------

    em["actual_err"] = (
        em["fc_mean"] - em["obs"]
    ).abs()

    # ---------------------------------------------------------
    # Pivot variables into event-level columns
    # ---------------------------------------------------------
    #
    # Creates:
    #   fc_mean_<variable>
    #   pred_err_<variable>
    #   conf_<variable>
    #   spread_<variable>
    #   actual_err_<variable>
    # ---------------------------------------------------------

    pe = em.pivot_table(
        index=EVENT_KEYS,
        columns="variable",
        values=[
            "fc_mean",
            "pred_err",
            "conf",
            "spread",
            "actual_err",
        ],
        observed=True,
    )

    # Flatten MultiIndex columns.
    pe.columns = [
        f"{a}_{b}"
        for a, b in pe.columns
    ]

    pe = pe.reset_index()

    # ---------------------------------------------------------
    # Temporal fields
    # ---------------------------------------------------------

    pe["valid_date"] = pd.to_datetime(
        pe["valid_date"],
        errors="coerce",
    )

    pe["month"] = pe["valid_date"].dt.month

    pe["season"] = _season(
        pe["month"]
    )

    pe["region_id"] = pe["region_id"].astype(
        "category"
    )

    pe["lead_time_days"] = pd.to_numeric(
        pe["lead_time_days"],
        errors="coerce",
    ).astype("Int64")

    # ---------------------------------------------------------
    # Ensemble spread summary
    # ---------------------------------------------------------

    spread_cols = [
        c
        for c in pe.columns
        if c.startswith("spread_")
    ]

    if spread_cols:
        pe["spread_mean"] = pe[
            spread_cols
        ].mean(axis=1)

        pe["spread_max"] = pe[
            spread_cols
        ].max(axis=1)
    else:
        pe["spread_mean"] = np.nan
        pe["spread_max"] = np.nan

    # ---------------------------------------------------------
    # Historical bust frequency
    # ---------------------------------------------------------

    if historical_bust_freq is not None:

        key = list(
            zip(
                pe["region_id"].astype(str),
                pe["season"].astype(str),
            )
        )

        pe[
            "historical_bust_frequency_region_season"
        ] = [
            historical_bust_freq.get(
                k,
                np.nan,
            )
            for k in key
        ]

    else:

        pe[
            "historical_bust_frequency_region_season"
        ] = np.nan

    # ---------------------------------------------------------
    # Historical bust target
    # ---------------------------------------------------------
    #
    # Verification/training only.
    # Never returned by classifier_feature_columns().
    # ---------------------------------------------------------

    if bust_threshold:

        ratios = []

        for var, thr in bust_threshold.items():

            col = f"actual_err_{var}"

            if (
                col in pe.columns
                and thr is not None
                and np.isfinite(thr)
                and thr > 0
            ):
                ratios.append(
                    pe[col] / float(thr)
                )

        if ratios:

            pe["bust_ratio"] = pd.concat(
                ratios,
                axis=1,
            ).max(axis=1)

        else:

            pe["bust_ratio"] = np.nan

        pe["y_bust"] = (
            pd.to_numeric(
                pe["bust_ratio"],
                errors="coerce",
            )
            .ge(1.0)
            .fillna(False)
            .astype(int)
        )

    return pe


def classifier_feature_columns(
    event_df: pd.DataFrame,
) -> list[str]:
    """
    Return only forecast-available features that are safe
    for the bust classifier.

    Allowed forecast/model-derived features:
        fc_mean_*
        pred_err_*
        conf_*
        spread_*
        spread_mean
        spread_max
        ensemble_member_count
        lead_time_days
        month
        season
        region_id

    Explicitly forbidden:
        actual_err_*
        observed_value
        observed_*
        obs
        obs_*
        bust_ratio
        y_bust
        verification_status
        historical target-derived features

    The function is intentionally deterministic so that
    training and inference use the same feature contract.
    """

    if event_df is None or event_df.empty:
        return []

    forbidden_exact = {
        "observed_value",
        "obs",
        "actual_err",
        "bust_ratio",
        "y_bust",
        "verification_status",
    }

    forbidden_prefixes = (
        "actual_err_",
        "observed_",
        "obs_",
    )

    allowed_exact = {
        "lead_time_days",
        "month",
        "season",
        "region_id",
        "ensemble_member_count",
        "spread_mean",
        "spread_max",
    }

    allowed_prefixes = (
        "fc_mean_",
        "pred_err_",
        "conf_",
        "spread_",
    )

    selected: list[str] = []

    for column in event_df.columns:

        name = str(column)

        # Never allow verification/target-derived fields.
        if name in forbidden_exact:
            continue

        if any(
            name.startswith(prefix)
            for prefix in forbidden_prefixes
        ):
            continue

        # Explicitly safe scalar features.
        if name in allowed_exact:
            selected.append(name)
            continue

        # Forecast/model-derived variable features.
        if any(
            name.startswith(prefix)
            for prefix in allowed_prefixes
        ):
            selected.append(name)
            continue

    # Stable ordering is important for model artifact
    # reproducibility and inference compatibility.
    return sorted(
        dict.fromkeys(selected)
    )