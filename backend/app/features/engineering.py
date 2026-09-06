from __future__ import annotations

import numpy as np
import pandas as pd


FORECAST = "forecast"
OBSERVED = "observed"

_RATE_OF_CHANGE_VARS = {
    "pressure_hpa": "pressure_rate_of_change",
    "atmospheric_moisture_kgm2": "moisture_rate_of_change",
}

_SEASONS = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJAS",
    7: "JJAS",
    8: "JJAS",
    9: "JJAS",
    10: "ON",
    11: "ON",
}

EVENT_KEYS = [
    "region_id",
    "init_date",
    "valid_date",
    "lead_time_days",
]

MEMBER_KEYS = EVENT_KEYS + [
    "ensemble_member_id",
]


def _season(month: pd.Series) -> pd.Series:
    """Convert calendar month into meteorological season."""
    return month.map(_SEASONS).astype("category")


def _normalise_verification_status(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize observation verification status.

    Historical observations from the original archive may have NULL status.
    Those observations are already settled archive values and therefore count
    as final.

    Explicit provisional values remain provisional.
    """
    status = series.astype("string").str.strip().str.lower()

    return (
        status
        .fillna("final")
        .replace(
            {
                "": "final",
                "none": "final",
                "nan": "final",
                "<na>": "final",
                "null": "final",
            }
        )
    )


def build_training_frame(
    canonical: pd.DataFrame,
    historical_bust_freq: dict | None = None,
    require_observed: bool = True,
) -> pd.DataFrame:
    """
    Build the forecast/observation paired training frame.

    Important contracts:

    - Forecast rows remain forecast-side data.
    - Observation status is preserved.
    - Missing/legacy observation status is treated as final.
    - Provisional observations are NOT silently discarded here.
      The verification/training policy decides whether provisional data
      can participate in a particular training operation.
    - Previous verification error is never exposed as a feature.
    - Forecast-only lag/delta features are safe for live inference.
    - Rate-of-change features tolerate a single available lead day.
    """

    if canonical is None or canonical.empty:
        return pd.DataFrame()

    required_columns = {
        "region_id",
        "variable",
        "value_type",
        "value",
    }

    missing = required_columns - set(canonical.columns)

    if missing:
        raise ValueError(
            f"build_training_frame missing required columns: "
            f"{sorted(missing)}"
        )

    df = canonical.copy()

    # ---------------------------------------------------------
    # Basic normalization
    # ---------------------------------------------------------

    df = df[df["region_id"].notna()].copy()

    df["value_type"] = (
        df["value_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["variable"] = (
        df["variable"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Forecast side
    # ---------------------------------------------------------

    fc = df[df["value_type"] == FORECAST].copy()

    if fc.empty:
        return pd.DataFrame()

    for col in [
        "init_date",
        "valid_date",
    ]:
        if col in fc.columns:
            fc[col] = pd.to_datetime(
                fc[col],
                errors="coerce",
            )

    if "lead_time_days" in fc.columns:
        fc["lead_time_days"] = pd.to_numeric(
            fc["lead_time_days"],
            errors="coerce",
        )

    # The authoritative historical temporal contract is validated
    # upstream by M1/data-validation.
    #
    # We intentionally do NOT silently discard rows here because this
    # generic feature builder is also used by live verification tests
    # where the canonical verification contract may already have been
    # established by another layer.
    #
    # Invalid rows are still removed later if their required fields
    # cannot participate in a valid paired frame.

    # ---------------------------------------------------------
    # Observation side
    # ---------------------------------------------------------

    ob_cols = [
        "region_id",
        "valid_date",
        "variable",
        "value",
    ]

    has_verification_status = (
        "verification_status" in df.columns
    )

    if has_verification_status:
        ob_cols.append("verification_status")

    ob = df[df["value_type"] == OBSERVED][ob_cols].copy()

    if ob.empty:
        if require_observed:
            return pd.DataFrame()
    else:
        ob["valid_date"] = pd.to_datetime(
            ob["valid_date"],
            errors="coerce",
        )

        if has_verification_status:
            ob["verification_status"] = (
                _normalise_verification_status(
                    ob["verification_status"]
                )
            )

    # ---------------------------------------------------------
    # Aggregate observations
    # ---------------------------------------------------------

    if ob.empty:
        if require_observed:
            return pd.DataFrame()

        paired = fc.rename(
            columns={
                "value": "forecast_value",
            }
        ).copy()

        paired["observed_value"] = np.nan

        if has_verification_status:
            paired["verification_status"] = "final"

    else:
        agg = {
            "value": "mean",
        }

        if has_verification_status:
            def _aggregate_status(values: pd.Series) -> str:
                normalized = (
                    _normalise_verification_status(values)
                )

                # If any observation is provisional, preserve that
                # information instead of falsely upgrading it to final.
                if (normalized == "provisional").any():
                    return "provisional"

                return "final"

            agg["verification_status"] = _aggregate_status

        ob = (
            ob.groupby(
                [
                    "region_id",
                    "valid_date",
                    "variable",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(agg)
            .rename(
                columns={
                    "value": "observed_value",
                }
            )
        )

        # -----------------------------------------------------
        # Merge forecast with observation
        # -----------------------------------------------------

        fc = fc.rename(
            columns={
                "value": "forecast_value",
            }
        )

        # Both forecast and observation may carry a status column.
        # The observation-side status is authoritative for the paired
        # verification frame, so remove forecast-side status before merge.
        fc = fc.drop(
            columns=["verification_status"],
            errors="ignore",
        )

        paired = fc.merge(
            ob,
            on=[
                "region_id",
                "valid_date",
                "variable",
            ],
            how="inner" if require_observed else "left",
        )

    if paired.empty:
        return paired

    # ---------------------------------------------------------
    # Numeric normalization
    # ---------------------------------------------------------

    paired["forecast_value"] = pd.to_numeric(
        paired["forecast_value"],
        errors="coerce",
    )

    paired["observed_value"] = pd.to_numeric(
        paired["observed_value"],
        errors="coerce",
    )

    # Legacy archive values with missing status are final.
    if has_verification_status:
        paired["verification_status"] = (
            _normalise_verification_status(
                paired["verification_status"]
            )
        )
    else:
        paired["verification_status"] = "final"

    # ---------------------------------------------------------
    # Known soil-moisture saturation artifact
    # ---------------------------------------------------------

    sat = (
        (paired["variable"] == "soil_moisture_pct")
        & (paired["forecast_value"] >= 99.5)
    )

    paired = paired.loc[~sat].copy()

    if paired.empty:
        return paired

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    paired["abs_error"] = (
        paired["forecast_value"]
        - paired["observed_value"]
    ).abs()

    required = [
        "abs_error",
        "lead_time_days",
    ]

    if not require_observed:
        required = ["lead_time_days"]

    paired = paired.dropna(
        subset=required,
    ).copy()

    if paired.empty:
        return paired

    paired["lead_time_days"] = (
        pd.to_numeric(
            paired["lead_time_days"],
            errors="coerce",
        )
        .astype(int)
    )

    # ---------------------------------------------------------
    # Dates / temporal features
    # ---------------------------------------------------------

    paired["valid_date"] = pd.to_datetime(
        paired["valid_date"],
        errors="coerce",
    )

    paired["init_date"] = pd.to_datetime(
        paired["init_date"],
        errors="coerce",
    )

    paired = paired.dropna(
        subset=[
            "valid_date",
            "init_date",
        ]
    ).copy()

    if paired.empty:
        return paired

    paired["month"] = (
        paired["valid_date"].dt.month
    )

    paired["season"] = _season(
        paired["month"]
    )

    paired["region_id"] = (
        paired["region_id"].astype("category")
    )

    # ---------------------------------------------------------
    # Ensemble statistics
    # ---------------------------------------------------------

    grp = (
        paired
        .groupby(
            EVENT_KEYS + ["variable"],
            observed=True,
        )["forecast_value"]
    )

    paired["ensemble_spread"] = (
        grp.transform("std")
    )

    paired["ensemble_member_count"] = (
        grp.transform("count")
    )

    # A one-member ensemble has undefined std.
    # Represent that honestly as zero spread rather than inventing
    # a non-zero uncertainty estimate.
    paired["ensemble_spread"] = (
        paired["ensemble_spread"]
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Forecast rate of change
    # ---------------------------------------------------------

    paired = _add_rate_of_change(
        paired
    )

    # ---------------------------------------------------------
    # Forecast-only temporal lag
    # ---------------------------------------------------------

    paired = paired.sort_values(
        [
            "region_id",
            "init_date",
            "variable",
            "ensemble_member_id",
            "lead_time_days",
        ]
    )

    # IMPORTANT:
    # This is a forecast-value lag, NOT a previous verification error.
    #
    # Actual/observed error must never enter a live feature.
    paired["forecast_value_lag"] = (
        paired
        .groupby(
            [
                "region_id",
                "init_date",
                "variable",
                "ensemble_member_id",
            ],
            observed=True,
        )["forecast_value"]
        .shift(1)
    )

    paired["forecast_delta_from_previous_lead"] = (
        paired["forecast_value"]
        - paired["forecast_value_lag"]
    )

    # ---------------------------------------------------------
    # Concurrent forecast variables
    # ---------------------------------------------------------

    paired = _add_concurrent_variable_forecasts(
        paired
    )

    # ---------------------------------------------------------
    # Historical bust frequency
    # ---------------------------------------------------------

    if historical_bust_freq is not None:
        key = list(
            zip(
                paired["region_id"].astype(str),
                paired["season"].astype(str),
            )
        )

        paired[
            "historical_bust_frequency_region_season"
        ] = [
            historical_bust_freq.get(
                k,
                np.nan,
            )
            for k in key
        ]
    else:
        paired[
            "historical_bust_frequency_region_season"
        ] = np.nan

    return paired.reset_index(
        drop=True
    )


def _add_rate_of_change(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add forecast-only rate-of-change features.

    A single lead day naturally produces NaN because there is no
    previous lead from which a rate can be calculated. It does not
    invalidate the entire training frame.
    """

    for var, colname in _RATE_OF_CHANGE_VARS.items():
        sub = paired[
            paired["variable"] == var
        ].copy()

        if sub.empty:
            paired[colname] = np.nan
            continue

        sub = sub.sort_values(
            [
                "region_id",
                "init_date",
                "ensemble_member_id",
                "valid_date",
            ]
        )

        grouped = sub.groupby(
            [
                "region_id",
                "init_date",
                "ensemble_member_id",
            ],
            observed=True,
        )

        days = (
            grouped["valid_date"]
            .diff()
            .dt.days
            .replace(0, np.nan)
        )

        value_delta = (
            grouped["forecast_value"]
            .diff()
        )

        rate = (
            value_delta / days
        ).rename(colname)

        ev = sub[
            EVENT_KEYS + [
                "ensemble_member_id",
            ]
        ].copy()

        ev[colname] = rate.to_numpy()

        paired = paired.merge(
            ev,
            on=EVENT_KEYS + [
                "ensemble_member_id",
            ],
            how="left",
        )

    return paired


def _add_concurrent_variable_forecasts(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add forecast-only concurrent-variable columns.

    The current variable is blanked so the model does not receive
    a duplicated copy of its own forecast value.
    """

    if paired.empty:
        return paired

    wide = (
        paired.pivot_table(
            index=MEMBER_KEYS,
            columns="variable",
            values="forecast_value",
            aggfunc="mean",
            observed=True,
        )
        .add_prefix("fc_")
        .reset_index()
    )

    merged = paired.merge(
        wide,
        on=MEMBER_KEYS,
        how="left",
    )

    for var in paired["variable"].dropna().unique():
        col = f"fc_{var}"

        if col in merged.columns:
            merged.loc[
                merged["variable"] == var,
                col,
            ] = np.nan

    return merged


def compute_historical_bust_frequency(
    paired_train: pd.DataFrame,
    large_error_pct: float = 75.0,
) -> dict:
    """
    Compute historical large-error frequency by region and season.

    This helper is intended for train-only historical calculations.
    """

    if paired_train is None or paired_train.empty:
        return {}

    if "abs_error" not in paired_train.columns:
        return {}

    thresholds = {
        var: np.percentile(
            group["abs_error"].dropna(),
            large_error_pct,
        )
        for var, group in paired_train.groupby(
            "variable"
        )
        if group["abs_error"].notna().any()
    }

    if not thresholds:
        return {}

    p = paired_train.copy()

    p["is_large"] = [
        row.abs_error
        > thresholds.get(
            row.variable,
            np.inf,
        )
        for row in p.itertuples()
    ]

    rate = (
        p.groupby(
            [
                p["region_id"].astype(str),
                p["season"].astype(str),
            ]
        )["is_large"]
        .mean()
    )

    return {
        tuple(key): float(value)
        for key, value in rate.items()
    }