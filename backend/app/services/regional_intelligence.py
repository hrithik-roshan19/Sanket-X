from __future__ import annotations

from typing import Iterable
import math

import numpy as np
import pandas as pd


EXPECTED_OPERATIONAL_GEFS_MEMBERS = 31


def validate_operational_members(members: Iterable[str]) -> dict:
    """
    Validate the complete operational GEFS ensemble catalog.

    Operational GEFS:
    - 1 control member: gec00
    - 30 perturbed members: gep01 ... gep30
    - total = 31 members

    This validates the operational catalog only.
    It does not mean that the historical training model used 31 members.
    """

    normalized = sorted(
        {
            str(member).strip()
            for member in members
            if str(member).strip()
        }
    )

    control = [
        member
        for member in normalized
        if member == "gec00"
    ]

    perturbed = [
        member
        for member in normalized
        if member.startswith("gep")
        and member[3:].isdigit()
    ]

    valid_ids = {
        "gec00",
        *[
            f"gep{i:02d}"
            for i in range(1, 31)
        ],
    }

    invalid = sorted(
        set(normalized) - valid_ids
    )

    return {
        "count": len(normalized),
        "expected": EXPECTED_OPERATIONAL_GEFS_MEMBERS,
        "complete": (
            len(normalized)
            == EXPECTED_OPERATIONAL_GEFS_MEMBERS
            and not invalid
        ),
        "control_count": len(control),
        "perturbed_count": len(perturbed),
        "invalid_members": invalid,
        "members": normalized,
    }


def ensemble_stats(values: Iterable[float]) -> dict:
    """
    Calculate finite ensemble statistics.
    """

    array = np.asarray(
        list(values),
        dtype=float,
    )

    array = array[np.isfinite(array)]

    if array.size == 0:
        return {
            "member_count": 0,
            "mean": None,
            "spread": None,
            "min": None,
            "max": None,
        }

    return {
        "member_count": int(array.size),
        "mean": float(array.mean()),
        "spread": (
            float(array.std(ddof=0))
            if array.size > 1
            else 0.0
        ),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def reliability_score(
    bust_probability: float | None,
    predicted_error: float | None,
    threshold: float | None,
    data_coverage: float = 1.0,
    member_count: int = 31,
    expected_member_count: int = EXPECTED_OPERATIONAL_GEFS_MEMBERS,
) -> float | None:
    """
    Calculate reliability on a 0-100 scale.

    Higher score = higher reliability.

    The expected member count is configurable because:
    - historical Sanket-X model currently uses 5 members
    - operational GEFS catalog contains 31 members
    """

    if bust_probability is None:
        return None

    try:
        probability = float(bust_probability)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(probability):
        return None

    probability = float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )

    try:
        coverage = float(data_coverage)
    except (TypeError, ValueError):
        coverage = 1.0

    if not math.isfinite(coverage):
        coverage = 1.0

    coverage = float(
        np.clip(
            coverage,
            0.0,
            1.0,
        )
    )

    try:
        actual_members = max(
            int(member_count),
            0,
        )
    except (TypeError, ValueError):
        actual_members = 0

    try:
        expected_members = max(
            int(expected_member_count),
            1,
        )
    except (TypeError, ValueError):
        expected_members = 1

    member_factor = float(
        np.clip(
            actual_members / expected_members,
            0.0,
            1.0,
        )
    )

    # Lower bust probability = higher reliability.
    risk_component = 1.0 - probability

    # If predicted error is unavailable, use risk component.
    error_component = risk_component

    if (
        predicted_error is not None
        and threshold is not None
    ):
        try:
            error = float(predicted_error)
            threshold_value = float(threshold)

            if (
                math.isfinite(error)
                and math.isfinite(threshold_value)
                and threshold_value > 0
            ):
                error_component = float(
                    np.clip(
                        1.0
                        - (
                            error
                            / threshold_value
                        ),
                        0.0,
                        1.0,
                    )
                )
        except (TypeError, ValueError):
            pass

    score = 100.0 * (
        0.65 * risk_component
        + 0.35 * error_component
    )

    # Data coverage penalty.
    score *= (
        0.75
        + 0.25 * coverage
    )

    # Ensemble completeness penalty.
    score *= (
        0.85
        + 0.15 * member_factor
    )

    return round(
        float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        ),
        2,
    )


def confidence_label(
    score: float | None,
) -> str:
    """
    Convert reliability score to a human-readable label.
    """

    if score is None:
        return "unknown"

    if score >= 70:
        return "high"

    if score >= 40:
        return "medium"

    return "low"


def summarize_regions(
    events: pd.DataFrame,
    thresholds: dict,
    expected_member_count: int = EXPECTED_OPERATIONAL_GEFS_MEMBERS,
) -> pd.DataFrame:
    """
    Calculate reliability/confidence for each region and lead day.
    """

    output_columns = [
        "region_id",
        "lead_time_days",
        "reliability_score",
        "confidence_label",
    ]

    if events.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    rows = []

    for (
        region_id,
        lead_time_days,
    ), group in events.groupby(
        [
            "region_id",
            "lead_time_days",
        ],
        observed=True,
    ):

        # ---------------------------------------------------------------
        # Bust probability
        # ---------------------------------------------------------------

        if (
            "bust_probability" in group.columns
            and group["bust_probability"].notna().any()
        ):
            probability = float(
                group[
                    "bust_probability"
                ]
                .dropna()
                .mean()
            )
        else:
            probability = None

        # ---------------------------------------------------------------
        # Predicted error
        # ---------------------------------------------------------------

        if (
            "pred_err" in group.columns
            and group["pred_err"].notna().any()
        ):
            predicted_error = float(
                group["pred_err"]
                .dropna()
                .mean()
            )
        else:
            predicted_error = None

        # ---------------------------------------------------------------
        # Dominant variable threshold
        # ---------------------------------------------------------------

        dominant_variable = None

        if "dominant_variable" in group.columns:

            values = (
                group[
                    "dominant_variable"
                ]
                .dropna()
                .astype(str)
            )

            if not values.empty:
                dominant_variable = (
                    values.mode().iloc[0]
                )

        threshold = None

        if (
            dominant_variable
            and dominant_variable in thresholds
        ):
            try:
                threshold = float(
                    thresholds[
                        dominant_variable
                    ]
                )
            except (TypeError, ValueError):
                threshold = None

        # Fallback to mean threshold.
        if threshold is None and thresholds:

            numeric_thresholds = []

            for value in thresholds.values():

                try:
                    numeric_value = float(
                        value
                    )

                    if math.isfinite(
                        numeric_value
                    ):
                        numeric_thresholds.append(
                            numeric_value
                        )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            if numeric_thresholds:
                threshold = float(
                    np.mean(
                        numeric_thresholds
                    )
                )

        # ---------------------------------------------------------------
        # Data coverage
        # ---------------------------------------------------------------

        if "data_coverage" in group.columns:

            coverage_values = pd.to_numeric(
                group[
                    "data_coverage"
                ],
                errors="coerce",
            )

            if coverage_values.notna().any():
                coverage = float(
                    coverage_values
                    .dropna()
                    .mean()
                )
            else:
                coverage = 1.0

        else:
            coverage = 1.0

        # ---------------------------------------------------------------
        # Ensemble member count
        # ---------------------------------------------------------------

        if (
            "ensemble_member_count"
            in group.columns
        ):

            member_values = pd.to_numeric(
                group[
                    "ensemble_member_count"
                ],
                errors="coerce",
            )

            if member_values.notna().any():
                members = int(
                    member_values
                    .dropna()
                    .max()
                )
            else:
                members = expected_member_count

        else:
            members = expected_member_count

        # ---------------------------------------------------------------
        # Reliability
        # ---------------------------------------------------------------

        score = reliability_score(
            bust_probability=probability,
            predicted_error=predicted_error,
            threshold=threshold,
            data_coverage=coverage,
            member_count=members,
            expected_member_count=expected_member_count,
        )

        rows.append(
            {
                "region_id": str(
                    region_id
                ),
                "lead_time_days": int(
                    lead_time_days
                ),
                "reliability_score": score,
                "confidence_label": (
                    confidence_label(
                        score
                    )
                ),
            }
        )

    return pd.DataFrame(rows)