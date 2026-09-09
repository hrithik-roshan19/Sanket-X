from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow direct execution:
# python .\scripts\activate_live_2026.py
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT = (
    BACKEND_DIR
    / "data"
    / "live_2026"
    / "live_forecast_canonical.parquet"
)


# ---------------------------------------------------------------------------
# Frozen 2026 live/model contract
# ---------------------------------------------------------------------------

EXPECTED_LIVE_MEMBERS = {
    "gec00",
    "gep01",
    "gep02",
    "gep03",
    "gep04",
}

EXPECTED_LEAD_DAYS = {1, 2, 3, 4, 5}

SUCCESSFUL_INGEST_STATUSES = {
    "success",
    "completed",
    "confirmed",
    "ingested",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ingest the verified 2026 GEFS forecast into Sanket-X "
            "and score it with the current trained model."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to validated live forecast parquet.",
    )

    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Ingest the live forecast without running model inference.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_live_input(df: pd.DataFrame) -> None:
    """
    Validate the prepared live 2026 canonical forecast before ingestion.

    This function does NOT:
    - fill missing values
    - interpolate
    - fabricate values
    - regrid data
    """

    required = {
        "city",
        "state",
        "region",
        "latitude",
        "longitude",
        "init_date",
        "valid_date",
        "lead_day",
        "ensemble_member_id",
        "variable",
        "value",
        "value_type",
        "verification_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Live file missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Live canonical forecast file is empty."
        )

    # ------------------------------------------------------------------
    # Forecast-only contract
    # ------------------------------------------------------------------

    value_types = set(
        df["value_type"]
        .dropna()
        .astype(str)
    )

    if value_types != {"forecast"}:
        raise ValueError(
            "Live activation accepts forecast rows only; "
            f"found {sorted(value_types)}."
        )

    # ------------------------------------------------------------------
    # Verification status consistency
    # ------------------------------------------------------------------

    verification_statuses = set(
        df["verification_status"]
        .dropna()
        .astype(str)
    )

    if len(verification_statuses) > 1:
        raise ValueError(
            "Unexpected mixed verification_status values: "
            f"{sorted(verification_statuses)}"
        )

    # ------------------------------------------------------------------
    # Member contract
    # ------------------------------------------------------------------

    members = set(
        df["ensemble_member_id"]
        .dropna()
        .astype(str)
    )

    if members != EXPECTED_LIVE_MEMBERS:
        raise ValueError(
            "Live member contract mismatch: "
            f"expected {sorted(EXPECTED_LIVE_MEMBERS)}, "
            f"found {sorted(members)}"
        )

    # ------------------------------------------------------------------
    # Lead-day contract
    # ------------------------------------------------------------------

    lead_numeric = pd.to_numeric(
        df["lead_day"],
        errors="coerce",
    )

    if lead_numeric.isna().any():
        raise ValueError(
            "Live forecast contains invalid lead_day values."
        )

    lead_days = set(
        lead_numeric.astype(int)
    )

    if lead_days != EXPECTED_LEAD_DAYS:
        raise ValueError(
            "Live lead-day contract mismatch: "
            f"expected {sorted(EXPECTED_LEAD_DAYS)}, "
            f"found {sorted(lead_days)}"
        )

    # ------------------------------------------------------------------
    # Numeric value validation
    #
    # Soil moisture can legitimately contain missing source values.
    # We preserve those NaNs exactly as received.
    # ------------------------------------------------------------------

    numeric_values = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    non_finite = ~numeric_values.notna()

    if non_finite.any():

        bad_variables = sorted(
            set(
                df.loc[
                    non_finite,
                    "variable",
                ].astype(str)
            )
            - {"soil_moisture_pct"}
        )

        if bad_variables:
            raise ValueError(
                "Non-finite live values found outside the explicitly "
                "allowed soil moisture field: "
                f"{bad_variables}"
            )

    # ------------------------------------------------------------------
    # Duplicate protection
    # ------------------------------------------------------------------

    duplicate_keys = [
        "city",
        "init_date",
        "lead_day",
        "ensemble_member_id",
        "variable",
    ]

    if df.duplicated(duplicate_keys).any():
        raise ValueError(
            "Duplicate live canonical keys detected for "
            f"{duplicate_keys}."
        )


# ---------------------------------------------------------------------------
# Mapping confirmation
# ---------------------------------------------------------------------------

def confirm_all_measurement_mappings(result, session):
    """
    Confirm only measurement mappings explicitly proposed by ingestion.
    """

    if result.status != "pending_confirmation":
        return result

    from app.ingestion.pipeline import confirm_mapping

    confirmations = []
    seen = set()

    for proposal in sorted(
        result.mapping_proposals,
        key=lambda item: item.get(
            "source_column",
            "",
        ),
    ):

        if proposal.get("role") != "measurement":
            continue

        if proposal.get("decision") != "needs_confirmation":
            continue

        variable = proposal.get(
            "suggested_variable"
        )

        value_type = proposal.get(
            "suggested_value_type"
        )

        if not variable or not value_type:
            continue

        key = (
            variable,
            value_type,
        )

        if key in seen:
            continue

        seen.add(key)

        confirmations.append(
            {
                "source_column": proposal[
                    "source_column"
                ],
                "variable": variable,
                "value_type": value_type,
                "unit_conversion": proposal.get(
                    "unit_conversion"
                ),
            }
        )

    if not confirmations:
        return result

    return confirm_mapping(
        session,
        result.batch_id,
        confirmations,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = parse_args()

    input_path = Path(
        args.input
    ).resolve()

    print("=" * 72)
    print("SANKET-X — 2026 LIVE FORECAST ACTIVATION")
    print("=" * 72)
    print(
        f"Input : {input_path}"
    )
    print(
        "Mode  : "
        + (
            "INGEST + MODEL SCORE"
            if not args.no_score
            else "INGEST ONLY"
        )
    )
    print()

    # ------------------------------------------------------------------
    # File existence
    # ------------------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Live canonical file not found: {input_path}"
        )

    # ------------------------------------------------------------------
    # Read and validate before touching database
    # ------------------------------------------------------------------

    df = pd.read_parquet(
        input_path
    )

    validate_live_input(df)

    print("Live input validation : PASS")
    print(
        f"Input rows            : {len(df):,}"
    )
    print(
        f"Cities                : {df['city'].nunique()}"
    )
    print(
        "Members               : "
        + ", ".join(
            sorted(
                df["ensemble_member_id"]
                .astype(str)
                .unique()
            )
        )
    )
    print(
        "Lead days             : "
        + ", ".join(
            str(x)
            for x in sorted(
                pd.to_numeric(
                    df["lead_day"]
                )
                .astype(int)
                .unique()
            )
        )
    )
    print()

    # ------------------------------------------------------------------
    # Existing ingestion pipeline
    # ------------------------------------------------------------------

    from app.db.base import SessionLocal
    from app.ingestion.pipeline import ingest_upload

    session = SessionLocal()

    try:

        result = ingest_upload(
            session,
            input_path,
            input_path.name,
        )

        result = confirm_all_measurement_mappings(
            result,
            session,
        )

        # Commit only after the ingestion/mapping flow succeeds.
        session.commit()

    except Exception:

        session.rollback()

        raise

    finally:

        session.close()

    # ------------------------------------------------------------------
    # Ingestion report
    # ------------------------------------------------------------------

    report = {
        "ingestion_status": result.status,
        "batch_id": result.batch_id,
        "rows_ingested": getattr(
            result,
            "row_count_ingested",
            None,
        ),
        "input": str(input_path),
        "input_rows": int(len(df)),
        "input_members": sorted(
            df[
                "ensemble_member_id"
            ]
            .astype(str)
            .unique()
        ),
        "input_lead_days": sorted(
            pd.to_numeric(
                df["lead_day"]
            )
            .astype(int)
            .unique()
            .tolist()
        ),
    }

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # The actual current ingestion pipeline returns "ingested".
    # This is a successful state, not an error.
    # ------------------------------------------------------------------

    if result.status not in SUCCESSFUL_INGEST_STATUSES:

        raise RuntimeError(
            "Live ingestion did not complete safely: "
            f"{result.status}"
        )

    print(
        f"Ingestion status      : {result.status}"
    )
    print(
        f"Batch ID              : {result.batch_id}"
    )
    print(
        f"Rows ingested         : "
        f"{getattr(result, 'row_count_ingested', None)}"
    )
    print(
        "Live ingestion        : PASS"
    )
    print()

    # ------------------------------------------------------------------
    # Ingestion-only mode
    # ------------------------------------------------------------------

    if args.no_score:

        report["status"] = "INGESTED_NO_SCORE"

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        print(
            "\nLIVE 2026 INGESTION: PASS"
        )

        return

    # ------------------------------------------------------------------
    # Load current model
    # ------------------------------------------------------------------

    from app.ml import inference

    inference.invalidate_caches()

    state = inference.load_model_state()

    if state is None:
        raise RuntimeError(
            "No current trained model is available."
        )

    # ------------------------------------------------------------------
    # Verify model/live ensemble compatibility
    # ------------------------------------------------------------------

    manifest = state.manifest or {}

    expected_members = int(
        manifest.get(
            "training_ensemble_member_count",
            0,
        )
    )

    live_member_count = len(
        EXPECTED_LIVE_MEMBERS
    )

    if expected_members != live_member_count:

        raise RuntimeError(
            "Model/live ensemble contract mismatch: "
            f"model expects {expected_members} members, "
            f"live pipeline supplies {live_member_count}."
        )

    # ------------------------------------------------------------------
    # Score latest live cycle
    # ------------------------------------------------------------------

    scored = inference.score_latest_cycle(
        state
    )

    if scored is None or scored.events.empty:

        raise RuntimeError(
            "Live forecast was ingested, but the current model "
            "produced no score."
        )

    # ------------------------------------------------------------------
    # Score report
    # ------------------------------------------------------------------

    report["model_run_id"] = state.run_id

    report[
        "model_training_ensemble_member_count"
    ] = expected_members

    report[
        "scored_init_date"
    ] = str(
        scored.init_date.date()
    )

    report[
        "scored_rows"
    ] = int(
        scored.n_rows_scored
    )

    report[
        "risk_rows"
    ] = int(
        len(scored.events)
    )

    report[
        "max_bust_probability"
    ] = float(
        scored.events[
            "bust_probability"
        ].max()
    )

    report[
        "mean_bust_probability"
    ] = float(
        scored.events[
            "bust_probability"
        ].mean()
    )

    if "risk_band" in scored.events.columns:

        report[
            "risk_band_counts"
        ] = {
            str(key): int(value)
            for key, value in (
                scored.events[
                    "risk_band"
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
                .items()
            )
        }

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    report["status"] = "PASS"

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        "\nLIVE 2026 INGEST + MODEL SCORE: PASS"
    )


if __name__ == "__main__":
    main()