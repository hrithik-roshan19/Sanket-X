"""
Focused replay diagnostic test.

The actual retraining fixture is shared through tests/conftest.py.
This test therefore works independently of test_ml.py fixture scope.
"""

from __future__ import annotations


def test_debug_replay_focus(_retrain):
    from app.ml import inference
    from app.services import replay_service

    # Ensure no stale replay/model cache survives from another test.
    inference.invalidate_caches()
    replay_service.invalidate()

    state = inference.load_model_state()

    print(
        "\nMODEL LOADED:",
        state is not None,
    )

    assert state is not None, (
        "Model state could not be loaded "
        "after the real retraining fixture."
    )

    cycles = replay_service.list_cycles()

    print(
        "CYCLES:",
        cycles,
    )

    assert cycles, (
        "No scoreable cycles after retraining."
    )

    target = str(
        cycles[0].init_date
    )

    scored = inference.score_cycle(
        state,
        target,
    )

    assert scored is not None
    assert not scored.events.empty

    events = scored.events
    per_variable = scored.per_variable

    peak = events.loc[
        events["bust_probability"].idxmax()
    ]

    print(
        "\n========== PEAK EVENT =========="
    )

    peak_columns = [
        "region_id",
        "lead_time_days",
        "bust_probability",
        "dominant_variable",
    ]

    available_peak_columns = [
        column
        for column in peak_columns
        if column in peak.index
    ]

    print(
        peak[
            available_peak_columns
        ].to_string()
    )

    peak_region = str(
        peak["region_id"]
    )

    print(
        "\n========== PEAK REGION PER_VARIABLE =========="
    )

    peak_per_variable = per_variable[
        per_variable["region_id"].astype(str)
        == peak_region
    ]

    if peak_per_variable.empty:
        print(
            "NO per_variable rows for peak region"
        )
    else:
        columns = [
            "region_id",
            "lead_time_days",
            "variable",
            "predicted_value",
            "observed_value",
        ]

        available_columns = [
            column
            for column in columns
            if column in peak_per_variable.columns
        ]

        print(
            peak_per_variable[
                available_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\n========== ALL PER_VARIABLE REGIONS =========="
    )

    if (
        not per_variable.empty
        and "region_id"
        in per_variable.columns
    ):
        print(
            sorted(
                per_variable[
                    "region_id"
                ]
                .astype(str)
                .unique()
                .tolist()
            )
        )
    else:
        print([])

    print(
        "\n========== BUILD FOCUS =========="
    )

    default_focus, options = (
        replay_service._build_focus(
            scored,
            state,
            None,
        )
    )

    print(
        "DEFAULT FOCUS:",
        (
            default_focus.region_id
            if default_focus
            else None
        ),
    )

    print(
        "OPTIONS:",
        [
            option.region_id
            for option in options
        ],
    )

    print(
        "\nEXPECTED PEAK:",
        peak_region,
    )

    # The replay focus should resolve to the highest-risk region.
    if default_focus is not None:
        assert (
            default_focus.region_id
            == peak_region
        )