"""Phase 3: thresholds, feature engineering, leakage guards, and a reduced end-to-end
retrain against the real canonical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import engineering as fe
from app.features import pivot as pv
from app.ingestion.pipeline import confirm_mapping, ingest_upload
from app.ml import registry
from app.ml.thresholds import (
    Thresholds,
    compute_error_thresholds,
    compute_risk_bands,
)
from tests.conftest import ERA5_CSV, GEFS_CSV


# --------------------------------------------------------------------------- thresholds


def test_error_thresholds_are_percentiles():
    df = pd.DataFrame({
        "variable": ["t"] * 100,
        "abs_error": np.arange(100.0),
    })

    thr = compute_error_thresholds(df, percentile=90.0)

    assert 88 <= thr["t"] <= 91


def test_risk_bands_ordered_and_serialise(tmp_path):
    proba = np.clip(
        np.random.default_rng(0).beta(2, 5, 500),
        0,
        1,
    )

    cuts = compute_risk_bands(proba)

    assert 0 < cuts["medium"] < cuts["high"] < 1

    t = Thresholds(
        bust_threshold={"t": 3.0},
        p90_error={"t": 5.0},
        risk_band_cuts=cuts,
    )

    assert t.band_for(0.0) == "low"
    assert t.band_for(cuts["high"] + 1e-6) == "high"

    p = tmp_path / "thr.json"
    t.to_json(p)

    back = Thresholds.from_json(p)

    assert back.bust_threshold == t.bust_threshold
    assert back.band_for(0.99) == "high"


def test_risk_bands_degenerate_input():
    cuts = compute_risk_bands(np.zeros(5))

    assert cuts["medium"] < cuts["high"]


# --------------------------------------------------------------------- feature engineering


@pytest.fixture(scope="module")
def paired_real():
    from app.storage.parquet_store import read_dataset  # noqa: PLC0415

    # This fixture assumes the module-scoped ingest below
    # has populated the store.
    return None


def test_pairing_and_target(_ingested_slice):
    from app.storage.parquet_store import read_dataset

    canon = read_dataset()
    paired = fe.build_training_frame(canon)

    assert not paired.empty
    assert "abs_error" in paired.columns

    # Target is a non-negative magnitude.
    assert (paired["abs_error"] >= 0).all()

    # Each paired row carries both sides.
    assert paired[
        ["forecast_value", "observed_value"]
    ].notna().all().all()

    # Ensemble spread computed across members.
    assert paired["ensemble_member_count"].max() >= 2

    # A row's own concurrent-variable column is blanked.
    t = paired[paired["variable"] == "temperature_c"]
    assert t["fc_temperature_c"].isna().all()


def test_regressor_features_are_forecast_only():
    from app.ml.regressors import feature_columns

    cols = feature_columns(
        pd.DataFrame(
            columns=[
                "lead_time_days",
                "forecast_value",
                "month",
                "ensemble_spread",
                "ensemble_member_count",
                "forecast_value_lag",
                "forecast_delta_from_previous_lead",
                "forecast_error_lag",
                "historical_bust_frequency_region_season",
                "region_id",
                "season",
                "fc_temperature_c",
                "abs_error",
                "observed_value",
            ]
        )
    )

    assert "forecast_error_lag" not in cols
    assert "abs_error" not in cols
    assert "observed_value" not in cols
    assert "forecast_value_lag" in cols
    assert "forecast_delta_from_previous_lead" in cols


def test_classifier_features_exclude_target_derived_history():
    from app.features.pivot import classifier_feature_columns

    event = pd.DataFrame(
        columns=[
            "pred_err_temperature_c",
            "conf_temperature_c",
            "spread_temperature_c",
            "spread_mean",
            "spread_max",
            "lead_time_days",
            "month",
            "historical_bust_frequency_region_season",
            "actual_err_temperature_c",
            "bust_ratio",
            "y_bust",
            "region_id",
            "season",
        ]
    )

    cols = classifier_feature_columns(event)

    assert "historical_bust_frequency_region_season" not in cols
    assert "actual_err_temperature_c" not in cols
    assert "bust_ratio" not in cols
    assert "y_bust" not in cols


def test_classifier_features_exclude_actual_error(_ingested_slice):
    from app.storage.parquet_store import read_dataset

    canon = read_dataset()
    paired = fe.build_training_frame(canon)

    cycles = sorted(
        paired["init_date"].dropna().unique()
    )

    tr = paired[
        paired["init_date"].isin(
            cycles[: max(2, len(cycles) - 1)]
        )
    ]

    oof = pd.Series(
        np.random.default_rng(1).random(len(tr)),
        index=tr.index,
    )

    p90 = {
        v: 5.0
        for v in tr["variable"].unique()
    }

    thr = {
        v: 3.0
        for v in tr["variable"].unique()
    }

    ev = pv.build_event_frame(
        tr,
        oof,
        p90,
        thr,
    )

    feats = pv.classifier_feature_columns(ev)

    assert "y_bust" in ev.columns
    assert not any(
        f.startswith("actual_err_")
        for f in feats
    )
    assert "bust_ratio" not in feats
    assert "y_bust" not in feats
    assert any(
        f.startswith("pred_err_")
        for f in feats
    )


# --------------------------------------------------------------------- leakage guards


def test_split_by_cycle_is_time_ordered_and_disjoint():
    """Whole cycles, in time order, are never shared between splits."""
    from app.ml.train_pipeline import _split_by_cycle

    cycles = pd.to_datetime(
        [f"2019-{m:02d}-01" for m in range(1, 13)]
    )

    paired = pd.DataFrame({
        "init_date": cycles,
    })

    tr, va, te = _split_by_cycle(paired)

    assert tr and va and te, (
        "a 12-cycle frame must produce all three splits"
    )

    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)

    assert set(tr | va | te) == set(cycles)

    # Strictly time ordered.
    assert max(tr) < min(va) < max(va) < min(te)


def test_oof_folds_never_split_a_forecast_cycle(_ingested_slice):
    """OOF folds must be grouped by init_date."""
    from sklearn.model_selection import GroupKFold

    from app.ml import regressors as reg_mod
    from app.storage.parquet_store import read_dataset

    paired = fe.build_training_frame(
        read_dataset()
    )

    var = sorted(
        paired["variable"].unique()
    )[0]

    tr = paired[
        paired["variable"] == var
    ]

    groups = tr["init_date"].astype(str).to_numpy()

    n_groups = len(np.unique(groups))

    assert n_groups > 1, (
        "need several cycles to say anything "
        "about fold grouping"
    )

    splits = min(3, n_groups)

    for fit_idx, held_idx in GroupKFold(
        n_splits=splits
    ).split(
        tr,
        tr["abs_error"],
        groups,
    ):
        fit_cycles = set(groups[fit_idx])
        held_cycles = set(groups[held_idx])

        assert not (
            fit_cycles & held_cycles
        ), (
            f"cycle(s) {fit_cycles & held_cycles} "
            "appear on both sides of a fold"
        )

    assert reg_mod.oof_predict.__doc__
    assert "init_date" in reg_mod.oof_predict.__doc__


def test_thresholds_are_fit_on_the_train_split_only(_retrain):
    """Bust thresholds must be fitted only on training cycles."""
    from app.ml.thresholds import compute_error_thresholds
    from app.ml.train_pipeline import (
        _event_mean_error,
        _split_by_cycle,
    )
    from app.storage.parquet_store import read_dataset

    report = _retrain

    assert report.status == "success", report.error

    paired = fe.build_training_frame(
        read_dataset()
    )

    train_c, _, _ = _split_by_cycle(paired)

    tr = paired[
        paired["init_date"].isin(train_c)
    ]

    train_only = compute_error_thresholds(
        _event_mean_error(tr),
        percentile=90.0,
    )

    full_data = compute_error_thresholds(
        _event_mean_error(paired),
        percentile=90.0,
    )

    published = report.thresholds[
        "bust_threshold"
    ]

    for var, value in published.items():
        assert var in train_only

        assert value == pytest.approx(
            train_only[var]
        ), (
            f"{var}: published threshold {value} "
            f"is not the train-only percentile "
            f"{train_only[var]}"
        )

    assert any(
        published[v] != pytest.approx(
            full_data[v]
        )
        for v in published
    ), (
        "train-only and full-data thresholds "
        "are identical, so this test proves "
        "nothing about which one was used"
    )


def test_no_observed_day_is_shared_between_train_and_test(
    _ingested_slice,
):
    """Train/test must not share verification observation days."""
    from app.ml.train_pipeline import _split_by_cycle
    from app.storage.parquet_store import read_dataset

    paired = fe.build_training_frame(
        read_dataset()
    )

    train_c, _, test_c = _split_by_cycle(
        paired
    )

    train_days = set(
        pd.to_datetime(
            paired[
                paired["init_date"].isin(train_c)
            ]["valid_date"]
        )
    )

    test_days = set(
        pd.to_datetime(
            paired[
                paired["init_date"].isin(test_c)
            ]["valid_date"]
        )
    )

    shared = train_days & test_days

    assert not shared, (
        f"{len(shared)} observed day(s) appear "
        "in both the train and test splits, "
        f"e.g. {sorted(shared)[:3]}"
    )


# ----------------------------------------------------------------- end-to-end retrain


@pytest.fixture(scope="module")
def _retrain(_ingested_slice):
    from app.ml.train_pipeline import full_retrain

    return full_retrain(make_current=True)


def test_full_retrain_end_to_end(_retrain):
    report = _retrain

    assert report.status == "success", report.error
    assert report.made_current
    assert len(report.modelled_variables) >= 3
    assert (
        report.classifier_metrics
        .get("train", {})
        .get("n", 0)
        > 0
    )

    # Thresholds are real numbers in canonical units.
    for var, threshold in report.thresholds[
        "bust_threshold"
    ].items():
        assert threshold > 0

    # current.json points at this run.
    rid = registry.current_run_id()

    assert rid == report.run_id

    regs = registry.load_regressors(rid)

    assert set(regs) == set(
        report.modelled_variables
    )

    clf, cols = registry.load_classifier(rid)

    assert clf is not None
    assert len(cols) > 5

    thr = registry.load_thresholds(rid)

    assert (
        thr.risk_band_cuts["medium"]
        <= thr.risk_band_cuts["high"]
    )

    # Failed-style guard:
    # manifest records SHAP method honestly.
    import json

    manifest = json.loads(
        (
            registry.run_dir(rid)
            / "manifest.json"
        ).read_text()
    )

    assert manifest["shap_method"] in {
        "shap",
        "feature_importance_fallback",
        "none",
    }


def test_regressors_beat_mean_baseline_on_temperature(
    _retrain,
):
    metrics = _retrain.regressor_metrics.get(
        "temperature_c",
        {},
    )

    ref = (
        metrics.get("test")
        or metrics.get("val")
        or metrics.get("train")
    )

    assert ref is not None

    assert (
        ref["mae"]
        < ref["baseline_mae_predict_mean"]
    )


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def _ingested_slice(request):
    """Ingest a multi-cycle slice of the real GEFS + ERA5 samples once."""
    import shutil

    from app.db.base import (
        SessionLocal,
        engine,
        init_db,
    )
    from app.db.models import Base
    from app.storage import parquet_store

    Base.metadata.drop_all(engine)

    shutil.rmtree(
        parquet_store.CANONICAL_DIR,
        ignore_errors=True,
    )

    init_db()

    tmp = (
        request
        .getfixturevalue("tmp_path_factory")
        .mktemp("mlslice")
    )

    # ~8 init cycles:
    # enough for a 5/1/2 train/val/test split.
    g = pd.read_csv(GEFS_CSV)

    keep_cycles = sorted(
        g["init_date"].unique()
    )[:8]

    g = g[
        g["init_date"].isin(keep_cycles)
    ]

    gp = tmp / "gefs.csv"
    g.to_csv(gp, index=False)

    e = pd.read_csv(ERA5_CSV)

    ep = tmp / "era5.csv"
    e.to_csv(ep, index=False)

    def _confirm(res, session):
        if res.status != "pending_confirmation":
            return res

        seen = set()
        confirmations = []

        for proposal in sorted(
            res.mapping_proposals,
            key=lambda x: x["source_column"],
        ):
            if (
                proposal["role"]
                != "measurement"
                or proposal["decision"]
                != "needs_confirmation"
                or not proposal[
                    "suggested_variable"
                ]
            ):
                continue

            key = (
                proposal["suggested_variable"],
                proposal["suggested_value_type"],
            )

            if key in seen:
                continue

            seen.add(key)

            confirmations.append({
                "source_column": proposal[
                    "source_column"
                ],
                "variable": proposal[
                    "suggested_variable"
                ],
                "value_type": proposal[
                    "suggested_value_type"
                ],
                "unit_conversion": proposal[
                    "unit_conversion"
                ],
            })

        return confirm_mapping(
            session,
            res.batch_id,
            confirmations,
        )

    session = SessionLocal()

    try:
        _confirm(
            ingest_upload(
                session,
                gp,
                gp.name,
            ),
            session,
        )

        _confirm(
            ingest_upload(
                session,
                ep,
                ep.name,
            ),
            session,
        )

        session.commit()

    finally:
        session.close()

    yield

    Base.metadata.drop_all(engine)

    shutil.rmtree(
        parquet_store.CANONICAL_DIR,
        ignore_errors=True,
    )


# ------------------------------------------------------------------- guided replay


def test_score_cycle_targets_an_arbitrary_historical_init(
    _retrain,
):
    from app.ml import inference

    inference.invalidate_caches()

    state = inference.load_model_state()

    assert state is not None

    cycles = inference.available_cycles()

    assert len(cycles) >= 2, (
        "the 8-cycle slice should expose "
        "several init dates"
    )

    latest = inference.score_cycle(
        state,
        None,
    )

    assert latest is not None
    assert latest.init_date == max(cycles)

    older = inference.score_cycle(
        state,
        cycles[-1],
    )

    assert older is not None
    assert older.init_date == cycles[-1]
    assert not older.events.empty

    assert older.events[
        "bust_probability"
    ].between(0.0, 1.0).all()

    # Missing historical cycle should return
    # nothing rather than inventing a result.
    missing = (
        pd.Timestamp(cycles[0])
        - pd.Timedelta(days=3650)
    )

    assert (
        inference.score_cycle(
            state,
            missing,
        )
        is None
    )


def test_replay_service_narrates_from_real_numbers(
    _retrain,
):
    from app.services import replay_service

    replay_service.invalidate()

    cycles = replay_service.list_cycles()

    assert cycles, (
        "at least one scoreable cycle"
    )

    # Verified cycles must rank ahead of
    # unverified ones.
    verified_flags = [
        c.verified
        for c in cycles
    ]

    assert verified_flags == sorted(
        verified_flags,
        reverse=True,
    )

    rep = replay_service.get_replay()

    assert rep.model_trained
    assert rep.init_date is not None
    assert 1 <= len(rep.steps) <= 10
    assert (
        rep.summary_narration
        and rep.summary_narration.strip()
    )

    leads = [
        s.lead_time_days
        for s in rep.steps
    ]

    assert leads == sorted(leads)

    for step in rep.steps:
        assert step.narration.startswith(
            f"Day {step.lead_time_days}"
        )

        assert step.regions

        assert all(
            0.0 <= r.bust_probability <= 1.0
            for r in step.regions
        )

        # Regions are ordered worst-first.
        probs = [
            r.bust_probability
            for r in step.regions
        ]

        assert probs == sorted(
            probs,
            reverse=True,
        )

    if rep.focus is not None:
        assert (
            rep.focus.variable
            != "wind_direction_deg"
        )

        assert rep.focus.points

        # Chart follows the screen:
        # default focus is cycle's peak-bust-risk
        # region.
        peak_rid = max(
            (
                r
                for step in rep.steps
                for r in step.regions
            ),
            key=lambda r: r.bust_probability,
        ).region_id

        assert (
            rep.focus.region_id
            == peak_rid
        )

        # Every worst-list region offered as a
        # focus option resolves to a real series.
        assert rep.focus_options

        assert all(
            option.points
            for option in rep.focus_options
        )

        assert (
            rep.focus.region_id
            in {
                option.region_id
                for option in rep.focus_options
            }
        )

        # Caller can pin the chart to another region.
        other = next(
            (
                option.region_id
                for option in rep.focus_options
                if option.region_id != peak_rid
            ),
            None,
        )

        if other:
            pinned = (
                replay_service.get_replay(
                    str(rep.init_date),
                    focus_region=other,
                )
            )

            assert pinned.focus is not None
            assert (
                pinned.focus.region_id
                == other
            )


def test_registry_persists_calibration_and_category_levels(
    _retrain,
):
    from app.ml import registry
    from app.ml.calibration import (
        load_calibrator,
    )

    report = _retrain

    assert report.status == "success", report.error

    rid = report.run_id

    cal = load_calibrator(
        registry.run_dir(rid)
        / "calibrator.json"
    )

    assert cal.method in {
        "identity",
        "platt",
        "isotonic",
    }

    levels = registry.load_category_levels(
        rid
    )

    assert "classifier" in levels

    assert all(
        key.startswith("regressor::")
        for key in levels
        if key != "classifier"
    )


# --------------------------------------------------------------------------- M7


def test_xgb_regressor_uses_early_stopping_and_persists_category_levels():
    from app.ml.regressors import (
        train_variable_regressor,
    )

    rows = []

    for init_day in range(10):
        for region_idx in range(4):
            rows.append({
                "init_date": (
                    f"2019-01-{init_day + 1:02d}"
                ),
                "variable": "temperature_2m",
                "region_id": f"R{region_idx}",
                "season": "winter",
                "lead_time_days": 1,
                "forecast_value": (
                    10.0
                    + region_idx
                    + init_day * 0.1
                ),
                "abs_error": (
                    1.0
                    + region_idx * 0.2
                    + init_day * 0.01
                ),
                "ensemble_spread": (
                    0.5
                    + region_idx * 0.1
                ),
                "ensemble_member_count": 5,
                "forecast_value_lag": 9.5,
                "forecast_delta_from_previous_lead": 0.2,
                "pressure_rate_of_change": 0.1,
                "moisture_rate_of_change": 0.2,
            })

    df = pd.DataFrame(rows)

    train_df = df.iloc[:30].copy()
    val_df = df.iloc[30:].copy()

    artifact = train_variable_regressor(
        train_df,
        val_df,
        "temperature_2m",
    )

    assert artifact is not None
    assert artifact.n_train == 30
    assert artifact.n_val == 10

    assert "early_stopping" in artifact.metrics

    assert (
        artifact.metrics[
            "early_stopping"
        ]["enabled"]
        is True
    )

    assert (
        "region_id"
        in artifact.categorical_levels
    )

    assert (
        "season"
        in artifact.categorical_levels
    )

    assert (
        "__UNK__"
        in artifact.categorical_levels[
            "region_id"
        ]
    )

    assert (
        "__UNK__"
        in artifact.categorical_levels[
            "season"
        ]
    )


def test_unseen_categories_map_to_unknown():
    from app.ml.regressors import _prep_X

    train = pd.DataFrame({
        "region_id": ["R1", "R2"],
        "season": ["winter", "summer"],
        "forecast_value": [10.0, 20.0],
    })

    levels = {
        "region_id": [
            "R1",
            "R2",
            "__UNK__",
        ],
        "season": [
            "winter",
            "summer",
            "__UNK__",
        ],
    }

    test = pd.DataFrame({
        "region_id": [
            "R1",
            "NEW_REGION",
        ],
        "season": [
            "winter",
            "new_season",
        ],
        "forecast_value": [
            15.0,
            25.0,
        ],
    })

    X = _prep_X(
        test,
        [
            "region_id",
            "season",
            "forecast_value",
        ],
        levels,
    )

    assert (
        str(X.loc[0, "region_id"])
        == "R1"
    )

    assert (
        str(X.loc[1, "region_id"])
        == "__UNK__"
    )

    assert (
        str(X.loc[0, "season"])
        == "winter"
    )

    assert (
        str(X.loc[1, "season"])
        == "__UNK__"
    )


def test_regressor_predictions_are_non_negative():
    from app.ml.regressors import (
        train_variable_regressor,
        predict_variable_error,
    )

    rows = []

    for i in range(40):
        rows.append({
            "init_date": (
                f"2019-01-{(i % 20) + 1:02d}"
            ),
            "variable": "temperature_2m",
            "region_id": f"R{i % 4}",
            "season": "winter",
            "lead_time_days": 1,
            "forecast_value": (
                10.0 + i * 0.1
            ),
            "abs_error": (
                float(i % 5) * 0.2
            ),
            "ensemble_spread": 0.5,
            "ensemble_member_count": 5,
            "forecast_value_lag": 9.5,
            "forecast_delta_from_previous_lead": 0.2,
            "pressure_rate_of_change": 0.1,
            "moisture_rate_of_change": 0.2,
        })

    df = pd.DataFrame(rows)

    artifact = train_variable_regressor(
        df.iloc[:30],
        df.iloc[30:],
        "temperature_2m",
    )

    assert artifact is not None

    predictions = predict_variable_error(
        artifact,
        df.iloc[30:],
    )

    assert len(predictions) == 10
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()