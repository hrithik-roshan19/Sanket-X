from app.services.regional_intelligence import (
    EXPECTED_OPERATIONAL_GEFS_MEMBERS,
    confidence_label,
    ensemble_stats,
    reliability_score,
    validate_operational_members,
)


def test_operational_gefs_member_contract():
    members = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    out = validate_operational_members(members)
    assert out["count"] == EXPECTED_OPERATIONAL_GEFS_MEMBERS
    assert out["complete"] is True


def test_operational_member_contract_rejects_bad_ids_and_wrong_count():
    out = validate_operational_members(["gec00", "gep01", "bad"])
    assert out["complete"] is False
    assert "bad" in out["invalid_members"]


def test_ensemble_stats_handles_nan_and_single_member():
    out = ensemble_stats([1.0, 3.0, float("nan")])
    assert out["member_count"] == 2
    assert out["mean"] == 2.0
    assert out["spread"] == 1.0
    assert ensemble_stats([5.0])["spread"] == 0.0


def test_reliability_score_monotonic_with_risk():
    safe = reliability_score(0.1, 0.1, 1.0)
    risky = reliability_score(0.8, 0.8, 1.0)
    assert safe is not None and risky is not None
    assert safe > risky
    assert 0 <= risky <= 100
    assert safe <= 100


def test_confidence_labels_are_stable():
    assert confidence_label(80) == "high"
    assert confidence_label(50) == "medium"
    assert confidence_label(20) == "low"
    assert confidence_label(None) == "unknown"


def test_registry_save_classifier_accepts_categorical_levels_argument():
    import inspect
    from app.ml import registry
    params = inspect.signature(registry.save_classifier).parameters
    assert "categorical_levels" in params
