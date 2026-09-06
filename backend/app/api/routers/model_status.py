from __future__ import annotations

from fastapi import APIRouter

from app.api import schemas
from app.ml import inference, registry
from app.realtime.broadcaster import manager
from app.services import upload_service
from app.services.region_service import _last_trained_at
from app.storage import parquet_store

router = APIRouter(prefix="/api/model", tags=["model"])


def _training_data(manifest: dict) -> dict:
    splits = manifest.get("split_cycles") or {}
    counts = {k: splits.get(k) for k in ("train", "val", "test")}
    known = [v for v in counts.values() if isinstance(v, int)]
    return {
        "cycles": sum(known) if known else None,
        "train_cycles": counts["train"],
        "val_cycles": counts["val"],
        "held_out_cycles": counts["test"],
        "canonical_rows": manifest.get("data_rows"),
        "paired_rows": manifest.get("paired_rows"),
        "first_train_date": (splits.get("train_dates") or [None])[0],
    }


@router.get("/status", response_model=schemas.ModelStatusResponse)
def model_status() -> schemas.ModelStatusResponse:
    data_volume = parquet_store.dataset_summary()
    state = inference.load_model_state()

    if state is None:
        return schemas.ModelStatusResponse(
            model_trained=False,
            training_in_progress=upload_service.training_in_progress(),
            last_training_error=upload_service.last_training_error(),
            data_volume=data_volume,
            websocket_clients=manager.connection_count,
            message=(
                "No trained model yet. Upload a dataset containing both forecasts and "
                "matching observations; training starts automatically."
            ),
        )

    manifest = state.manifest or {}
    return schemas.ModelStatusResponse(
        model_trained=True,
        current_run_id=state.run_id,
        training_data=_training_data(manifest),
        baselines=registry.load_baselines(state.run_id) or {},
        last_trained_at=_last_trained_at(),
        training_in_progress=upload_service.training_in_progress(),
        last_training_error=upload_service.last_training_error(),
        data_volume=data_volume,
        modelled_variables=manifest.get("modelled_variables", state.variables),
        skipped_variables=manifest.get("skipped_variables", {}),
        validation_metrics=inference.model_validation_metrics(state),
        thresholds={
            "bust_threshold": state.thresholds.bust_threshold,
            "p90_error": state.thresholds.p90_error,
            "risk_band_cuts": state.thresholds.risk_band_cuts,
            "threshold_percentile": state.thresholds.threshold_percentile,
        },
        explanation_method=manifest.get("shap_method"),
        websocket_clients=manager.connection_count,
    )


@router.get("/runs", tags=["model"])
def list_runs() -> dict:
    return {"current_run_id": registry.current_run_id(), "runs": registry.list_runs()}
