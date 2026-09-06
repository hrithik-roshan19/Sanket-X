# Sanket — Implementation Plan

## Context

This is Smart India Hackathon 2026 Problem Statement 26079 (NCMRWF / Ministry of Earth
Sciences): detect where and when medium-range (Day 1–10) NWP forecasts are likely to "bust"
(show large error) across Indian regions, and explain why. The deliverable is a working
full-stack prototype — FastAPI backend running two XGBoost heads (error regression +
bust-probability classification), a React dashboard centered on a clickable India map, and a
format-agnostic ingestion layer, since the exact columns/format of the real dataset aren't known
yet. The project directory is currently empty — this is a from-scratch build.

The overriding constraint, repeated throughout the source spec, is honesty: no synthetic/mocked
numbers anywhere in the shipped app, ever, even as a fallback. Every number on screen must trace
back to a real upload and a real trained model, or the UI must say plainly that neither exists
yet. This shapes almost every design choice below (background retrain instead of stale
placeholder state, explicit `model_trained: false` states, thresholds derived from data rather
than hardcoded, etc.).

**Locked decisions** (confirmed with the user):
- NetCDF (.nc) ingestion is a stretch goal — build CSV/TSV/XLSX/JSON robustly first.
- Persistence: SQLite (metadata) + local parquet (canonical dataset) + XGBoost native
  `save_model` (models). No external DB service to install/host.

**Environment note**: local Python is 3.9.6, which is old enough to constrain library versions.
At Phase 1 kickoff, check for Python 3.10/3.11 (`python3.11 --version`); if unavailable, pin
`xgboost>=2.0,<2.1`, `shap>=0.44`, `pandas>=2.0,<2.2`, `scikit-learn>=1.3`, all of which still
support 3.9. Node v26 / npm 11 are already available for the frontend.

The user wants to check in **between phases**, not just at the end — this plan is phased
accordingly, and Phase 1 itself (stack + folder layout + canonical schema, below) is the
artifact to be approved before any ingestion/ML/API code is written. A real sample dataset from
the user is a **hard blocking dependency for Phase 2** — never invent example rows to unblock it.

---

## Tech Stack

**Backend**: Python (3.10/3.11 preferred), FastAPI + uvicorn, pandas/numpy, xgboost,
scikit-learn (splits/metrics), shap (feature_importances_ fallback), SQLAlchemy + SQLite,
pyarrow (parquet), openpyxl (xlsx), shapely (point-in-polygon region resolution), rapidfuzz
(fuzzy header matching).

**Frontend**: React + Vite + TypeScript, TanStack Query (REST/caching), Zustand (live-update
state), react-simple-maps + a vendored India-states TopoJSON (Datameet's `maps-india` repo —
reliable, commonly used for Indian administrative boundaries; converted/trimmed with mapshaper
if needed) for the choropleth, Recharts for time-series/curve charts.

**Realtime**: WebSocket (`/ws`) broadcasting typed events; SSE is a trivial one-way fallback if
WS proves troublesome mid-hackathon.

---

## Folder Structure

```
backend/
  app/
    main.py                      # FastAPI app, CORS, router includes, startup (geo index, current-model pointer, SQLite init)
    config.py                    # pydantic BaseSettings: DATA_DIR, DB_PATH, MODEL_DIR, RAW_UPLOAD_DIR, CORS_ORIGINS
    db/
      base.py                    # SQLAlchemy engine + SessionLocal + Base
      models.py                  # UploadBatch, ColumnMapping, SourceProfile, TrainingRun, Alert
      crud.py
    ingestion/
      parsers.py                 # read_csv/tsv/xlsx/json -> (DataFrame, detected_format)
      canonical_schema.py        # CanonicalVariable enum, ValueType enum, CanonicalRow model
      schema_mapper.py           # SchemaMapper: synonym dict + range heuristics + confidence scoring + profile reuse
      pipeline.py                # parse -> map -> (confirm) -> canonicalize -> append parquet -> record lineage
      netcdf_stretch.py          # stub only for now; xarray-based grid flattening, wired later
    features/
      engineering.py             # per-variable error features, ensemble spread, rate-of-change proxies, bust frequency
      pivot.py                   # long-to-wide pivot for classifier grain
    ml/
      regressors.py              # train_variable_regressor(), predict_variable_error()
      classifier.py              # train_bust_classifier(), predict_bust_probability()
      thresholds.py              # compute_error_thresholds(), compute_risk_bands(), (de)serialization
      explain.py                 # SHAP wrapper + feature_importances_ fallback, top_factors_for(region, lead_time)
      registry.py                # save/load xgboost native models, run_id versioning, "current" pointer
      train_pipeline.py          # full retrain orchestration
    realtime/
      broadcaster.py             # WS ConnectionManager, broadcast(event, payload)
      events.py                  # event envelope + payload schemas
    api/
      routers/{upload,regions,alerts,model_status,ws}.py
      schemas.py                 # all Pydantic response models
    services/{upload_service,region_service,alert_service}.py
    storage/
      parquet_store.py           # append_batch(), read_dataset() via pyarrow.dataset
    utils/
      geo.py                     # point-in-polygon region resolution (shapely + STRtree)
      india_state_codes.py       # name/alias -> ISO 3166-2:IN region_id lookup
      logging.py
  data/
    raw_uploads/{batch_id}/<original filename>
    canonical/batch_id=<uuid>/part-0.parquet
    models/{run_id}/{variable}_regressor.json, classifier.json, thresholds.json, shap_summary.parquet
    models/current.json          # {"run_id": ...}, written atomically only after a full successful retrain
    geo/india_states.geojson     # vendored (Datameet maps-india)
  metadata.db                    # SQLite
  tests/{test_schema_mapper.py, test_geo.py, test_pivot.py}
  requirements.txt / .env.example

frontend/
  src/
    main.tsx / App.tsx
    api/{client.ts, types.ts, upload.ts, regions.ts, alerts.ts, modelStatus.ts}
    hooks/{useRegions, useRegionDetail, useAlerts, useModelStatus}.ts   # TanStack Query
    hooks/useLiveSocket.ts        # WS -> Zustand + queryClient.invalidateQueries
    store/liveStore.ts            # Zustand: connectionStatus, lastEventType, lastEventAt
    components/
      map/{IndiaChoroplethMap,MapLegend,LeadDaySelector}.tsx
      detail/{RegionDetailPanel,VariableTrajectoryChart,BustProbabilityCurve,ShapFactorsList,AnalogCasesList}.tsx
      dashboard/{AlertsPanel,BustSummaryChart}.tsx
      upload/{UploadDropzone,ColumnMappingConfirmModal}.tsx
      common/{EmptyState,LoadingState,StatusBadge}.tsx
    pages/DashboardPage.tsx
    assets/geo/india_states.topojson
  vite.config.ts / tsconfig.json / package.json / .env.example
```

---

## Canonical Internal Data Schema

The canonical dataset is a **long-format fact table** — one row per
`(variable, value_type, region, valid_date[, init_date, lead_time_days, ensemble_member_id])`.
Forecast and observed values are kept as separate rows, joined later at feature-engineering time
on `(region_id, valid_date, variable)`, not pre-paired at ingestion — this matches how forecast
verification actually works (many init_date/lead_time forecasts verify against one observed
value for a given valid_date) and avoids the ingestion layer having to guess how a specific file
paired its columns.

**Moisture is split into two canonical variables**, not one: `atmospheric_moisture_kgm2` (TCWV /
precipitable water) and `soil_moisture_pct` (volumetric). They're physically different
quantities from different instruments/levels; conflating them would corrupt both the regressor's
scale and any instability-proxy feature engineering. Their value ranges overlap with each other
and with humidity, so header-synonym match is weighted over range heuristics for this pair, and
weak-confidence cases route to manual confirmation rather than being guessed.

| variable | unit | plausible range |
|---|---|---|
| `rainfall_mm` | mm | ≥0, heavy right-skew |
| `temperature_c` | °C | −50 to 60 |
| `humidity_pct` | % RH | 0–100 |
| `pressure_hpa` | hPa | 800–1100 |
| `atmospheric_moisture_kgm2` | kg/m² (TCWV) | 0–90 |
| `soil_moisture_pct` | % volumetric | 0–100 |
| `wind_speed_ms` | m/s | 0–100 |
| `wind_direction_deg` | degrees, meteorological "from" convention | 0–360 |

**Wind u/v derivation**: if only `u_wind`/`v_wind` are mapped, derive before emitting canonical
rows: `speed = sqrt(u**2 + v**2)`, `direction = (270 - degrees(atan2(v, u))) % 360`, emitting two
canonical rows; raw u/v kept as lineage-only extras, not fed to models.

**Row schema (parquet columns)**:

| field | type | notes |
|---|---|---|
| `record_id` | string (uuid4) | synthetic PK |
| `upload_batch_id` | string | FK to `UploadBatch` |
| `source_file` / `source_column` | string | audit trail back to the original upload |
| `variable` | category | one of the 8 canonical variables |
| `value_type` | category | `forecast` \| `observed` |
| `value` | float64 | in canonical unit |
| `region_id` | string, nullable | ISO 3166-2:IN code, e.g. `IN-MH` |
| `region_name` | string | normalized display name |
| `lat` / `lon` | float64, nullable | if provided |
| `init_date` | date, nullable | forecast rows only |
| `valid_date` | date | required |
| `lead_time_days` | Int16, nullable | forecast rows only; `valid_date − init_date` |
| `ensemble_member_id` | string, nullable | `"control"`, `"mean"`, or member index |
| `mapping_confidence` | float64 | copied from the schema-mapper decision, for audit |
| `ingested_at` | timestamp | |

**Time reconciliation**: any 2 of `{init_date, valid_date, lead_time_days}` determine the third.
Forecast rows auto-derive the missing field only when ≥2 of 3 are present; if only 1 is present,
that column routes to manual confirmation rather than being guessed. Observed rows need only
`valid_date`.

**Location resolution**: a textual region/state column normalizes via
`utils/india_state_codes.py` (handles aliases like Odisha/Orissa, Uttarakhand/Uttaranchal)
directly to `region_id`. Lat/lon-only data resolves via point-in-polygon against the vendored
India states GeoJSON (shapely + STRtree spatial index built once at startup), falling back to
nearest-polygon-centroid for points that miss all polygons (coastal offsets). `region_id` uses
ISO 3166-2:IN codes so it matches the state property key used to color the choropleth (exact
property name confirmed once the vendored file is in place in Phase 1).

**Lineage**: `upload_batch_id`/`source_file`/`source_column` on every parquet row, plus a
`ColumnMapping` audit row per confirmed/auto column mapping per batch in SQLite. Each batch's
parquet partition and mapping record are independent and additive — re-uploads extend the
dataset, nothing is overwritten.

---

## Schema-Mapping Algorithm

`ingestion/schema_mapper.py`, class `SchemaMapper`.

**Per-column scoring**: normalize header → `synonym_score` (exact 1.0, substring/token 0.7,
fuzzy via rapidfuzz only above 0.6) → `range_score` (fraction of up to 500 sampled numeric values
falling in each candidate variable's plausible range) → `confidence = clip(0.65*synonym_score +
0.35*range_score, 0, 1)` (synonym weighted higher since ranges overlap heavily, e.g. humidity vs.
soil_moisture) → `ambiguity_gap = confidence(top) − confidence(2nd)`. `value_type`
(forecast/observed) is scored the same way against its own synonym dictionary; with no signal it
is left **undetermined**, never silently defaulted.

**Decision thresholds**:
- **Auto-accept**: `confidence ≥ 0.85` AND `ambiguity_gap ≥ 0.15` AND (`value_type` n/a or its
  own confidence ≥ 0.7).
- **Route to confirm-mapping UI**: `confidence ∈ [0.4, 0.85)`, or `ambiguity_gap < 0.15`, or
  `value_type` undetermined, or a sanity check fails (e.g. a humidity-mapped column has too many
  values outside 0–100).
- **Unmapped**: `confidence < 0.4` — shown to the user as excluded, never silently dropped.

**Confirm-mapping payload**: backend returns per ambiguous column: source header, sample values,
suggested variable + value_type, confidence, ambiguity_gap, alternatives. The frontend modal
pre-fills dropdowns from the top suggestion with a confidence badge; submitting triggers
canonicalization + training.

**Reuse across re-uploads (`SourceProfile`)**: fingerprint = hash of sorted, normalized header
list. Exact/near match (Jaccard ≥ 0.9) → auto-apply the stored confirmed mapping with a
dismissible "auto-applied from a similar previous file" banner. Partial match (0.6–0.9) →
pre-fill from the closest profile but still route differing columns through confirmation. No
match → full mapper run. Every manual confirmation upserts the fingerprint → mapping into
`SourceProfile`.

---

## ML Architecture

**One XGBRegressor per canonical variable actually present** (dynamically discovered, not a
hardcoded list of 8) rather than multi-output — variables have very different error
distributions and row counts, and independent models degrade gracefully when a variable is
sparse. A variable with fewer than ~30 paired forecast/observed rows is skipped and surfaced (not
hidden) in `/api/model/status`.

**Regressor target**: `abs(forecast_value − observed_value)` for the joined pair.

**Regressor features** (per joined `region × variable × valid_date × lead_time_days × init_date
× ensemble_member_id` row): `lead_time_days`, `region_id` (native category,
`enable_categorical=True`), season/month, `forecast_value`, `ensemble_spread` +
`ensemble_member_count` (NaN if no ensemble — xgboost handles missing natively),
`pressure_rate_of_change` / `moisture_rate_of_change` (instability proxies vs. previous available
valid_date), `historical_bust_frequency_region_season`, `forecast_error_lag` (previous
lead_time's error, same init_date/region/variable), and concurrent forecast values of other
variables for the same event (NaN where absent).

**Per-variable confidence**: `clip(1 − predicted_error / p90_error, 0, 1)`, where `p90_error` is
the 90th percentile of that variable's actual training-split error, persisted at training time.

**Classifier grain — long-to-wide pivot** (`features/pivot.py`): one row per forecast *event*
`(region_id, valid_date, lead_time_days, init_date)`, pivoting per-variable values into columns
(`err_rainfall_mm`, `err_temperature_c`, …), NaN where a variable is absent for that event.

**Leakage guard**: the classifier is trained on the regressors' **out-of-fold predicted** error
(via cross-validation during training), not actual error — using actual error would diverge
train/inference feature distributions since actual error is unknown at real inference time.

**Bust label** (training only): an event is a bust if any variable's actual error exceeds that
variable's own data-driven threshold (`max over variables of actual_error / threshold ≥ 1.0`).

**Classifier features**: pivoted wide table of regressor-predicted (OOF) per-variable error +
confidence + ensemble-spread aggregates + `region_id`, `lead_time_days`, season,
`historical_bust_frequency_region_season`.

**Thresholds & risk bands**: per-variable bust threshold = a percentile (default 90th,
configurable) of that variable's actual training-split error — self-calibrating per variable's
own units. Risk bands (Low/Medium/High) are percentile bands of the classifier's
predicted-probability distribution on the validation split. Both persisted in
`data/models/{run_id}/thresholds.json` alongside frozen feature-column order for each model, and
the active run is only exposed via `data/models/current.json`, written atomically after a full
successful retrain (a failed retrain never leaves the API serving a half-written model).

**SHAP**: `shap.TreeExplainer` run once per training pass on the validation set for every
regressor and the classifier, aggregated into `top_factors` per `(region_id, lead_time_days)` and
written to `shap_summary.parquet` — precomputed at train time, served on read. Fallback to
`feature_importances_` is surfaced via an explicit `method` field so the UI never implies SHAP
ran when it didn't.

**Retrain strategy — full retrain on every upload**, not `xgb_model=` warm-start: a re-upload can
introduce a new region/variable or shift value ranges, which warm-start would silently degrade
into rather than fail loudly on; thresholds must be recomputed from the full accumulated error
distribution regardless (explicit spec requirement); and at hackathon data scale, full retrain
is on the order of seconds. `POST /api/upload` ingests synchronously (parse → map → canonicalize
→ append parquet → record lineage) and returns immediately; retrain runs as a FastAPI
`BackgroundTask` against the full accumulated parquet dataset, writes a new `run_id`, and flips
`current.json` only on success, broadcasting `training_started` → `training_complete` (or
`training_failed`) over WS.

---

## API Contract

- **`POST /api/upload`** (multipart) → if columns need confirmation: `{batch_id, status:
  "pending_confirmation", detected_format, row_count_raw, mapping_proposals[], source_profile_match}`.
  If fully auto-accepted: `{batch_id, status: "training_started", row_count_ingested,
  canonical_variables_found[]}`.
- **`POST /api/upload/{batch_id}/confirm-mapping`** → `{mappings: [{source_column, variable,
  value_type}]}` → `{batch_id, status: "training_started", row_count_ingested}`.
- **`GET /api/regions?lead_time_days=N`** → `{lead_time_days, model_trained, last_trained_at,
  regions: [{region_id, region_name, bust_probability, risk_band, confidence, data_available}]}`.
  `model_trained: false` → `regions: []`, frontend renders the empty state, never a placeholder.
- **`GET /api/regions/{region_id}`** → per-variable Day1-10 series (`predicted_value,
  observed_value, predicted_error, confidence, model_mae, model_rmse, model_r2`),
  `bust_probability_curve`, `top_factors` (with `method: "shap"|"feature_importance_fallback"`),
  `analog_cases` if similarity search is supported. Missing variables are omitted or marked
  `{available: false}`, never fabricated.
- **`GET /api/alerts?limit&risk_band`** → `{generated_at, risk_band_definitions, alerts: [{alert_id,
  region_id, region_name, lead_time_days, bust_probability, risk_band, dominant_variable,
  created_at, training_run_id}]}`.
- **`GET /api/model/status`** → `{model_trained, current_run_id, last_trained_at,
  training_in_progress, data_volume: {...}, validation_metrics: {regressors: {...}, classifier:
  {...}}, thresholds: {...}}`.
- **WebSocket `/ws`** — envelope `{event, timestamp, payload}`; events: `upload_received`,
  `training_started`, `training_complete` (`{run_id, validation_metrics, regions_updated}`),
  `training_failed` (`{run_id, error}`), `new_alert`. Frontend reacts by invalidating TanStack
  Query keys (`['regions']`,`['regions', id]`, `['alerts']`, `['modelStatus']`) rather than
  merging WS payloads directly, keeping WS and REST shapes decoupled.

---

## Phased Build Order

1. **Stack + schema (this document) + scaffolding**: create folder skeletons,
   `ingestion/canonical_schema.py` (enums/dataclasses only, no logic), pinned
   `requirements.txt`/`package.json`, `.env.example` files, vendor the India-states
   GeoJSON/TopoJSON and confirm its region-name property key. **Checkpoint: request a real sampleea
   dataset from the user (even a few hundred rows) — hard blocker for Phase 2, never invented.**
2. **Ingestion + schema mapping** (gated on the sample): `parsers.py`, `schema_mapper.py`,
   `pipeline.py`, `db/models.py` (`UploadBatch`, `ColumnMapping`, `SourceProfile`), `utils/geo.py`
   + state-code lookup, `storage/parquet_store.py`. `tests/test_schema_mapper.py` runs against
   the real sample and asserts sane mapping proposals.
3. **ML pipeline**: `features/engineering.py`, `features/pivot.py`, `ml/regressors.py`,
   `ml/classifier.py`, `ml/thresholds.py`, `ml/explain.py`, `ml/registry.py`,
   `ml/train_pipeline.py`. Run against the ingested sample; report real MAE/RMSE/R² and
   classifier accuracy/precision/recall/F1/ROC-AUC — expect noisy metrics on a small sample and
   say so explicitly. **Checkpoint before building the API.**
4. **FastAPI layer**, then **React dashboard**: routers/services/broadcaster wired to the real
   trained model (smoke-tested with no mock fixtures); then Vite+TS scaffold, map + detail panel
   + alerts/summary + upload/confirm flow + `useLiveSocket`, all against the real running
   backend.
5. **Design pass** (palette/typography/layout) — explicitly deferred; palette/type plan shown
   before final CSS is written.

## Verification

- Phase 2: `pytest backend/tests/test_schema_mapper.py` against the user's real sample; manually 
  inspect a handful of mapped rows for unit/column correctness.
- Phase 3: run `train_pipeline.py` end-to-end on the ingested sample and print the validation
  metrics block directly to the user for review (no UI needed yet).
- Phase 4a: exercise every endpoint via FastAPI's `TestClient` or `curl` against the real
  ingested/trained data; confirm `model_trained: false` / empty-state responses correctly precede
  any upload in a fresh environment.
- Phase 4b: run `npm run dev` + the FastAPI server together, upload the real sample through the
  UI, confirm mappings, and verify the map/detail panel/alerts update live over the WebSocket
  with no page reload.
 