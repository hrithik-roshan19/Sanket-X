# API surface

Run it:

```bash
# macOS / Linux
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
# interactive docs: http://localhost:8000/docs
```

```powershell
# Windows (PowerShell)
cd backend; .\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness + whether a model exists |
| `POST` | `/api/upload` | multipart file upload → ingest (+ background retrain) |
| `POST` | `/api/upload/{batch_id}/confirm-mapping` | resolve ambiguous columns, then ingest |
| `GET` | `/api/regions?lead_time_days=N` | choropleth payload: one row per region for lead day N |
| `GET` | `/api/regions/{region_id}` | detail panel: per-variable series, bust curve, top factors |
| `GET` | `/api/alerts?limit&risk_band` | medium/high-risk (region, lead) events, riskiest first |
| `GET` | `/api/model/status` | what is trained, on how much data, and how well it scored |
| `GET` | `/api/model/runs` | every persisted run id + the current one |
| `GET` | `/api/replay/cycles` | every scoreable forecast cycle, most demo-worthy (a real verified bust that grows into the medium range) first |
| `GET` | `/api/replay?init_date=YYYY-MM-DD` | guided replay of one cycle: per lead-day region scores + a narration string generated from those numbers, plus the one variable that diverged most. Omit `init_date` for the top-ranked cycle |
| `WS` | `/ws` | typed events: `connected`, `upload_received`, `mapping_pending`, `training_started`, `training_complete`, `training_failed`, `new_alert` |

## The honesty contract

Every response distinguishes "no model / no data" from "a real number":

- No trained model → `model_trained: false`, **empty** `regions` / `alerts` / `variables`,
  and a `message` explaining what to do. Never a placeholder or example value.
- A variable the model could not train (too few paired rows) → `available: false` with an
  empty `points` list, and it is listed under `skipped_variables` in `/api/model/status`.
- `top_factors` carries `top_factors_method` (`shap` or `feature_importance_fallback`), so
  the UI never implies SHAP ran when it did not.
- `analog_cases` is `[]` because similarity search is not implemented — it is empty rather
  than invented.
- `/api/replay` narration strings are assembled from the cycle's own scored numbers
  (which region climbed, by how much, its dominant variable, how many crossed into
  `high`). Nothing is scripted; an unverified cycle says so rather than showing an
  observed line.
- `risk_band_definitions.basis` states that the bands are percentiles of the current run's
  validation-split probabilities, not fixed cutoffs.
- `/api/model/status.validation_metrics` reports which `split` each number came from,
  preferring the held-out `test` split.

## Upload → training lifecycle

```
POST /api/upload
  ├─ columns all auto-accepted (or an exact SourceProfile match)
  │    → 200 {status: "training_started"}  + BackgroundTask retrain
  └─ anything ambiguous
       → 200 {status: "pending_confirmation", mapping_proposals: [...]}
         POST /api/upload/{batch_id}/confirm-mapping
           → 200 {status: "training_started"} + BackgroundTask retrain

WebSocket during a retrain:
  upload_received → [mapping_pending] → training_started → training_complete | training_failed
```

Retraining is **full**, never warm-start, and runs against the whole accumulated canonical
store. `data/models/current.json` is rewritten atomically and only after a run finishes
completely, so a failed retrain leaves the previously-good model serving.

A second upload while a retrain is running does not queue a duplicate — `run_retrain`
holds a non-blocking lock and skips, and `/api/model/status.training_in_progress` reports it.

## Frontend contract note

WS payloads are intentionally small and carry no measurements. The frontend reacts to the
**event name** by invalidating the matching TanStack Query keys (`['regions']`,
`['regions', id]`, `['alerts']`, `['modelStatus']`) and refetching over REST — that keeps
the socket and REST shapes decoupled.

## Performance

`/api/regions` and `/api/regions/{id}` are served from an in-process cache of the scored
forecast cycle (~0.2 s warm; ~1.7 s on the first request after a retrain). The cache key is
`(run_id, init_date, canonical row count)`, so it invalidates by itself when a retrain
lands or new data is ingested. The cache holds the last ~24 scored cycles, so guided
replay flipping between historical cycles stays warm.

`/api/replay/cycles` scores every historical cycle once to rank them (~1–2 min the first
time), so the app warms it in a background thread at startup — the endpoint is instant by
the time anyone opens the replay tab. `/api/replay` for a single cycle is ~2 s cold.

## Live ingestion (Phase 6)

| Endpoint | Purpose |
|---|---|
| `GET /api/ingest/status` | feed health: last cycle ingested, last observation refresh, scheduler state |
| `POST /api/ingest/run-cycle` | pull the newest published GEFS cycle now; `?wait=true` runs inline |
| `POST /api/ingest/refresh-observations?tier=final\|provisional` | pull one observation tier now |
| `POST /api/ingest/backfill?cycles=N` | pull N past cycles, oldest first, resumable |

`/api/regions` gained **`available_lead_days`**: a 06/12/18Z cycle cannot produce a
whole-calendar-day forecast for its own init day, so it has no day 1. The dashboard reads
this rather than assuming day 1 exists.

`/api/regions/{id}` variable points gained **`observed_status`** — `"final"` (ERA5, the
training baseline), `"provisional"` (near-real-time, subject to revision) or `null` when
that lead has not verified yet. Provisional values are shown badged and are never trained on.

See [`app/live/README.md`](../live/README.md) for the feed itself and the consistency rules
that keep live data on the same footing as the training data.
