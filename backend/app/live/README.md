# Live ingestion (Phase 6)

Automated ingestion of real operational forecasts and the observations that verify them.

```bash
# macOS / Linux  — off by default; switch on deliberately
echo "LIVE_INGEST_ENABLED=true" >> backend/.env
uvicorn app.main:app --port 8000        # scheduler starts with the app
```

```powershell
# Windows (PowerShell)
Add-Content backend\.env "LIVE_INGEST_ENABLED=true"
uvicorn app.main:app --port 8000
```

Nothing here runs on import, and nothing runs in tests: a fresh clone never reaches out to
NOAA just by starting the server.

## What runs, and when

| Job | Cadence | Retrains? |
|---|---|---|
| Forecast cycle (GEFS operational) | every published cycle, `00/06/12/18Z` | **no** |
| Observations, provisional | every 6 h | no |
| Observations, final (ERA5) | daily | **yes**, past a row threshold |

A forecast pull never retrains: a forecast that has not verified yet carries no label to
learn from. Only newly-final observations create new training pairs.

The scheduler ([`scheduler.py`](scheduler.py)) is a daemon thread that ticks every 15 min
and asks the orchestrator what is due. Ticks are cheap when nothing is: the idempotency
check returns immediately for a cycle already stored.

## Endpoints

| | |
|---|---|
| `GET /api/ingest/status` | feed health: last cycle, last observation refresh, scheduler state |
| `POST /api/ingest/run-cycle` | pull one cycle now ("Refresh now"); `?wait=true` to run inline |
| `POST /api/ingest/refresh-observations?tier=final\|provisional` | pull one observation tier now |
| `POST /api/ingest/backfill?cycles=N` | pull N past cycles, oldest first; resumable |

## Where the data comes from

**Forecasts** — NOAA GEFS operational, `pgrb2sp25` (0.25°), Day 1–10 in 3-hourly steps.
Two transports, same decode path:

- **NOMADS grib-filter** subsets to an India box server-side: **~0.19 MB per step**
  instead of ~4.4 MB of byte-ranged global messages, and one request rather than nine.
  Retains roughly the last **3 days**. A full cycle is ~75 MB in ~100 s.
- **AWS S3** `noaa-gefs-pds` for anything older, byte-ranging the wanted messages off each
  file's `.idx`. Correct but ~23× heavier — about **1.8 GB per cycle**. Backfills reaching
  past the NOMADS window are expensive; size them deliberately.

**Observations** — Open-Meteo, two tiers (below).

## The consistency rules

These are what keep live data on the same footing as the data the models learned from.
Each was measured, not assumed; changing any of them requires retraining.

**0.25° only.** Measured at the 30 city points, the 0.5° product differs by up to
**6.1 °C** in 2 m temperature and **10 %RH** — worst in exactly the mountain regions
(Sikkim, Himachal, Ladakh) the model flags most often.

**Exactly 5 members** (`gec00`, `gep01`–`gep04`) are currently used for serving because the
historical models were trained on five members. Operational GEFS carries 31. The configuration
keeps the full 31-member catalog available, while the live orchestrator blocks a cycle whenever
the configured ensemble size differs from the current model's recorded training size. A
31-member serving run therefore requires a retrain on 31-member-compatible historical data;
it is never enabled by configuration alone.

**RH is read directly**, where the training fetch derived it from specific humidity via
Bolton (1980). Measured on the same cycle at the same points, the two agree to within
**0.50 %RH** — far below the models' error scale.

**`valid_date = init_date + (lead − 1)`.** Lead day 1 is the first calendar day the run
covers. Aggregation is by the valid time's calendar day, not by hours since issue: for a
00Z cycle those are identical, but for 06/12/18Z a lead-hour window straddles midnight and
could never line up with the calendar-day observation verifying it. A 06/12/18Z cycle
therefore has **no day 1** — its own init day is only partly covered — and the API reports
`available_lead_days` so the dashboard lands on a day that exists.

**Precipitation uses only the 6-hourly buckets.** APCP accumulations reset every 6 h, so
the 3-hourly steps overlap (`f003` is 0–3 h, `f006` is 0–6 h). Summing every 3-hourly value
would roughly **double** daily rainfall.

**Only `mslp_hpa` becomes `pressure_hpa`.** Both files also carry `psfc_hpa`, which is
dominated by station elevation (Leh reads ~684 hPa) and is explicitly excluded.

## Two-tier verification

ERA5 lands ~5 days late, so a forecast issued today would show no verification for most of
a week. Observations therefore arrive twice:

- **`provisional`** — near-real-time analysis, available within hours. Fills the charts
  immediately, is badged in the UI, and is **excluded from training**.
- **`final`** — ERA5/ERA5T. Supersedes the provisional row for the same reading and is
  what the models learn from.

Training reads `read_dataset(exclude_provisional=True)`. That filter tests
`!= "provisional"` **plus** `is_null()`: rows ingested before Phase 6 carry a NULL status
and came from the ERA5 archive, so they are already final — and in Arrow, as in SQL,
comparing a null yields null rather than true, so a bare inequality would silently discard
every one of them.

Provisional data omits `atmospheric_moisture_kgm2` (TCWV is not served by the
near-real-time endpoint), and provisional **precipitation** is materially weaker than
ERA5's — it is shown, badged, and replaced.

## Revisions, and why deduplication exists

Each ingest writes its own parquet partition, so a final observation lands *alongside* the
provisional one rather than replacing it. `parquet_store.read_dataset()` collapses these on
read, preferring `final` over `provisional` and then the most recently ingested.

This matters more than it looks: a duplicated observation would **double every forecast row
it merges with** in feature engineering, silently reweighting training. The identity key is
`(region_id, valid_date, variable, value_type, init_date, lead_time_days,
ensemble_member_id, source_column)`.

`source_column` is part of it on purpose — a file can legitimately map two columns onto one
canonical variable, and those are two measurements, not a duplicate. A null `region_id`
falls back to the row's coordinates rather than a shared sentinel, so unresolved points are
not merged into each other.

## Idempotency and failure

- A cycle is keyed by **date *and* hour** in `ingest_run`: a date alone cannot separate 00Z
  from 06Z, and partitions are per-batch, so a repeat pull would add a duplicate rather
  than overwrite.
- `run_due_work` holds a lock, so a slow pull and the next tick cannot overlap.
- Every attempt writes an `IngestRun` row **including skips and failures**, so
  `/api/ingest/status` reports real feed health rather than implying the data is current.
- A missing step is recorded, not fatal — but a lead day reduced from fewer samples than a
  full day is reported as `undersampled` rather than quietly accepted.
- A failed pull never touches the live model or the last good cycle.

## Mapping is explicit, never inferred

Live files are generated by this codebase, so `orchestrator.py` passes their column mapping
explicitly instead of leaving it to the schema mapper. An unattended job must not depend on
fuzzy matching.

This is load-bearing. The observation file's free-text `source` column was detected as a
per-row **value_type discriminator**, and because it read "…Open-Meteo forecast-api…", an
entire file of observations was ingested as *forecasts*. The fix is threefold: the column is
excluded explicitly, `_apply_confirmations` now clears a discriminator that confirmation has
excluded, and the provenance string no longer contains the word "forecast".
