# Sanket-X — M1 Data Validation Gate

## Objective

Prove that the forecast and verification datasets can be joined without temporal, spatial, unit, duplicate, or provenance errors before any ML model is trained.

## Current repository sample

- `backend/data/samples/gefs_reforecast_india_2019.parquet`
- `backend/data/samples/era5_observations_india_2019.parquet`

The samples are real-data artifacts. Their Parquet metadata records PyArrow 16.1.0 and the expected forecast/observation columns.

## Validation rules

### GEFS

- Required init/valid dates.
- `valid_date = init_date + (lead_day - 1)` for the repository's daily lead convention.
- `lead_day` must be 1–10.
- Historical members must be a subset of `c00,p01,p02,p03,p04` and the sample must contain all five for a complete 5-member slice.
- Latitude/longitude must be valid.
- Forecast variables must pass plausible physical ranges.
- No duplicate `(city, init_date, member, valid_date)` keys.
- Source/provenance fields must be present.

### ERA5

- Valid date and coordinates.
- Plausible physical ranges.
- No duplicate `(city, date)` keys.
- Daily fields use the documented canonical units.

### Pairability

- Forecast valid dates are joined to observation dates, never initialization dates.
- Pair rate is measured before labels are generated.
- A low pair rate is a hard failure.

## Execution

From `backend/`:

```text
python scripts/validate_samples.py
```

The command writes `data/analysis/data_validation_report.json` and exits non-zero on a hard validation failure.

## What is deliberately NOT validated by the sample script yet

- IMD 0.25° grid-to-grid rainfall alignment, because that dataset is not currently in the repository.
- 36 IMD subdivision polygon coverage, because only state/UT GeoJSON is currently present.
- Operational 31-member GEFS consistency, because the bundled sample is historical 5-member data.
- Cross-source ERA5/IMD/GPM disagreement.

These are explicit M1 follow-up gates, not assumptions.
