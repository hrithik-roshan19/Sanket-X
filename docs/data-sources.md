# Sanket-X — Data Source Contract

## Core sources

| Source | Role | Status | Notes |
|---|---|---|---|
| NOAA GEFSv12 reforecast | historical forecast | REQUIRED | 5 members routinely; weekly 11-member extension |
| NOAA operational GEFSv12 | live forecast | REQUIRED | 31 members |
| ECMWF ERA5 | verification/background | REQUIRED | hourly; aggregate to verification windows |
| IMD 0.25° rainfall | India rainfall verification | REQUIRED | daily 0.25° grid, 1901–2024 archive |
| IMD subdivision geometry | regional aggregation | REQUIRED | 36 meteorological subdivisions |
| GPM IMERG | independent rainfall cross-check | OPTIONAL | use only when available |
| ECMWF IFS/AIFS | multi-model comparison | OPTIONAL | must never be an MVP hard dependency |

## Source-of-truth URLs

- IMD 0.25° rainfall: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- IMD real-time/archived gridded products: https://imdpune.gov.in/lrfindex.php
- IMD Data Service Portal: https://dsp.imdpune.gov.in/
- IMD subdivision boundary endpoint: https://mausam.imd.gov.in/imd_latest/contents/district_shapefiles/sd_boundary.json

## Non-negotiable rule

No model training starts until every REQUIRED source used by the training slice passes schema, coordinate, temporal, unit, missingness, duplicate, and provenance validation.
