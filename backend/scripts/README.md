# Sample-data fetchers

Two scripts that build the **real** forecast/observation sample the rest of Sanket
AI is developed and tested against. They are the concrete answer to the plan's Phase 1 →
Phase 2 gate ("request a real sample dataset ... never invented"). Nothing they produce
is synthetic: every value traces to a named public archive record.

| Script | Produces | Source | Grain |
|---|---|---|---|
| `fetch_gefs_reforecast_sample.py` | `data/samples/gefs_reforecast_india_2019.{csv,parquet}` | NOAA **GEFSv12 reforecast** (AWS open data, US Gov public domain) | one row per `(city, init_date, member, valid_date)` — the **forecast** side |
| `fetch_era5_observations.py` | `data/samples/era5_observations_india_2019.{csv,parquet}` | **ERA5** reanalysis via Open-Meteo Historical Weather API (CC-BY 4.0; Copernicus C3S) | one row per `(city, date)` — the **observed** side |

`india_cities.json` — 30 points, one per state/UT capital-ish city, lat/lon carried from
the user's own `all_india_weather_2025_2026.csv` so points stay consistent with prior work.

## Why this pairing

The problem statement is about medium-range (Day 1–10) **NWP forecast** busts. That needs
forecast values with real initialisation dates and lead times, verified against
observations. GEFSv12 reforecast supplies the forecasts (5 members → real ensemble
spread); ERA5 is the standard verification truth. They join on `(city, valid_date)` at
feature-engineering time — exactly the join described in the plan's canonical-schema
section (forecast and observed kept as separate rows, paired later, never pre-joined at
ingestion).

## Running

```bash
cd backend
source .venv/bin/activate            # needs: xarray, cfgrib, eccodes, requests, pandas, pyarrow

python scripts/fetch_era5_observations.py            # ~1 min, 30 small API calls
python scripts/fetch_gefs_reforecast_sample.py       # ~60–90 min, ~2 GB transient download
```

The GEFS script checkpoints one parquet part per init date under
`data/samples/_gefs_parts/`; re-run with `--resume` to continue after an interruption.
Useful flags: `--max-inits N`, `--members c00,p01`, `--list-only`.

### How the GEFS fetch stays small

For each `(variable, init, member)` file it downloads **only the GRIB2 messages it needs**
via HTTP `Range` requests keyed off the `.idx` sidecar — not the full 30–70 MB files.
Messages for one lead day are contiguous, so they collapse into one ranged GET per lead
day. Each message is then decoded with cfgrib and sampled at the 30 city points by
nearest grid cell.

## Canonical variables and honest lead coverage

| canonical column | GEFS source | ERA5 source | unit | notes |
|---|---|---|---|---|
| `t2m_c` | `tmp_2m` | `temperature_2m` | °C | K→°C |
| `rh2m_pct` | derived from `spfh_2m` + `tmp_2m` + `pres_sfc` | `relative_humidity_2m` | % | Bolton (1980) `es`; standard q→e inversion |
| `apcp_mm` / `precip_mm` | `apcp_sfc` | `precipitation` | mm | GEFS: 24 h total tiled from 3/6 h buckets. ERA5: daily sum |
| `mslp_hpa` | `pres_msl` | `pressure_msl` | hPa | Pa→hPa |
| `psfc_hpa` | `pres_sfc` | `surface_pressure` | hPa | lower at altitude (e.g. Leh ≈ 660 hPa) |
| `pwat_kgm2` | `pwat_eatm` | `total_column_integrated_water_vapour` | kg/m² | atmospheric moisture / TCWV |
| `wspd10m_ms`, `wdir10m_deg` | `ugrd_hgt` / `vgrd_hgt` (10 m) | `wind_speed_10m` / `wind_direction_10m` | m/s, ° | speed/dir derived from daily-mean u,v (vector mean; a scalar mean of degrees is wrong across 0/360) |
| `soilw_vol_pct` / `soil_moisture_pct` | `soilw_bgrnd` (0–0.1 m) | `soil_moisture_0_to_7cm` | % volumetric | GEFS ×100; ERA5 ×100 (m³/m³ → %) |

**Lead-time coverage differs by variable in the GEFSv12 reforecast archive** and is
reported at the end of the run — do not assume Day 1–10 everywhere:

- `t2m`, `rh2m`, `apcp`, `mslp`, `psfc`, `pwat` — Day 1–10
- 10 m wind (`ugrd_hgt` / `vgrd_hgt`) — **Day 1–5 only**
- soil moisture (`soilw_bgrnd`) — **~Day 1–3 only**

The ML layer already treats variables with sparse paired rows as skip-and-surface rather
than assuming full coverage, so this is a data fact to carry forward, not a bug to patch.

## Daily aggregation

GEFS reforecast is 3-hourly. "Day k" is defined as valid at `init + k days`; every
3-hourly sample in the hour window `((k-1)·24, k·24]` is aggregated to one value per city
(mean for state variables, sum for precipitation, vector mean for wind). ERA5 is fetched
hourly and aggregated the same way per UTC calendar day. The ERA5 window runs to
2020-01-15 so Day-10 forecasts issued in mid-December 2019 still have a verifying
observation.

## Provenance

Every GEFS row carries `src_<var>` (the exact S3 key) and `srcmsg_<var>` (the GRIB2
message numbers within it) so any value can be traced back and re-derived. ERA5 rows
carry a `source` string.

## Licensing

- **GEFSv12 reforecast** — work of the U.S. Government (NOAA), public domain; hosted on
  the AWS Open Data registry (`noaa-gefs-retrospective`).
- **ERA5** — Copernicus Climate Change Service (C3S) information; redistributed by
  Open-Meteo under CC-BY 4.0. Attribute both if the sample is published.
