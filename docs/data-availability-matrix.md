# Sanket-X — Data Availability Matrix

Status is intentionally split into **source available** and **repo integrated**. A public source does not mean the current repository already contains the required data.

| Dataset / input | Needed | Publicly available | In current repo | Sanket-X decision |
|---|---:|---:|---:|---|
| NOAA GEFSv12 historical reforecast | Yes | Yes | Sample present | Core training source |
| NOAA GEFSv12 operational | Yes | Yes | Live fetcher exists | Core live source |
| ERA5 hourly/reanalysis | Yes | Yes | 2019 sample present | Core verification source |
| IMD 0.25° daily rainfall | Yes | Yes | Not integrated | **Add as primary India rainfall verification** |
| IMD 36 meteorological subdivisions | Yes | Yes (boundaries/monitoring concept) | Not integrated; repo currently has state/UT GeoJSON | **Add subdivision boundary layer** |
| GPM IMERG | Optional | Yes | Not integrated | Independent rainfall cross-check |
| ECMWF IFS/AIFS open subset | Optional | Yes, subject to terms | Not integrated | Multi-model agreement enhancement |
| Historical bust labels | Derived | Yes, from forecast + verification | Partial | Generate only after M1 validation |
| Historical analog cases | Derived | Yes, from validated archive | Not implemented | Phase 2 enhancement |

## Critical facts

1. Historical GEFSv12 reforecast is **not** a 31-member historical archive. It provides 5 members routinely and an 11-member weekly extension; operational GEFS has the larger ensemble.
2. Therefore the training schema must support variable ensemble size, while the live schema can accept the operational 31-member ensemble.
3. IMD 0.25° rainfall is a better India rainfall verification source than relying on ERA5 precipitation alone.
4. The repository's existing `india_states.geojson` contains administrative state/UT geometry, **not the required 36 IMD meteorological subdivision polygons**.
5. No model training should begin until forecast/observation temporal and spatial alignment passes the validation gate.
