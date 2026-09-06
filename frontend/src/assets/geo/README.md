# Vendored India boundaries

`india_states.topojson` is vendored from
[udit-001/india-maps-data](https://github.com/udit-001/india-maps-data)
(`topojson/india.json`, fetched 2026-08-27), a maintained repository that curates
publicly available Survey of India / Census 2011 administrative boundary data
specifically for web choropleth use (React/D3 friendly, pre-simplified, ~870KB).

Contains a TopoJSON `Topology` with two objects:
- `states` — 36 features (28 states + 8 UTs, current post-2019 J&K/Ladakh split and
  post-2020 Dadra & Nagar Haveli + Daman & Diu merger), properties `st_nm` (name),
  `st_code` (2011 census state code). This is what `IndiaChoroplethMap` renders.
- `districts` — 726 features, same property scheme plus `district`/`dt_code`. Not
  used by the current dashboard (state-level is the map's grain) but kept in the file
  in case district-level drill-down becomes useful later — no separate fetch needed.

The backend's canonical `region_id` (ISO 3166-2:IN, e.g. `IN-MH`) is *not* the same
scheme as `st_code`. The translation between them lives in exactly one place:
`backend/app/utils/india_state_codes.py` (`StateRecord.st_code` / `st_nm` alongside
`region_id`). Any code joining API data onto map geometry should go through that table
(or its frontend mirror, once built) rather than assuming the two codes align.

No explicit license is published for the geometry itself; per the source repo's
disclaimer, it curates publicly available boundary data for reuse. Boundaries reflect
Survey of India data as curated by that project and may not represent disputed
territories with full precision — acceptable for this hackathon prototype's purposes,
not a reference source.
