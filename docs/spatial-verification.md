# Sanket-X Spatial & Verification Contract

## M2 baseline

Sanket-X converts native 0.25-degree forecast/observation grids into the 36 IMD meteorological subdivisions before computing verification metrics.

### Spatial rule

1. Validate latitude/longitude ranges.
2. Use the supplied IMD subdivision geometry in WGS84 lon/lat coordinates.
3. Assign a grid cell by its centre point. A point on a polygon boundary is included with `covers`.
4. Reject a grid point that matches multiple subdivisions.
5. Exclude cells outside all subdivision polygons rather than assigning them to the nearest region.
6. Aggregate regular-grid values with a latitude cosine weight. This is the M2 baseline for equal-area weighting on a regular geographic grid.
7. Preserve `grid_cell_count` so every regional result is auditable.

This is intentionally not a nearest-city approximation.

## Temporal rule

The project convention is:

`lead_day = 1 -> valid_time = init_time`

`lead_day = 2 -> valid_time = init_time + 1 day`

and so on through Day 10.

Temporal alignment is validated using exact elapsed hours, not string/date comparison. A one-day shift is a hard error.

## Verification rule

A forecast is paired to an observation only when all of these match:

- subdivision ID
- variable
- valid date/time at the agreed daily grain
- forecast cycle / initialization metadata where applicable

No nearest-date matching is allowed for the M1/M2 label pipeline.

## Missing-data rule

Missing forecast or observation values are excluded from the weighted numerator and denominator. If a group has no valid cells, no regional value is emitted. A future quality gate will additionally enforce minimum valid-cell coverage before a label can be produced.

## Future upgrade

Exact polygon-cell intersection weighting can replace centre-point weighting after the M2 baseline is validated. The public API contract remains unchanged because the output still contains `subdivision_id`, value, and `grid_cell_count`.
