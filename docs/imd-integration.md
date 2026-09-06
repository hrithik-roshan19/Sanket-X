# Sanket-X — IMD Integration Gate

## Sources

- IMD 0.25° daily gridded rainfall is the primary India rainfall verification product.
- IMD meteorological subdivision geometry is fetched from the official subdivision boundary endpoint.
- The project does **not** ship invented or approximate subdivision polygons.

## Pipeline

```text
IMD NetCDF
  -> schema detection
  -> coordinate/range validation
  -> canonical daily grid
  -> 0.25° GEFS grid alignment
  -> subdivision polygon masking
  -> area-weighted aggregation
  -> forecast/observation join
  -> error
  -> bust label
```

## Fail-closed rules

1. Missing NetCDF variable/coordinate => stop.
2. Negative rainfall => stop.
3. Invalid latitude/longitude => stop.
4. Missing subdivision geometry => subdivision aggregation is blocked.
5. Fewer than 36 geometry features => blocked.
6. No synthetic fallback is permitted.

## Provenance

The downloaded geometry must be kept as a versioned data artifact and its source URL recorded in
pipeline metadata. The official IMD geometry endpoint currently referenced by the project is:
`https://mausam.imd.gov.in/imd_latest/contents/district_shapefiles/sd_boundary.json`

## Important scientific rule

Do not assume the historical GEFS archive has 31 members. Historical reforecast training uses the
members actually present in the archive; operational data can contain the current 31-member GEFS, but the serving model must use an ensemble size matching its training manifest; the current shipped model contract is five members.
