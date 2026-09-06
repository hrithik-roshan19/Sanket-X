
## Update — IMD integration foundation

- Added a strict registry of the 36 IMD meteorological subdivisions.
- Added an official-geometry downloader; no approximate polygons are generated as fallback.
- Added fail-closed geometry validation (FeatureCollection, geometry presence, minimum 36 features).
- Added an IMD gridded-rainfall NetCDF reader with explicit variable/coordinate discovery and physical-range checks.
- Added regression tests for the 36-subdivision registry and rainfall reader failure behavior.
- Added the IMD integration gate documentation.

### Validation run in the build environment

- IMD-specific tests: **3 passed**.
- Combined M1 + IMD tests: **7 passed, 1 skipped**.
- Python compile check: **passed**.
- Full backend suite remains environment-blocked until the project's declared `pyarrow` dependency is installed; this is not being treated as a test pass.

### Not yet marked complete

The official IMD subdivision geometry and 0.25° rainfall file have not been downloaded into this artifact because the build environment has no outbound network access. The fetcher is ready for a network-enabled developer machine/CI run. No synthetic geometry or rainfall values were inserted.
