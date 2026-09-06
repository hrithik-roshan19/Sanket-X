# M12 — Final SIH Validation & Release Gate

**Project:** Sanket-X  
**SIH:** 2026 Problem Statement 26079 — AI-based forecast bust detection  
**Milestone:** M12 / final engineering gate

## Release decision

The codebase is **engineering-ready for SIH demonstration**, subject to the explicit data and
runtime limitations below. No fabricated forecast, observation, probability, metric, geometry,
or warning value is introduced as a fallback.

## Verified invariants

`backend/scripts/final_validation.py` checks the release contract without network access:

- required backend/frontend/release files exist
- backend Python source parses
- request IDs and security headers are enabled
- API responses are marked `no-store`
- admin-key guard exists for mutation paths
- production upload is disabled by default
- production live ingestion is disabled by default
- production rate limiting is enabled
- the operational 31-member GEFS catalog is declared
- live ensemble size cannot silently differ from the model's training ensemble size
- no obvious fabricated numeric fallback assignment is present
- frontend has a production build script and entrypoint

Latest result: **22/22 invariants PASS**.

Run:

```bash
cd backend/..   # project root
python backend/scripts/final_validation.py
```

## Automated tests

Targeted regression suite for the scientific milestones:

```text
28 passed, 1 skipped
```

Covered areas include:

- sample/data validation
- exact temporal verification
- spatial aggregation
- calibration
- regional intelligence
- event/impact/action intelligence

The full pytest collection cannot be completed in the current offline sandbox because `pyarrow`
is not installed. Installation attempts cannot reach the package index. This is an environment
limitation, not reported as a passing full-suite result.

## Build verification

Backend source compilation passes with `compileall`.

The frontend production build is **not claimed as executed** in this sandbox because
`frontend/node_modules` is absent and dependency installation requires network access. On a
network-enabled machine/CI runner, execute:

```bash
cd frontend
npm ci
npm run build
```

## Scientific contract

### Historical training

The current historical GEFSv12 reforecast training contract uses the members actually available
in the archive (five routine members; the archive is not a historical 31-member dataset).

### Operational data

Operational GEFSv12 supports 31 members. Sanket-X records that catalog, but does **not** silently
feed a different ensemble size into a model trained on five members. The live orchestrator checks
the model manifest and blocks incompatible cycles. A future 31-member serving model requires
31-member-compatible training data and retraining.

### Observations

Provisional observations are never used for training. Final observations supersede them for
verification/retraining according to the documented live observation policy.

### Bust definition

Bust means large surface-variable forecast error according to thresholds fitted from training
data. It is not presented as an official IMD warning or the synoptic Z500 bust criterion.

### Event intelligence

Heavy rain, extreme rain, heat, wind, humidity, and pressure signals are contextual heuristics.
Their associated impacts/actions are decision-support context, not official disaster warnings.

## Known evidence gaps

These remain visible rather than hidden:

1. the shipped regional map/data path is still based on the available city/state artifacts rather
   than a fully integrated 36-IMD-subdivision polygon dataset;
2. historical training does not contain a full 31-member GEFS reforecast archive;
3. ERA5 rainfall has known limitations over India relative to gauge-based products;
4. IMDAA/IMD gauge-based verification and the synoptic Z500 criterion remain future extensions;
5. the serving container is memory-constrained and training remains a CI operation.

None of these gaps are patched with synthetic values.

## SIH demo flow

1. Open the dashboard and verify the loaded model/cycle status.
2. Select a lead day from the actually available forecast cycle.
3. Inspect regional bust probability and reliability/confidence.
4. Open a region and inspect variable trajectories, forecast error, SHAP factors, and event
   intelligence.
5. Open replay and select a historical cycle when persisted evaluation events are available.
6. Show the About/Model status view for model metrics and baseline comparison.
7. If a live cycle is unavailable, show the explicit feed/model status rather than presenting a
   fake "live" number.

## Final acceptance rule

A release is acceptable only when:

`data provenance → temporal verification → spatial coverage → leakage-safe features →
held-out evaluation → calibration → explanation → API → dashboard → operational safety`

is traceable. Any unavailable component must surface as unavailable; it must not be replaced by
mock data.
