# M10 — Historical Analog + Event + Impact Intelligence

## Scope
M10 adds forecast-time event classification, potential-impact context, recommended operational actions, and an opportunistic historical analog search.

## No-fabrication rule
Analog cases are returned only when persisted historical evaluation events exist and comparable numeric features are available. Missing history returns an empty list; no synthetic or guessed analog is generated.

## Event rules
The first version uses transparent threshold heuristics for heavy/extreme rain, heat-like conditions, high wind, high humidity, and low-pressure context. These are **context signals**, not official warnings and not disaster predictions.

## Impact/action layer
Impacts and actions are mapped from event types and phrased as decision-support prompts. They do not assert that an impact will occur. Observation verification and local official warnings remain authoritative.

## Analog distance
Historical rows are compared using standardized Euclidean distance over available forecast-side numeric features. At least half of the requested features (and at least two) must be finite. The current cycle is excluded.

## Failure handling
Analog search is enhancement-only: unreadable/missing history must never break the region API.

## Tests
M10 tests cover deterministic event classification, non-finite inputs, de-duplication, analog ranking, and current-cycle exclusion.
