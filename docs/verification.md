# Sanket-X Verification & Bust Labels

## Contract
A forecast row is verified against the observation at the exact `(region_id, valid_date, variable)` grain. The canonical lead contract is:

`valid_date = normalize(init_date) + (lead_time_days - 1) days`

Lead days are restricted to 1–10 for the SIH product.

## Fail-closed rules
- Missing required columns -> invalid run.
- Invalid or out-of-range lead time -> invalid row/run.
- Temporal misalignment -> row rejected and validation reported.
- Provisional observations are excluded from training labels.
- Conflicting observation values at the same canonical grain are rejected; they are never silently averaged.
- Exact duplicate observations are collapsed only after conflict checking.
- Forecast/observation pairing uses `many_to_one` validation.

## Error
Member error:

`abs_error = abs(forecast_value - observed_value)`

Event error is computed from the ensemble mean forecast against the observation.

## Bust labels
Thresholds are fitted **only on the training split**. Default threshold is the 90th percentile of event-level error per variable. Validation/test data receive the frozen thresholds; they never influence threshold fitting.

`bust_ratio = actual_error / train_threshold`

`y_bust = bust_ratio >= 1`

## Why this is separate from ML
The verification layer must remain deterministic and testable. ML can predict future bust probability, but it must never define the historical ground truth used to evaluate itself.
