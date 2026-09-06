# Sanket-X M6 — Feature Engineering and Leakage Contract

## Goal
Build features that are available at forecast issuance time and prevent observation/target leakage.

## Allowed forecast-time features
- forecast value
- ensemble mean/spread/member count
- forecast-only lag and lead-to-lead delta within the same cycle/member
- forecast pressure/moisture rate of change
- concurrent forecast variables
- lead day, month, season, region

## Explicitly forbidden
- observed value
- absolute forecast error
- previous verified error (`forecast_error_lag`)
- bust label / bust ratio
- target-derived historical frequency unless encoded strictly out-of-fold / prior-to-cycle

## Why
A live Day-N forecast cannot know the verification observation or its error. Any such feature would make offline metrics optimistic and fail at serving time.

## Training split
Splits remain chronological by forecast initialization cycle. A future implementation must use an embargo/purge period at least as long as the maximum verification lead when the dataset has enough cycles. Small datasets must be reported as insufficient rather than silently claiming a leakage-safe three-way evaluation.

## M6 gate
No XGBoost result is accepted as a production claim until: feature allowlist tests, target-leakage tests, missing-value tests, time split tests, and baseline comparison all pass.
