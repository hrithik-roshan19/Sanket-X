# M8 — Probability Calibration & Explainability

## Goal
Turn raw classifier scores into safer bust probabilities and provide explainable feature contributions without allowing validation/test information into model fitting.

## Calibration contract
- Base classifier is fitted on the training cycles.
- Calibration is fitted only from held-out validation predictions.
- If validation data are too small or single-class, calibration is explicitly `identity` rather than guessed.
- >=50 calibration rows: isotonic regression.
- 20–49 calibration rows: Platt/logistic scaling.
- Probabilities are clipped to [0, 1] and non-finite values are sanitized.
- Test metrics use calibrated probabilities.
- The calibration method is persisted in the model manifest.

## Explainability contract
- SHAP TreeExplainer is preferred for tree models.
- If SHAP is unavailable/incompatible, feature importance is used as an explicitly labelled fallback.
- Feature names and ordering come from the trained artifact.
- Categorical vocabularies are persisted and reused at inference time; unseen categories map to missing category codes rather than silently changing category encodings.
- SHAP output shape must equal the feature matrix shape; otherwise fallback is used.

## Bugs fixed in M8
1. Validation/test category codes could differ from training category codes.
2. Inference did not load the training categorical vocabulary.
3. Classifier probabilities were raw even after a calibration stage was requested.
4. Missing/invalid probabilities could propagate to risk-band logic.
5. Calibration could fail on small or single-class validation data; it now fails closed to identity.

## Verification
- Calibration unit tests cover identity, Platt, isotonic, bounds, monotonicity, and non-finite inputs.
- Python compilation is required before packaging.
- Full repository tests must be run in the supported Python 3.11/3.12 environment with `pyarrow` installed. The current sandbox uses Python 3.13 and lacks `pyarrow`, so a full-suite PASS is not claimed here.
