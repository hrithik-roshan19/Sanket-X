# Sanket-X M5 — Baseline Evaluation Contract

## Purpose

Before accepting the Sanket-X bust classifier, we must establish what can be predicted without the classifier. The baseline ladder is fitted **only on the training cycles** and evaluated on the identical held-out test rows.

## Baselines

1. **Climatology** — training bust rate only.
2. **Lead-day** — logistic model using forecast lead time.
3. **Ensemble spread** — logistic model using available ensemble spread statistics.
4. **Lead + spread + season** — logistic model using lead time, spread and season.
5. **Sanket bust classifier** — the learned model, when a trained evaluation artifact is available.

## Primary metrics

- Brier score: lower is better.
- Brier Skill Score (BSS) against training climatology: higher is better; 0 means no improvement over climatology.
- ROC-AUC and PR-AUC when both classes exist in the evaluation set.
- F1 is reported at probability threshold 0.5 but is not the sole acceptance criterion.
- BSS by lead day is reported when enough held-out events exist.

## Anti-leakage rules

- Baselines never fit on test rows.
- Climatology is the **training** bust rate, never the test bust rate.
- Thresholds used to construct `y_bust` must already be frozen from training data before test evaluation.
- All baseline/model comparisons must use exactly the same held-out event rows.

## Acceptance gate

The classifier is not declared successful merely because it has a high ROC-AUC. It must be compared against every baseline on the same held-out rows. If it does not beat a simpler baseline consistently, we keep the simpler model and document the result.

## M5 status

The baseline implementation is unit-tested. A real-data scorecard is intentionally blocked until the environment has the project's pinned Python/runtime dependencies, especially `pyarrow`, and a real M1 evaluation-event artifact exists. Synthetic tests are used only to verify implementation behavior, never as SIH performance claims.
