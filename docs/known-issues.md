# Known issues

Things that are true, unfixed, and deliberate to leave. Written down so they are found
here rather than discovered live.

## Behaviour a visitor could notice

- **First load after a deploy is slow.** Caches are per-process and start empty; CI warms
  the three expensive endpoints after each publish, but a visitor arriving during that
  window pays for a cold `/api/replay/cycles` (measured 50–120 s cold against ~1 s warm on
  the deployed instance). It is slow, never wrong.
- **The opening screen is one viewport on desktop only.** On phones the KPI strip, the
  cue row and the ticker sit below the fold. The page scrolls and what is visible is
  composed; only the single-screen effect is lost.
- **Replay offers the 10 most recent cycles**, not every cycle in the store
  (`replay_service._MAX_CYCLES`). Each candidate costs one scoring pass on first call.
- **Feature and variable names render raw, in snake_case.** The SHAP panel lists
  `spread_rainfall_mm`, `conf_pressure_hpa` and
  `historical_bust_frequency_region_season`; the variable tabs and the bust-threshold
  list show `atmospheric_moisture_kgm2` and its peers. Everything around them was put
  into plain language on 2026-09-02, so these are now the densest text on the page for a
  reader without a meteorology background. Fixing it needs a display-name map rather than
  a text edit — the names arrive from the model's own feature list, not from a string in
  the component — which is why it was left rather than rushed. Worth doing Friday morning
  if there is time before the freeze; it is cosmetic and nothing depends on it.

## Operational

- **Scheduled refreshes run 3–5 hours late.** GitHub's cron is best-effort and queues on
  shared capacity for public repos; all four daily slots fire, consistently late. It costs
  nothing because the refresh is catch-up driven — each run ingests whichever of the last
  four cycles the store lacks, oldest first — and the dashboard reports which cycle is
  actually loaded rather than implying "now". Moving the schedule off the hour (`:23`)
  already reduced queueing; the remaining delay is not controllable from here.
- **Serving memory runs close to the 512 MB ceiling.** Measured 442 MB after compaction,
  against 490 MB before. The instance is killed rather than throttled if it is exceeded,
  so anything that increases what is held at serve time needs measuring, not estimating.
  `/api/health` reports the live figure because the platform paywalls its own metrics.
- **The CI serving-memory harness is noisy.** The same store measured 566 MB and 510 MB
  peak on consecutive runs, so it cannot resolve differences below roughly ±55 MB. Use it
  for large effects only; for small ones read `/api/health` on the running instance, where
  readings are stable.

## Data

- **`docs/results.md` is generated, not committed.** Baselines are written into the model
  run that produced them and served from there, so the numbers cannot describe a different
  model than the one answering requests. The repo previously held a results file naming a
  superseded run.
- **Provisional observations are excluded from training** and badged in the UI. Verifying
  against a different product than the model was trained on would shift both the error and
  the bust label derived from it.
- **A cycle too incomplete to publish is refused, not partially ingested.** A short
  rainfall *sum* is roughly half the real accumulation, and rainfall drives most busts, so
  publishing a thin cycle would be worse than publishing nothing.
