# Sanket — what was built, in order

A record of how this project got to where it is, reconstructed from the 60 commits in
`git log` rather than from memory. Dates are commit dates (IST). Current metrics are
**not** repeated here — they are served at `/api/model/status` and rendered on the About
tab, because a number copied into a document is right until the next retrain and quietly
wrong afterwards. This file records *what happened and why*, which does not go stale.

---

## Before the first commit · The plan

`docs/plan.md`, written against an empty directory, fixed the things that everything after
it inherited: the canonical internal schema, the schema-mapping algorithm, the two-head ML
architecture (per-variable error regression feeding a bust classifier), the API contract,
and a phased build order with explicit checkpoints.

Two decisions from it shaped every later choice:

- **No synthetic or mocked numbers anywhere in the shipped app, ever, even as a fallback.**
  Every number on screen must trace to a real upload and a real trained model, or the UI
  must say plainly that it cannot. This is why there are `model_trained: false` states,
  thresholds derived from data rather than hardcoded, and em dashes with reasons.
- **The real sample dataset was a hard blocker for Phase 2** — "never invented". The build
  was gated on having real data rather than proceeding on placeholders.

## 28–29 Aug · Foundations

`c8876c6` `e71f259` `c969ba8`

`c8876c6` is not an ordinary commit: **106 files, 84,403 insertions**, landing Phases 1–4
of the plan in one go — 45 backend modules, 24 frontend components, 7 test files, 5 data
scripts. There is no finer-grained record of that work, so it is described here by what it
contains rather than by commits:

- **Ingestion** — `parsers.py`, `schema_mapper.py`, `pipeline.py`, the `UploadBatch` /
  `ColumnMapping` / `SourceProfile` tables, geo resolution, and the hive-partitioned
  `parquet_store.py`. Format-agnostic on purpose: the real dataset's columns were unknown.
- **ML** — feature engineering and pivot, per-variable regressors, the bust classifier,
  thresholds, SHAP explanations, and the run registry with `train_pipeline.py`.
- **API + dashboard** — FastAPI routers/services/broadcaster against the real trained model
  with no mock fixtures, then the Vite/TS dashboard: India map, region detail, alerts, and
  the upload → confirm-mapping flow.

`e71f259` and `c969ba8` follow with corrections and Windows compatibility, because the
setup had to work on a machine other than the one that wrote it.

## 30 Aug · The product takes shape

`0919035` `16a403d` `4e50aec` `737a2ac` `1226689`

Guided replay arrives — scoring a real historical cycle with the deployed model — along
with an all-lead-days regions endpoint and a rebuild onto 36 city points. The upload →
confirm-mapping → retrain loop is hardened, and CI starts running the README's own setup
steps on clean Windows and Linux, so the documented path is tested rather than assumed.

`737a2ac` is the design pass: the project becomes **Sanket**, gains the Divergence visual
system and a national hero view, and adopts the low/watch/bust vocabulary.

## 31 Aug · Making it survive a free tier

`0f0882d` … `72182d7`

The layout settles into a full-viewport opening screen and four tabs. Then deployment,
and the constraint that shaped everything after it: **512 MB, and the container is killed
rather than throttled.**

- `f17587b` — serving memory **947 MB → 343 MB**, via Arrow predicate pushdown, 16,384-row
  groups, and not deduplicating where it wasn't needed. Scored output verified identical
  throughout.
- `1da4a04` — the store summary is computed during packaging and shipped, never on the box.
- `7b19a3c` — the data is fetched at container **start**, not at build. Docker caches a RUN
  layer on its command string, so a data-only refresh would have redeployed stale data
  while reporting success.
- `03ffc5a` + `b778233` — a cycle too incomplete to trust is **refused**, not published,
  and gaps NOAA drops are refilled from S3 first. A short rainfall *sum* is roughly half
  the real accumulation, and rainfall drives most busts.
- `72182d7` — the refresh becomes catch-up driven: each run ingests whichever recent cycles
  the store lacks. This is what makes GitHub's 3–5 hour cron lateness harmless.

Also that day, and easy to overlook: `f0caad7` put the whole thing on Render + GitHub
Actions for **$0**; `518241d` packed only the *current* model run into the release asset,
after naive tarring grew it ~8 MB every six hours; `299ff50` made the startup cache warm
optional and disabled it on Render, where warming in the background while a real request
scored had been OOM-killing the box; `2df5710` made a cold start degrade instead of showing
a parser error; `65d9e42` answered **HEAD** as well as GET on `/api/health`, because uptime
monitors send HEAD and a 405 looks identical to an outage from outside; and `5333593`
deleted the side-panel components the new layout had superseded.

## 1 Sep · Mobile, honesty, and ten years of data

`2e27552` … `f7315c1`

Cache warming after both kinds of deploy — `2e27552` for scheduled publishes and `5043c53`
for pushes, which bypass the refresh workflow entirely — so the first visitor is never the
one who waits.

The mobile pass, across ten viewports in two engines because judges open links on phones:
`c116f35` stopped the Model and Replay tabs scrolling sideways on **10 of 10** phone
profiles (a single-column `1fr` track is `minmax(auto, 1fr)`, and that auto floor is
min-content, so it cannot shrink); `9f238db` raised tap targets from 19–32px to ≥44px,
gated on `(hover: none) and (pointer: coarse)` so a mouse-driven laptop can never match;
and `34d963b` let touch and keyboard users catch a ticker item that previously paused only
on hover.

The backfill also needed CI work that is invisible in the result: `8792e80` installed the
live extras in the gather job, `e410219` tested on a real runner whether ten years fit,
and `de2ef71` made that test survive being killed — since a kill was the answer it existed
to detect.

`cf58aff` measures how much the ground truth itself disagrees with an independent
reanalysis — the beginning of the MERRA-2 comparison now committed under `docs/analysis/`.

Then the backfill: `ef4ced9` makes the reforecast fetch year-agnostic and pulls 2010–2019
in parallel; `f3cf45b` and `f7315c1` make ten years of training data fit in memory — the
training frame is built in chunks of cycles, cutting peak memory **5×** — *without
changing a single result*.

## 2 Sep · Evidence, and the things that were quietly wrong

The longest day, and mostly not about features.

**Proving the model is worth believing**
- `f3b805b` — tests for the three ways the pipeline could score itself unfairly: thresholds
  fitted on train only (asserted so it cannot pass vacuously), out-of-fold folds grouped by
  cycle, no observed day on both sides of the split. All three came back clean.
- `62531ff` + `1c3cdf0` — a baseline ladder answering "compared to what?", scored on the
  pipeline's own event frames and **published inside the model run**, so the comparison
  cannot describe a different model than the one answering requests.
- `11db033` — every refresh trains on the archive. Without this the next cron run would
  have replaced a 170-cycle model with a 72-cycle one and looked healthy doing it.

**Shipping ten years without spending memory**
- `8bf1f89` — training provenance is read from the run manifest, because training and
  serving now deliberately see different data: CI trains on the full archive with 16 GB,
  the box carries only what it must serve.
- `d2d8b78` — compaction drops superseded rows (~15% of the store) on every publish, with
  scored output verified byte-identical across all 72 cycles.
- `b652f3f` — the box refuses to summarise a large store rather than dying on it; measured
  at 918 MB peak, which is an instant kill at 512.
- `506e4e6` — `/api/health` reports its own memory, because the platform paywalls metrics.
- `fa2e77b` — CI gains the ability to publish the backfill and to measure what it costs to
  serve; `4a6486f` and `5920d58` then fixed that harness twice, once to show its working
  and once because it was measuring a store no deployment would ever run.
- `6db471b` — test-only dependencies stopped shipping into the 512 MB image, and a dead
  frontend dependency was removed.

**Saying true things**
- `ea83a46` — the Model page stops implying continuous coverage.
- `4e647cf` — a results file naming a superseded run is removed.
- `b090241` — the README stops *understating* the project: it advertised ROC-AUC 0.76 while
  the site served 0.843. Limitations written down.
- `d274948` — the pipeline's own history is surfaced, **refusals included**.
- `fa9ebd6` — a one-pager drafted as the item that could not be fixed late; later folded
  into the About tab so it could not drift, and reduced to what does not go stale.
- `5643729` → the MERRA-2 figures become a placeholder naming its own source, then
  `98be94e` and `501dbc5` commit the comparison and its 21,492 paired city-days, and the
  figures are filled in from data that is now in the repo.

**Bugs found by looking, not by tests**
- `cb58392` — hovering the replay chart blanked it. A range series hands the tooltip an
  array; the formatter called `.toFixed()` on it and the throw unmounted both charts.
- `33d4dc6` — on a phone, the region detail panel painted over the chart below it. The
  mobile override released `overflow` but not `max-height`, so the panel laid out short and
  rendered past it.
- `0330d43` — "Refresh now" was rendered on a deployment that cannot ingest.

**Making the product explain itself**
- `8114785` — an About tab, and an opening-screen cue pointing at Replay, which until then
  was the fourth tab with nothing suggesting it existed.
- `91e58d5` + `12ad0ce` — Replay now verifies its own prediction: a tolerance band showing
  what counts as a bust, and the actual outcome encoded on the probability markers.

---

## Work that left no commit

A substantial part of 2 Sep produced no code, and would be invisible in `git log`:

- **Measurement runs** — CI probes for whether ten years fit in memory; a byte-identical
  verification scoring all 72 cycles before and after compaction; serving-memory harness
  runs; repeated mobile audits across ten viewports and two engines.
- **A rate-limiting incident** — an automated browser sweep against *production* tripped
  Cloudflare, the site returned 429/503, and the audit measured an error page and reported
  a layout bug that did not exist. Production is not a test fixture.
- **Decisions taken and rejected** — the Kerala 2018 case study was dropped in favour of
  the backfill (stronger evidence, less risk); app-level rate limiting and "anti-phishing"
  were declined as theatre on a service with no login and no user data; store compaction
  was deferred, then reversed when it turned out to be worth 15% of the store.
- **Two near-misses caught by looking forward rather than stopping at "it works"** — the
  scheduled refresh would have replaced the ten-year model with a 72-cycle one about two
  hours after it shipped; and a summary sidecar mismatch would have OOM-killed the box on
  the first request after a deploy.

## What this project learned about itself

1. **A number written into a file drifts from the thing that produces it.** This happened
   three times in one day — `docs/results.md`, the README, and a hardcoded "17 cycles"
   note. All three now point at served values or are generated.
2. **Refusing bad data is the product.** The completeness guard, the summary guard, and the
   visible refusals in the pipeline log are the same argument in three places.
3. **Verification beats reasoning.** A "+516 MB memory cost" turned out to be a harness
   that forgot to rebuild a cache; a "mobile layout bug" turned out to be a rate-limited
   error page; a CI memory harness turned out to have ±55 MB of noise. Each was caught by
   measuring twice, and each would have been shipped on the strength of a plausible story.
4. **Tests did not find the two bugs a person did.** Both came from looking at a real
   screen — one from hovering, one from a phone.

## Where to look for current state

- **`HANDOFF.md`** — operational state and traps. Note it is **gitignored**, so it exists
  only on the machine that wrote it.
- **`docs/known-issues.md`** — what is knowingly unfixed.
- **`/api/model/status`, `/api/health`, `/api/ingest/runs`** — the live numbers. Read them
  rather than trusting any document, including this one.
