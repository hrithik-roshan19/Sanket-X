"""Pull the newest GEFS cycle, verify what can be verified, retrain, and report.

This is what CI runs on a schedule to keep the deployed site current. It exists because a
512 MB serving box cannot retrain - a full retrain peaks around 2 GB - so the work happens
on a 16 GB GitHub Actions runner and only the *result* (model + canonical store + metadata
db) is shipped to the host.

Run it from the backend directory, with the live extras installed:

    python -m scripts.refresh_for_deploy            # catch up on the last 4 cycles
    python -m scripts.refresh_for_deploy --skip-obs # forecast only, no verification pull
    python -m scripts.refresh_for_deploy --cycle 2026-08-31T00 --force-cycle
                                                    # re-pull one cycle, replacing it

By default the run ingests every cycle in the last `--catch-up` that is not already in the
store, oldest first - not just the newest one. GitHub's scheduler is best-effort and has
already fired this workflow 3 h 25 m late once and skipped a slot entirely, so a run that
only ever pulled "the latest cycle" would leave a permanent hole in the store every time
cron misfired. Catching up turns "every run must fire" into "some run must fire".

Exit status is what CI branches on:

    0  a trained model is on disk and worth publishing
    1  the retrain failed, or nothing is trained
    2  the cycle arrived too incomplete to publish (see _reject_reason)

A forecast pull that finds nothing new is NOT a failure - the cycle may simply not have
published yet - so it exits 0 and leaves the previous artifact in place. Exit 2 is a
deliberate refusal rather than a crash: the run goes red so it is visible, and the live
site keeps serving the last cycle that was actually complete.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from typing import Optional

# The live pull reaches out to NOAA, which the scheduler gates behind this flag. CI is the
# one place it is deliberately on; set it before app.config is imported anywhere.
os.environ.setdefault("LIVE_INGEST_ENABLED", "true")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("refresh")


def _reject_reason(result: dict, args) -> Optional[str]:
    """Why this cycle is unfit to publish, or None if it is fine.

    A cycle arrives incomplete when NOAA is slow and some GRIB steps time out. The pipeline
    still produces values from what it got, and those values are real - but a daily
    aggregate built from fewer steps is not the quantity the models were trained on, and
    for accumulations it is not even close: rainfall summed from 2 of 4 steps is roughly
    half the true total, and rainfall drives most busts. Publishing that would understate
    risk on the exact variable the product exists to warn about.

    So the default is to refuse. The previous cycle stays live, which is a slightly older
    forecast rather than a subtly wrong one, and the next run re-attempts the same cycle.

    Only a completed pull is judged; "skipped" means nothing new was ingested at all.
    """
    if result.get("status") != "complete":
        return None

    # Both flags may be absent if the orchestrator is older than this script; an unknown
    # completeness is not evidence of a bad cycle, so the check is skipped rather than
    # guessed at.
    completeness = result.get("step_completeness")
    if completeness is None:
        fetched, expected = result.get("steps_fetched"), result.get("steps_expected")
        if fetched is not None and expected:
            completeness = fetched / expected
    if completeness is not None and args.min_steps > 0 and completeness < args.min_steps:
        return (f"only {completeness:.1%} of expected GRIB steps arrived "
                f"(need {args.min_steps:.0%}); {len(result.get('missing_steps') or [])} missing")

    short_sums = result.get("undersampled_sum_count") or 0
    if short_sums and not args.allow_short_accumulations:
        # The reported list is truncated, so it may hold no example even when the count is
        # non-zero; say so rather than printing an empty "e.g.".
        examples = [str(u) for u in (result.get("undersampled") or []) if "apcp" in str(u)]
        shown = ", ".join(examples[:3]) if examples else "not in the truncated sample"
        return (f"{short_sums} accumulated group(s) summed from too few steps, which "
                f"understates rainfall (e.g. {shown})")

    return None


def _parse_cycle(text: str) -> tuple:
    """``YYYY-MM-DDTHH`` -> ``(date, "HH")``, for re-pulling one named cycle."""
    raw = text.strip().replace(" ", "T").replace("/", "T")
    parts = raw.split("T")
    if len(parts) != 2:
        raise ValueError(f"cannot read cycle {text!r}; expected YYYY-MM-DDTHH, "
                         f"e.g. 2026-08-31T00")
    try:
        init, hour = date.fromisoformat(parts[0]), f"{int(parts[1]):02d}"
    except ValueError:
        raise ValueError(f"cannot read cycle {text!r}; expected YYYY-MM-DDTHH, "
                         f"e.g. 2026-08-31T00")
    if hour not in ("00", "06", "12", "18"):
        raise ValueError(f"{hour}Z is not a GEFS cycle hour (00, 06, 12 or 18)")
    return init, hour


def _cycles_to_pull(args) -> list:
    """The cycles this run should attempt, OLDEST FIRST.

    Oldest first because the run publishes all-or-nothing: if a later cycle comes back thin
    the whole run is discarded, and what was ingested up to that point should be a
    contiguous history rather than one with a hole in the middle.
    """
    if args.cycle:
        return [_parse_cycle(args.cycle)]
    from app.live import gefs
    # stride 6, not recent_cycles' 24 h default - that default deliberately spreads cycles
    # across days for the historical backfill, which is the opposite of what catching up on
    # consecutive missed slots needs.
    return list(reversed(gefs.recent_cycles(args.catch_up, stride_hours=6)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-forecast", action="store_true", help="do not pull a GEFS cycle")
    ap.add_argument("--cycle", metavar="YYYY-MM-DDTHH",
                    help="pull exactly this cycle instead of catching up, e.g. "
                         "2026-08-31T00. Only 00, 06, 12 and 18 are GEFS cycle hours.")
    ap.add_argument("--force-cycle", action="store_true",
                    help="re-pull a cycle already marked ingested. The store dedupes on "
                         "most-recently-ingested, so the new rows supersede the old ones - "
                         "this is how a cycle published while thin gets replaced.")
    ap.add_argument("--catch-up", type=int, default=4, metavar="N",
                    help="ingest any of the last N cycles missing from the store, oldest "
                         "first (default 4 = one day). 1 restores the old behaviour of "
                         "only ever pulling the newest cycle.")
    ap.add_argument("--skip-obs", action="store_true", help="do not pull observations")
    ap.add_argument("--skip-train", action="store_true", help="do not retrain")
    ap.add_argument("--force-train", action="store_true",
                    help="retrain even when no new observations landed")
    ap.add_argument("--min-steps", type=float, default=0.98, metavar="FRACTION",
                    help="reject a cycle that fetched less than this fraction of its "
                         "expected GRIB steps (default 0.98; 0 disables the check)")
    ap.add_argument("--allow-short-accumulations", action="store_true",
                    help="publish even when a day's rainfall was summed from fewer steps "
                         "than it should have been. Understates precipitation; only for "
                         "getting *something* out when a partial cycle beats none.")
    args = ap.parse_args()

    from app.db.base import init_db
    from app.ml import registry

    init_db()

    if not args.skip_forecast:
        from app.live import orchestrator
        try:
            wanted = _cycles_to_pull(args)
        except ValueError as exc:
            log.error("%s", exc)
            return 1

        ingested = 0
        for init, cycle_hour in wanted:
            try:
                result = orchestrator.run_forecast_cycle(
                    init, cycle_hour, trigger="scheduled", force=args.force_cycle)
            except Exception:  # noqa: BLE001
                # A cycle that is not published yet is normal - GEFS runs ~5.5 h behind its
                # init hour, and the oldest of a catch-up span may have aged off NOMADS
                # while S3 was briefly unreachable. Keep going; the others still count.
                log.exception("forecast pull failed for %s %sZ; continuing", init, cycle_hour)
                continue

            log.info("forecast cycle: %s", result)
            if result.get("status") == "skipped":
                continue

            reason = _reject_reason(result, args)
            if reason:
                # Stop before retraining, and discard the whole run rather than the one bad
                # cycle. The runner's store is rebuilt from the published artifact every
                # time and nothing persists unless we publish, so bailing now leaves no
                # partial state - which matters more than salvaging the cycles already
                # ingested, because those rows and the thin ones share a store.
                log.error("REJECTED %s: %s", result.get("target"), reason)
                log.error("nothing published; the site keeps the last good cycle and model")
                log.error("%d earlier cycle(s) this run are discarded with it", ingested)
                log.error("to publish anyway: --min-steps 0 --allow-short-accumulations")
                return 2
            ingested += 1

        log.info("ingested %d of %d cycle(s) attempted", ingested, len(wanted))

    new_rows = 0
    if not args.skip_obs:
        from app.live import orchestrator
        try:
            # retrain=False: this script decides when to train, not the orchestrator, so
            # the training run happens in *this* process where its exit code is visible.
            result = orchestrator.run_observation_refresh("final", None, trigger="scheduled",
                                                          retrain=False)
            new_rows = int(result.get("rows_ingested") or 0)
            log.info("observation refresh: %s", result)
        except Exception:  # noqa: BLE001
            log.exception("observation refresh failed; continuing")

    if not args.skip_train:
        if new_rows == 0 and not args.force_train and registry.current_run_id():
            log.info("no newly-verified rows and a model already exists; skipping retrain")
        else:
            from app.ml.train_pipeline import full_retrain
            report = full_retrain(triggered_by_batch_id=None, make_current=True)
            log.info("retrain: run_id=%s status=%s", report.run_id, report.status)
            if report.status != "success":
                log.error("retrain did not succeed; refusing to publish")
                return 1

    run_id = registry.current_run_id()
    if not run_id:
        log.error("no trained model on disk; nothing worth publishing")
        return 1

    log.info("ready to publish: run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
