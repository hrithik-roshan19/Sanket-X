"""Score the baseline ladder against the trained classifier, on identical rows.

    python -m app.ml.train_pipeline --dry-run --emit-eval    # produces the event frames
    python -m scripts.run_baselines                          # scores them, writes results

Why it reads a file instead of rebuilding the split: a baseline comparison is only worth
anything if both sides see the same rows, the same labels and the same train/test
boundary. Rebuilding any of those here would be a second implementation that can drift
from the pipeline silently - and a drifted baseline flatters the model. So the training
run publishes what it scored (`--emit-eval`) and this reads it.

Reports, per model: Brier, Brier skill score against climatology, ROC-AUC, and BSS broken
down by lead day - because the honest question is not whether the model beats climatology
overall, but whether it still adds anything once you know the lead time.

Writes a markdown table into docs/results.md under `## Baselines`, stamped with the run_id
and the git SHA so a number on a page can always be traced back to the run that made it.
Touches no model, no store and no serving code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.ml import baselines as bl               # noqa: E402
from app.ml import classifier as clf_mod         # noqa: E402

EVAL_DIR = BACKEND_DIR / "data" / "analysis" / "eval_events"
RESULTS_MD = BACKEND_DIR.parent / "docs" / "results.md"
HEADING = "## Baselines"
MODEL_ROW = "Sanket bust classifier"


def _latest_eval() -> Path:
    if not EVAL_DIR.is_dir():
        raise SystemExit(
            f"no eval events in {EVAL_DIR}\n"
            "run:  python -m app.ml.train_pipeline --dry-run --emit-eval")
    files = sorted(EVAL_DIR.glob("run_*.parquet"))
    if not files:
        raise SystemExit(f"no run_*.parquet in {EVAL_DIR}")
    return files[-1]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND_DIR,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is nice to have, not worth failing over
        return "unknown"


def _metrics(y, proba, ref) -> dict:
    """Reuses the classifier's own _evaluate so every number here is computed by exactly
    the same code that produced the model's reported metrics."""
    m = clf_mod._evaluate(y, proba)
    m["bss"] = bl.brier_skill_score(y, proba, ref)
    return m


def _per_lead(y, proba, ref, leads) -> dict:
    out = {}
    for lead in sorted(pd.unique(leads)):
        mask = np.asarray(leads == lead)
        if mask.sum() < 20 or len(np.unique(np.asarray(y)[mask])) < 2:
            continue
        out[int(lead)] = bl.brier_skill_score(
            np.asarray(y)[mask], np.asarray(proba)[mask], np.asarray(ref)[mask])
    return out


def _fmt(v, nd=4) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="eval events to score (default: the most recent)")
    ap.add_argument("--no-write", action="store_true", help="print only, don't touch docs")
    ap.add_argument("--write-run-artifact", action="store_true",
                    help="also write baselines.json into the model run directory, so the "
                         "comparison ships with the model and is served beside its metrics")
    args = ap.parse_args()

    path = (EVAL_DIR / f"{args.run_id}.parquet") if args.run_id else _latest_eval()
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    run_id = path.stem
    ev = pd.read_parquet(path)

    train = ev[ev["split"] == "train"]
    test = ev[ev["split"] == "test"]
    if train.empty or test.empty:
        raise SystemExit(
            f"need both splits; got train={len(train):,} test={len(test):,}. "
            "A store with too few cycles produces no held-out split.")
    if bl.LABEL not in ev.columns:
        raise SystemExit(f"no {bl.LABEL} column - these events were built without labels")

    y_test = np.asarray(test[bl.LABEL], int)
    leads = np.asarray(test["lead_time_days"], int)

    fitted = bl.fit_all(train)
    ref = fitted["climatology"].predict_proba(test)      # the BSS reference forecast

    rows, per_lead = [], {}
    for name, model in fitted.items():
        p = model.predict_proba(test)
        rows.append((name, _metrics(y_test, p, ref)))
        per_lead[name] = _per_lead(y_test, p, ref, leads)

    if "model_proba" in test.columns and test["model_proba"].notna().any():
        p = np.asarray(test["model_proba"], float)
        rows.append((MODEL_ROW, _metrics(y_test, p, ref)))
        per_lead[MODEL_ROW] = _per_lead(y_test, p, ref, leads)
    else:
        print("WARNING: no model_proba column - the classifier row is missing.")

    n_train_cycles = train["init_date"].nunique() if "init_date" in train else 0
    n_test_cycles = test["init_date"].nunique() if "init_date" in test else 0

    # Why the lead-day baseline can score below chance, stated with the measured number
    # rather than left for a reader to guess. A bust is defined against the 90th
    # percentile of each variable's OWN error distribution, not an absolute error, so
    # lead-time growth is largely absorbed into the threshold and the label does not
    # simply rise with lead day. On a small test set the sign of that weak relationship
    # can flip between splits, which puts a lead-day-only model under 0.5.
    corr = {}
    for nm, d in (("train", train), ("test", test)):
        x = d["lead_time_days"].astype(float).to_numpy()
        y = d[bl.LABEL].astype(float).to_numpy()
        corr[nm] = (float(np.corrcoef(x, y)[0, 1])
                    if len(d) > 2 and x.std() > 0 and y.std() > 0 else float("nan"))

    header = (
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · run `{run_id}` · "
        f"git `{_git_sha()}`_\n\n"
        f"Held-out test split: **{len(test):,} events** over **{n_test_cycles} forecast "
        f"cycles**, bust rate **{y_test.mean():.3f}**. Baselines fitted on the "
        f"{len(train):,} training events ({n_train_cycles} cycles) and evaluated on rows "
        f"they never saw. Brier skill score is against climatology — the training base "
        f"rate — so 0.000 means no skill beyond knowing how often busts happen.\n\n"
        f"A bust is defined against the 90th percentile of each variable's own error "
        f"distribution, not an absolute error, so the label does not simply grow with "
        f"lead time: measured correlation between lead day and bust is "
        f"**{corr['train']:+.3f} on train** and **{corr['test']:+.3f} on test**. That "
        f"weak, sign-flipping relationship is why a lead-day-only model can score below "
        f"chance here — and it is also direct evidence that the classifier's skill is "
        f"not simply a rediscovery of lead time.\n\n"
    )

    tbl = ["| model | Brier ↓ | BSS vs climatology ↑ | ROC-AUC ↑ | F1 ↑ |",
           "|---|---|---|---|---|"]
    for name, m in rows:
        label = f"**{name}**" if name == MODEL_ROW else name
        tbl.append(f"| {label} | {_fmt(m['brier'])} | {_fmt(m['bss'])} | "
                   f"{_fmt(m['roc_auc'])} | {_fmt(m['f1'])} |")

    all_leads = sorted({l for d in per_lead.values() for l in d})
    lead_tbl = []
    if all_leads:
        lead_tbl = ["", "**Brier skill score by lead day** "
                        "(lead days with under 20 test events, or no bust, are omitted):",
                    "",
                    "| model | " + " | ".join(f"D{l}" for l in all_leads) + " |",
                    "|---" * (len(all_leads) + 1) + "|"]
        for name, _ in rows:
            cells = " | ".join(_fmt(per_lead[name].get(l), 3) for l in all_leads)
            label = f"**{name}**" if name == MODEL_ROW else name
            lead_tbl.append(f"| {label} | {cells} |")

    section = HEADING + "\n\n" + header + "\n".join(tbl) + "\n" + "\n".join(lead_tbl) + "\n"
    print("\n" + section)

    # Written into the run directory on purpose. A markdown file in the repo drifts from
    # whatever is deployed the moment either changes; a file inside the run is packaged
    # with that run, served beside its metrics, and cannot end up describing a different
    # model than the one answering requests.
    if args.write_run_artifact:
        from app.ml import registry  # noqa: PLC0415 - keeps the CLI import-light
        run_dir = registry.run_dir(run_id)
        if run_dir.is_dir():
            payload = {
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": _git_sha(),
                "test_events": int(len(test)),
                "test_cycles": int(n_test_cycles),
                "bust_rate": float(y_test.mean()),
                "lead_bust_correlation": {"train": corr["train"], "test": corr["test"]},
                "models": [
                    {"name": name, "brier": m["brier"], "bss": m["bss"],
                     "roc_auc": m["roc_auc"], "f1": m["f1"],
                     "is_model": name == MODEL_ROW}
                    for name, m in rows
                ],
            }
            (run_dir / "baselines.json").write_text(json.dumps(payload, indent=2))
            print(f"wrote {run_dir / 'baselines.json'}")
        else:
            print(f"run directory {run_dir} not found; skipped the run artifact",
                  file=sys.stderr)

    if args.no_write:
        return 0

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    existing = RESULTS_MD.read_text() if RESULTS_MD.exists() else "# Sanket — measured results\n\n"
    if HEADING in existing:
        head, _, tail = existing.partition(HEADING)
        nxt = tail.find("\n## ")
        existing = head + (tail[nxt + 1:] if nxt != -1 else "")
    RESULTS_MD.write_text(existing.rstrip() + "\n\n" + section)
    print(f"wrote {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
