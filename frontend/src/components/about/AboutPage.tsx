import { useModelStatus } from "../../hooks/useDashboardData";
import { ErrorState, LoadingState } from "../common/States";

export function AboutPage({ onReplay }: { onReplay: () => void }) {
  const { data, isLoading, error } = useModelStatus();

  if (isLoading) return <main className="page page--wide"><LoadingState label="Loading…" /></main>;
  if (error) return <main className="page page--wide"><ErrorState error={error} /></main>;
  if (!data) return null;

  const clf = data.validation_metrics?.classifier ?? {};
  const td = data.training_data ?? {};
  const thr = data.thresholds?.bust_threshold ?? {};
  const bl = data.baselines ?? {};
  const n3 = (v: unknown, dp = 3) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(dp) : "—";

  return (
    <main className="page page--wide">
      <header className="pagehead">
        <div>
          <h1 className="pagehead__title">What this is</h1>
          <p className="pagehead__sub">
            Sanket — संकेत, “signal”. Predicting where a weather forecast will be wrong,
            and saying why.
          </p>
        </div>
      </header>

      <div className="notice">
        <p style={{ margin: 0 }}>
          <b>Start here:</b> Replay takes a real forecast cycle from the archive, scores it
          with the deployed model, and shows what this system would have told a forecaster
          that day — before anyone knew the outcome.
        </p>
        <p style={{ margin: "0.6rem 0 0" }}>
          <button type="button" className="chip" onClick={onReplay}>
            Open Replay →
          </button>
        </p>
      </div>

      <div className="page--cols">
        <div className="page__col">
          <section className="card">
            <header className="card__head"><h3>The question it answers</h3></header>
            <p className="muted">
              Every operational centre already issues a forecast. This does not try to make a
              better one. It answers the question a duty forecaster actually has at 6am:
            </p>
            <p><b>“How likely is the forecast I am holding to be badly wrong today?”</b></p>
            <p className="muted small">
              Correction and confidence are different services. NCMRWF already corrects its
              forecasts statistically, and busts still happen. Correction removes the errors a
              model makes <i>consistently</i>; a bust comes from the particular weather pattern
              of that day, which is not consistent and so is not corrected away. Knowing when
              to distrust a forecast is what decides whether a warning goes out.
            </p>
            <p className="muted small">
              A <b>bust</b> is a forecast whose error lands in the tail of that variable’s
              own historical error distribution — the 90th percentile, computed on training
              data only. Each variable therefore has its own threshold, in its own units:
            </p>
            <ul className="taglist">
              {Object.entries(thr).map(([v, t]) => (
                <li key={v} className="tag">{v} ≥ {n3(t, 2)}</li>
              ))}
            </ul>
          </section>

          <section className="card">
            <header className="card__head"><h3>How well it works</h3></header>
            <dl className="metrics metrics--compact metrics--hero">
              <div><dt>ROC-AUC</dt><dd>{n3(clf.roc_auc)}</dd></div>
              <div><dt>F1</dt><dd>{n3(clf.f1)}</dd></div>
              <div><dt>Brier</dt><dd>{n3(clf.brier)}</dd></div>
              <div>
                <dt>Held-out forecasts</dt>
                <dd>{typeof clf.n === "number" ? clf.n.toLocaleString() : "—"}</dd>
              </div>
            </dl>
            <p className="muted small">
              <b>Reading these:</b> ROC-AUC is the chance the model ranks a real bust above a
              non-bust — 0.5 is a coin flip, 1.0 is perfect. F1 balances how often its warnings
              are right against how many busts it catches. Brier is the average error in the
              probability itself, so lower is better.
            </p>
            <p className="muted small">
              Measured on the <b>{String(clf.split ?? "held-out")}</b> split — forecast
              cycles the model never trained on{" "}
              {typeof td.held_out_cycles === "number" ? <>— {td.held_out_cycles} of them</> : null}
              {typeof td.cycles === "number" ? <> out of {td.cycles} in total</> : null}.
            </p>
            <p className="muted small">
              Because a bust is defined against each variable’s <i>own</i> error percentile
              rather than an absolute error, the label does not simply grow with lead time —
              so this skill is not a rediscovery of “day 10 is worse than day 1”.
            </p>

            {bl.models?.length ? (
              <>
                <header className="card__head"><h4>Compared to what?</h4></header>
                <p className="muted small">
                  Every baseline is fitted on the training split alone and scored on the same
                  held-out rows as the model. Each is compared against climatology — simply
                  guessing the long-run bust rate every time. <b>0.000 means it does no better
                  than that guess</b>; higher is better.
                </p>
                <div className="tablewrap">
                  <table className="dtable">
                    <thead>
                      <tr>
                        <th>model</th><th>Brier ↓</th><th>skill vs climatology ↑</th><th>ROC-AUC ↑</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bl.models.map((m) => (
                        <tr key={m.name} className={m.is_model ? "is-active" : undefined}>
                          <td>{m.is_model ? <b>{m.name}</b> : m.name}</td>
                          <td className="mono">{n3(m.brier, 4)}</td>
                          <td className="mono">{n3(m.bss, 4)}</td>
                          <td className="mono">{n3(m.roc_auc, 4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="muted small">↓ lower is better · ↑ higher is better</p>
                {typeof bl.lead_bust_correlation?.test === "number" ? (
                  <p className="muted small">
                    Guessing from the lead day alone scores at or below zero, because busts do
                    not simply become more likely further out. The measured link between lead
                    day and bust is only {n3(bl.lead_bust_correlation.train, 3)} on training
                    data and {n3(bl.lead_bust_correlation.test, 3)} on held-out data.
                  </p>
                ) : null}
              </>
            ) : (
              <p className="muted small">
                The baseline comparison is generated by the training run; this model
                predates it and the table is written on the next retrain.
              </p>
            )}
          </section>
        </div>

        <div className="page__col">
          <section className="card">
            <header className="card__head"><h3>Why the numbers can be trusted</h3></header>
            <ul className="notes">
              <li>
                <b>No synthetic, mocked or placeholder data anywhere</b> — not even as a
                fallback. Where a number cannot be computed from real data, this interface
                shows an em dash <i>and the reason</i>. That rule is enforced in the code,
                not just stated here.
              </li>
              <li>
                <b>Real data end to end.</b> NOAA’s GEFS forecasts — historical re-runs for
                training, the live feed for today — checked against ERA5, a reconstructed
                record of what the weather actually did.
              </li>
              <li>
                <b>Leakage is tested, not asserted.</b> Bust thresholds are fitted on the
                training split only; when the model is checked against itself, whole forecast
                runs are kept together so it is never tested on a run it partly trained on; and
                no observed day appears on both sides of the train/test split. Each is a test in
                the suite, written so it cannot pass by accident.
              </li>
              <li>
                <b>Even the “truth” we score against is uncertain.</b> Two leading global
                weather records disagree with each other by roughly a quarter to a half of a
                bust threshold, measured over ~21,000 city-days. Stated up front rather than
                waiting to be asked.
              </li>
            </ul>
          </section>

          <section className="card">
            <header className="card__head"><h3>What it does not do</h3></header>
            <ul className="notes">
              <li>
                Coverage is <b>sampled, not continuous</b> — an archive that re-runs the
                forecast model on a handful of past dates each year, plus live runs since
                deployment.
              </li>
              <li>
                City points, not full regional coverage of India. Region-level readings are
                indicative.
              </li>
              <li>
                GEFS runs 31 parallel forecasts; this uses 5 of them. How much those disagree
                is one of the model’s inputs, so that input is noisier here than it would be
                with all 31.
              </li>
              <li>
                ERA5 precipitation is weak over India relative to gauge-based products, and
                rainfall results carry that caveat. Rainfall is also the hardest variable
                here: most days have none at all, and the rest are dominated by a few extreme
                ones.
              </li>
              <li>
                <b>It does not send alerts to anyone.</b> Warnings appear in this interface
                and nowhere else — there is no SMS, email or push delivery, and nothing is
                dispatched to a duty desk. This is a decision-support view a forecaster
                reads, not a warning system that acts on its own.
              </li>
              <li>
                A bust is defined on <b>surface-variable error</b>, not the synoptic Z500
                criterion of Rodwell et al. (2013). That is a deliberate choice: surface
                error is what reaches agriculture and disaster response, whereas Z500 is
                what reaches meteorologists.
              </li>
            </ul>
          </section>

        <section className="card">
          <header className="card__head"><h3>How it runs</h3></header>
          <p className="muted small">
            FastAPI and XGBoost behind React and TypeScript, over a hive-partitioned Parquet
            store — one origin, one process. Training runs on CI with 16 GB; the site is
            served from a 512 MB free-tier instance that <b>cannot train</b>, which is why
            serving memory was cut to fit with scored output verified byte-identical at every
            step. When the upstream feed drops forecast steps mid-pull, a fallback source
            refills them before the daily reduction — a short rainfall <i>sum</i> would
            otherwise silently halve the quantity that drives most busts. Total hosting
            cost: nothing.
          </p>
        </section>
        </div>
      </div>
    </main>
  );
}
