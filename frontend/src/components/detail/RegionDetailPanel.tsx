import { useState } from "react";
import { useRegionDetail } from "../../hooks/useDashboardData";
import { EmptyState, ErrorState, LoadingState, RiskBadge } from "../common/States";
import { BustProbabilityCurve } from "./BustProbabilityCurve";
import { ShapFactorsList } from "./ShapFactorsList";
import { VariableTrajectoryChart } from "./VariableTrajectoryChart";

export function RegionDetailPanel({
  regionId,
  onClose,
  riskCuts,
}: {
  regionId: string | null;
  onClose: () => void;
  riskCuts?: { medium: number; high: number };
}) {
  const { data, isLoading, error } = useRegionDetail(regionId);
  const [activeVariable, setActiveVariable] = useState<string | null>(null);

  if (!regionId) {
    return (
      <aside className="panel">
        <EmptyState title="No region selected" message="Click a state on the map to see its forecast detail." />
      </aside>
    );
  }
  if (isLoading) return <aside className="panel"><LoadingState label="Loading region…" /></aside>;
  if (error) return <aside className="panel"><ErrorState error={error} /></aside>;
  if (!data) return null;

  if (!data.model_trained) {
    return (
      <aside className="panel">
        <EmptyState title={data.region_name ?? regionId} message={data.message} />
      </aside>
    );
  }

  const available = data.variables.filter((v) => v.available);
  const unavailable = data.variables.filter((v) => !v.available);
  const current = available.find((v) => v.variable === activeVariable) ?? available[0];
  const worst = [...data.bust_probability_curve].sort((a, b) => b.bust_probability - a.bust_probability)[0];

  return (
    <aside className="panel">
      <header className="panel__head">
        <div>
          <h2>{data.region_name ?? data.region_id}</h2>
          <p className="muted small">
            {data.region_id}
            {data.init_date ? ` · cycle ${data.init_date}` : ""}
          </p>
        </div>
        <button type="button" className="btn btn--ghost" onClick={onClose} aria-label="Close detail panel">×</button>
      </header>

      {data.message ? <p className="muted">{data.message}</p> : null}

      {worst ? (
        <section className="panel__section">
          <h3>Peak bust risk</h3>
          <div className="peak">
            <span className="peak__value">{(worst.bust_probability * 100).toFixed(1)}%</span>
            <div className="peak__meta">
              <RiskBadge band={worst.risk_band} />
              <span className="muted small">at lead day {worst.lead_time_days}</span>
            </div>
          </div>
          {worst.dominant_variable ? (
            <p className="muted small">Mostly driven by: {worst.dominant_variable.replace(/_/g, " ")}</p>
          ) : null}
        </section>
      ) : null}

      <section className="panel__section">
        <h3>Bust probability by lead day</h3>
        <BustProbabilityCurve points={data.bust_probability_curve} cuts={riskCuts} />
      </section>

      <section className="panel__section">
        <h3>Forecast vs what actually happened</h3>
        {available.length ? (
          <>
            <div className="tabs" role="tablist">
              {available.map((v) => (
                <button
                  key={v.variable}
                  role="tab"
                  aria-selected={current?.variable === v.variable}
                  className={current?.variable === v.variable ? "tab tab--active" : "tab"}
                  onClick={() => setActiveVariable(v.variable)}
                >
                  {v.variable}
                </button>
              ))}
            </div>
            {current ? (
              <>
                <VariableTrajectoryChart series={current} />
                <dl className="metrics">
                  <div><dt>Average error (MAE)</dt><dd>{fmt(current.model_mae)} {current.unit ?? ""}</dd></div>
                  <div><dt>Typical error (RMSE)</dt><dd>{fmt(current.model_rmse)}</dd></div>
                  <div><dt>Variance explained (R²)</dt><dd>{fmt(current.model_r2)}</dd></div>
                  <div><dt>Bust threshold</dt><dd>{fmt(current.bust_threshold)} {current.unit ?? ""}</dd></div>
                </dl>
                <p className="muted small">
                  Average error is how far this variable&apos;s forecast lands from reality on a
                  normal day{current.unit ? <>, in {current.unit}</> : null}. It counts as a bust
                  once the error passes the threshold.
                </p>
                <p className="muted small">
                  Measured on forecasts this model never trained on (the{" "}
                  <b>{current.metrics_split ?? "unknown"}</b> split of the current run).
                </p>
              </>
            ) : null}
          </>
        ) : (
          <p className="muted">No variable has a trained model yet.</p>
        )}
        {unavailable.length ? (
          <p className="muted small">
            Not modelled (too few matched forecast–observation pairs):{" "}
            {unavailable.map((v) => v.variable).join(", ")}
          </p>
        ) : null}
      </section>

      <section className="panel__section">
        <h3>Why this region — what drove the prediction</h3>
        <ShapFactorsList factors={data.top_factors} method={data.top_factors_method} />
      </section>

      <section className="panel__section">
        <h3>Similar past cases</h3>
        <p className="muted small">
          Searching for similar past forecasts is not built yet, so nothing is shown here
          rather than an invented match.
        </p>
      </section>
    </aside>
  );
}

const fmt = (v: number | null | undefined) =>
  v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(3);
