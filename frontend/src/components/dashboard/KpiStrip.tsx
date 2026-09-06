import { useMemo } from "react";
import type { AllRegionsResponse, RegionsResponse } from "../../api/types";

export function KpiStrip({ all, day }: { all?: AllRegionsResponse; day?: RegionsResponse }) {
  const stats = useMemo(() => derive(all, day), [all, day]);
  if (!stats) return null;

  return (
    <section className="kpis" aria-label="Cycle summary">
      <Kpi
        cap={stats.meanCap}
        label="Mean bust risk"
        value={stats.mean != null ? `${(stats.mean * 100).toFixed(0)}%` : "—"}
        note={
          stats.scored
            ? <>across <b>{stats.scored}</b> scored regions · day {stats.lead}</>
            : "no scored regions for this lead day"
        }
      />
      <Kpi
        cap="bust"
        label="Bust-risk regions"
        value={
          stats.scored ? (
            <>
              {stats.high}
              <small>/ {stats.scored}</small>
            </>
          ) : (
            "—"
          )
        }
        note={stats.scored ? <>in the bust band at day {stats.lead}</> : "nothing scored yet"}
      />
      <Kpi
        cap="blue"
        label="Confidence change"
        value={
          stats.decayPct != null ? (
            <>
              {stats.decayPct > 0 ? "+" : ""}
              {stats.decayPct.toFixed(0)}
              <small>%</small>
            </>
          ) : (
            "—"
          )
        }
        note={
          stats.decayPct != null && stats.decayFrom != null && stats.decayTo != null ? (
            <>
              from day {stats.decayFrom} to day {stats.decayTo}
            </>
          ) : (
            "needs two or more scored lead days"
          )
        }
      />
      <Kpi
        cap="watch"
        label="Peak risk"
        value={stats.peak ? `${(stats.peak.value * 100).toFixed(0)}%` : "—"}
        note={
          stats.peak ? (
            <>
              <b>{stats.peak.name}</b>
              {stats.peak.driver ? <> · {label(stats.peak.driver)}</> : null}
            </>
          ) : (
            "no region scored for this lead day"
          )
        }
      />
    </section>
  );
}

function Kpi({ cap, label, value, note }: {
  cap: string;
  label: string;
  value: React.ReactNode;
  note: React.ReactNode;
}) {
  return (
    <article className="kpi rise">
      <div className={`kpi__cap kpi__cap--${cap}`} aria-hidden="true" />
      <div className="kpi__label">{label}</div>
      <div className="kpi__value">{value}</div>
      <p className="kpi__note">{note}</p>
    </article>
  );
}

function derive(all?: AllRegionsResponse, day?: RegionsResponse) {
  if (!all?.model_trained || !day) return null;

  const scoredRegions = day.regions.filter((r) => r.bust_probability != null);
  const probs = scoredRegions.map((r) => r.bust_probability as number);
  const mean = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null;
  const high = day.regions.filter((r) => r.risk_band === "high").length;

  const peakRow = scoredRegions.reduce<(typeof scoredRegions)[number] | null>(
    (best, r) => (best == null || (r.bust_probability as number) > (best.bust_probability as number) ? r : best),
    null,
  );

  const withConfidence = all.days
    .map((d) => ({ lead: d.lead_time_days, mean: meanConfidence(d) }))
    .filter((d): d is { lead: number; mean: number } => d.mean != null)
    .sort((a, b) => a.lead - b.lead);

  const first = withConfidence[0];
  const last = withConfidence[withConfidence.length - 1];
  const canDecay = first && last && first.lead !== last.lead && first.mean > 0;

  return {
    lead: day.lead_time_days,
    scored: scoredRegions.length,
    mean,
    meanCap: mean == null ? "blue" : mean >= 0.57 ? "bust" : mean >= 0.36 ? "watch" : "calm",
    high,
    peak: peakRow
      ? {
          name: peakRow.region_name ?? peakRow.region_id,
          value: peakRow.bust_probability as number,
          driver: peakRow.dominant_variable,
        }
      : null,
    decayPct: canDecay ? ((last.mean - first.mean) / first.mean) * 100 : null,
    decayFrom: canDecay ? first.lead : null,
    decayTo: canDecay ? last.lead : null,
  };
}

function meanConfidence(day: RegionsResponse): number | null {
  const vals = day.regions
    .map((r) => r.confidence)
    .filter((c): c is number => c != null);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

function label(variable: string): string {
  return variable.replace(/_(c|pct|hpa|mm|ms|deg|kgm2)$/, "").replace(/_/g, " ");
}
