import { useMemo } from "react";
import type { AllRegionsResponse, RegionsResponse } from "../../api/types";

export function LeadDayRail({ all, value, onChange }: {
  all?: AllRegionsResponse;
  value: number;
  onChange: (d: number) => void;
}) {
  const rows = useMemo(() => (all?.days ?? []).map(summarise).filter(Boolean) as Row[], [all]);
  const drivers = useMemo(() => driverRuns(rows), [rows]);

  if (rows.length < 2) return null;
  const peak = Math.max(...rows.map((r) => r.mean));

  return (
    <aside className="rail" aria-label="Forecast horizon">
      <h3 className="rail__title">Horizon</h3>

      <div className="rail__rows" role="group" aria-label="Lead day">
        {rows.map((r) => (
          <button
            key={r.lead}
            type="button"
            aria-pressed={r.lead === value}
            className={r.lead === value ? "railrow railrow--active" : "railrow"}
            onClick={() => onChange(r.lead)}
            title={`Lead day ${r.lead}: ${r.bust} bust, ${r.watch} watch, ${r.low} low`}
          >
            <span className="railrow__day">D{r.lead}</span>

            <span className="railrow__bar" aria-hidden="true">
              {r.low ? <i className="railrow__seg railrow__seg--low" style={{ flexGrow: r.low }} /> : null}
              {r.watch ? <i className="railrow__seg railrow__seg--watch" style={{ flexGrow: r.watch }} /> : null}
              {r.bust ? <i className="railrow__seg railrow__seg--bust" style={{ flexGrow: r.bust }} /> : null}
            </span>

            <span className={r.mean >= peak - 1e-9 ? "railrow__mean railrow__mean--peak" : "railrow__mean"}>
              {(r.mean * 100).toFixed(0)}%
            </span>
          </button>
        ))}
      </div>

      <p className="rail__key">
        <b><i className="railrow__seg--low" />low</b>
        <b><i className="railrow__seg--watch" />watch</b>
        <b><i className="railrow__seg--bust" />bust</b>
      </p>

      <p className="rail__note">
        How many regions fall in each band at each lead day, and their average bust risk.
      </p>

      {drivers ? <p className="rail__driver">{drivers}</p> : null}
    </aside>
  );
}

type Row = {
  lead: number;
  n: number;
  low: number;
  watch: number;
  bust: number;
  mean: number;
  driver: string | null;
};

function summarise(day: RegionsResponse): Row | null {
  const regions = day.regions ?? [];
  if (!regions.length) return null;

  const probs: number[] = [];
  const drivers = new Map<string, number>();
  let low = 0;
  let watch = 0;
  let bust = 0;

  for (const r of regions) {
    if (r.risk_band === "high") bust += 1;
    else if (r.risk_band === "medium") watch += 1;
    else if (r.risk_band === "low") low += 1;
    if (r.bust_probability != null) probs.push(r.bust_probability);
    if (r.dominant_variable) drivers.set(r.dominant_variable, (drivers.get(r.dominant_variable) ?? 0) + 1);
  }
  if (!probs.length) return null;

  const top = [...drivers.entries()].sort((a, b) => b[1] - a[1])[0];
  return {
    lead: day.lead_time_days,
    n: regions.length,
    low,
    watch,
    bust,
    mean: probs.reduce((s, p) => s + p, 0) / probs.length,
    driver: top?.[0] ?? null,
  };
}

function driverRuns(rows: Row[]): string | null {
  const named = rows.filter((r) => r.driver);
  if (!named.length) return null;

  const runs: { driver: string; from: number; to: number }[] = [];
  for (const r of named) {
    const last = runs[runs.length - 1];
    if (last && last.driver === r.driver) last.to = r.lead;
    else runs.push({ driver: r.driver as string, from: r.lead, to: r.lead });
  }

  if (runs.length === 1) return `${pretty(runs[0].driver)} is the main cause at every lead day.`;

  if (runs.length > 3) return `The main cause changes ${runs.length - 1} times across the ten days.`;
  return `Mostly driven by: ${runs.map((r) => `${pretty(r.driver)} ${span(r)}`).join(", then ")}.`;
}

const span = (r: { from: number; to: number }) => (r.from === r.to ? `at D${r.from}` : `D${r.from}–D${r.to}`);

const pretty = (v: string) => v.replace(/_(pct|mm|c|hpa|ms|deg|kgm2)$/, "").replace(/_/g, " ");
