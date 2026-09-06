import { useMemo } from "react";
import type { Alert, RiskBand } from "../../api/types";
import { stamp } from "../../format";
import { useAlerts } from "../../hooks/useDashboardData";
import { EmptyState, ErrorState, LoadingState, RiskBadge } from "../common/States";
import { bandLabel } from "../../theme";

const LIMIT = 200;

export function AlertsPage({ onSelect, filter, onFilter }: {
  onSelect: (regionId: string, lead: number) => void;
  filter: RiskBand | undefined;
  onFilter: (b: RiskBand | undefined) => void;
}) {
  const { data, isLoading, error } = useAlerts(LIMIT, filter);
  const alerts = data?.alerts;

  const stats = useMemo(() => summarise(alerts), [alerts]);

  return (
    <main className="page page--wide">
      <header className="pagehead">
        <div>
          <h1 className="pagehead__title">Alerts</h1>
          <p className="pagehead__sub">
            Every region above the watch level in the current forecast run, worst first.
            {stats && stats.total >= LIMIT ? ` Capped at the ${LIMIT} most severe.` : ""}
          </p>
        </div>
        <div className="filters">
          {(["high", "medium"] as RiskBand[]).map((b) => (
            <button
              key={b}
              type="button"
              className={filter === b ? "chip chip--active" : "chip"}
              onClick={() => onFilter(filter === b ? undefined : b)}
            >
              {bandLabel(b)}
            </button>
          ))}
        </div>
      </header>

      {isLoading ? <LoadingState label="Loading alerts…" /> : null}
      {error ? <ErrorState error={error} /> : null}
      {data && !data.model_trained ? <EmptyState title="No alerts yet" message={data.message} /> : null}
      {data?.model_trained && !alerts?.length ? (
        <EmptyState
          title="No watch or bust regions"
          message="Every region in the current cycle scored low."
        />
      ) : null}

      {stats ? (
        <div className="kpis kpis--flush">
          <Stat cap="bust" label="In the bust band" value={String(stats.bust)}
            note={<>of <b>{stats.total}</b> alerts shown</>} />
          <Stat cap="watch" label="In the watch band" value={String(stats.watch)}
            note={<>across <b>{stats.regions}</b> distinct regions</>} />
          <Stat cap="blue" label="Peak bust risk" value={`${(stats.peak.bust_probability * 100).toFixed(0)}%`}
            note={<><b>{stats.peak.region_name ?? stats.peak.region_id}</b> · D{stats.peak.lead_time_days}</>} />
          <Stat cap="blue" label="Most common cause" value={stats.topDriver?.[0] ?? "—"}
            note={stats.topDriver
              ? <>the main cause in <b>{stats.topDriver[1]}</b> of {stats.total}</>
              : <>no dominant variable recorded</>} />
        </div>
      ) : null}

      {alerts?.length ? (
        <section className="card card--table">
          <div className="tablewrap">
            <table className="dtable dtable--fill">
              <thead>
                <tr>
                  <th>Region</th>
                  <th className="dtable__num">Lead</th>
                  <th>Valid date</th>
                  <th className="dtable__num">Bust risk</th>
                  <th>Band</th>
                  <th>Main cause</th>
                </tr>
              </thead>
              <tbody>
                {alerts.map((a) => (
                  <tr
                    key={a.alert_id}
                    className="dtable__row"
                    tabIndex={0}
                    role="button"
                    onClick={() => onSelect(a.region_id, a.lead_time_days)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelect(a.region_id, a.lead_time_days);
                      }
                    }}
                  >
                    <td className="dtable__strong">{a.region_name ?? a.region_id}</td>
                    <td className="dtable__num mono">D{a.lead_time_days}</td>
                    <td className="mono muted">{a.valid_date ?? "—"}</td>
                    <td className="dtable__num mono dtable__strong">
                      {(a.bust_probability * 100).toFixed(0)}%
                    </td>
                    <td><RiskBadge band={a.risk_band} /></td>
                    <td className="mono muted">{a.dominant_variable ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {data?.generated_at ? (
        <p className="pagefoot">Generated {stamp(data.generated_at)}</p>
      ) : null}
    </main>
  );
}

function Stat({ cap, label, value, note }: {
  cap: string; label: string; value: string; note: React.ReactNode;
}) {
  return (
    <div className="kpi">
      <div className={`kpi__cap kpi__cap--${cap}`} />
      <span className="kpi__label">{label}</span>
      <p className="kpi__value">{value}</p>
      <p className="kpi__note">{note}</p>
    </div>
  );
}

function summarise(alerts: Alert[] | undefined) {
  if (!alerts?.length) return null;

  const drivers = new Map<string, number>();
  let bust = 0;
  let watch = 0;
  const regions = new Set<string>();
  let peak = alerts[0];

  for (const a of alerts) {
    if (a.risk_band === "high") bust += 1;
    else if (a.risk_band === "medium") watch += 1;
    regions.add(a.region_id);
    if (a.bust_probability > peak.bust_probability) peak = a;
    if (a.dominant_variable) {
      drivers.set(a.dominant_variable, (drivers.get(a.dominant_variable) ?? 0) + 1);
    }
  }

  const topDriver = [...drivers.entries()].sort((a, b) => b[1] - a[1])[0] ?? null;
  return { total: alerts.length, bust, watch, regions: regions.size, peak, topDriver };
}
