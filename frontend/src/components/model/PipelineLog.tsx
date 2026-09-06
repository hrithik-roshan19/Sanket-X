import type { IngestRunRow } from "../../api/types";
import { stampShort } from "../../format";
import { useIngestRuns } from "../../hooks/useDashboardData";

const LABEL: Record<string, string> = {
  forecast: "GEFS cycle",
  observations_provisional: "observations (provisional)",
  observations_final: "observations (final)",
};

const BAND: Record<string, string> = {
  complete: "low",
  skipped: "medium",
  failed: "high",
  running: "medium",
};

function when(iso: string | null): string {
  return stampShort(iso);
}

export function PipelineLog() {
  const { data, error, isLoading } = useIngestRuns(25);
  if (isLoading) return null;

  if (error || !data?.runs?.length) return null;

  const runs: IngestRunRow[] = data.runs;
  const refused = runs.filter((r) => r.status === "failed" || r.status === "skipped").length;

  return (
    <section className="card">
      <header className="card__head"><h3>Pipeline activity</h3></header>
      <p className="muted small">
        The last {runs.length} attempts by the ingestion pipeline, newest first — each one a
        record written when it ran. <b>Refused and skipped runs are shown too.</b> A cycle
        delivered too incompletely to trust is rejected rather than published, because a
        short rainfall total would understate the quantity that drives most busts. A
        refusal here is the guard working.
        {refused > 0 ? <> {refused} of these {runs.length} did not ingest.</> : null}
      </p>
      <div className="tablewrap">
        <table className="dtable">
          <thead>
            <tr>
              <th>started (IST)</th><th>what</th><th>target</th><th>status</th>
              <th>rows</th><th>took</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td className="mono small">{when(r.started_at)}</td>
                <td>{LABEL[r.kind] ?? r.kind}</td>
                <td className="mono small">{r.target ?? "—"}</td>
                <td>
                  <span className={`dot dot--${BAND[r.status] ?? "medium"}`} />{" "}
                  {r.status}
                  {r.trigger && r.trigger !== "schedule" ? (
                    <span className="muted small"> · {r.trigger}</span>
                  ) : null}
                </td>
                <td className="mono">{r.rows_ingested ? r.rows_ingested.toLocaleString() : "—"}</td>
                <td className="mono small">{r.seconds != null ? `${r.seconds}s` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {runs[0]?.detail ? (
        <p className="muted small mono" style={{ wordBreak: "break-word" }}>
          latest: {runs[0].detail}
        </p>
      ) : null}
      {runs.find((r) => r.error) ? (
        <p className="muted small">
          Most recent error: {runs.find((r) => r.error)?.error?.slice(0, 240)}
        </p>
      ) : null}
    </section>
  );
}
