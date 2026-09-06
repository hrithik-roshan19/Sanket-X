import { useMutation, useQueryClient } from "@tanstack/react-query";
import { triggerCycleRun } from "../../api/ingest";
import { useIngestStatus } from "../../hooks/useDashboardData";

export function FeedFreshness() {
  const { data, error } = useIngestStatus();
  const qc = useQueryClient();

  const refresh = useMutation({
    mutationFn: triggerCycleRun,
    onSuccess: () => {

      setTimeout(() => qc.invalidateQueries({ queryKey: ["ingestStatus"] }), 2000);
    },
  });

  if (error || !data) return null;

  const last = data.last_forecast;
  const failed = last?.status === "failed";

  return (
    <div className="feed-strip">
      <div className="feed-strip__main">
        <span className={`feed-dot feed-dot--${feedState(data.scheduler.running, failed)}`} />
        <span className="feed-strip__text">
          {data.last_cycle_ingested ? (
            <>
              Forecast cycle{" "}
              <b title="Forecast runs are issued at fixed hours in UTC (00, 06, 12, 18Z) and are named by them worldwide, so this label is not converted to local time.">
                {data.last_cycle_ingested}Z
              </b>
              {last?.finished_at ? <> · ingested {relative(last.finished_at)}</> : null}
            </>
          ) : (
            <>No live cycle ingested yet — showing whatever is in the store</>
          )}
        </span>
      </div>

      <div className="feed-strip__meta">
        {data.scheduler.running ? (
          <span className="muted small">
            auto-pull every {Math.round(data.scheduler.tick_seconds / 60)} min
            {data.scheduler.next_tick ? ` · next ${clock(data.scheduler.next_tick)} IST` : ""}
          </span>
        ) : (
          <span className="muted small">auto-pull off</span>
        )}
        {data.enabled ? (
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => refresh.mutate()}
            disabled={refresh.isPending || last?.status === "running"}
          >
            {refresh.isPending || last?.status === "running" ? "Pulling…" : "Refresh now"}
          </button>
        ) : null}
      </div>

      {failed && last?.error ? (
        <p className="warn small feed-strip__error">Last pull failed: {last.error}</p>
      ) : null}
      {refresh.isSuccess ? (
        <p className="muted small feed-strip__error">
          Pull started — a cycle takes a couple of minutes to download and ingest.
        </p>
      ) : null}
    </div>
  );
}

function feedState(running: boolean, failed: boolean): string {
  if (failed) return "bad";
  return running ? "live" : "idle";
}

function relative(iso: string): string {
  const then = new Date(iso.endsWith("Z") ? iso : `${iso}Z`).getTime();
  const mins = Math.round((Date.now() - then) / 60000);
  if (!Number.isFinite(mins)) return "recently";
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} h ago`;
  return `${Math.round(hrs / 24)} d ago`;
}

function clock(iso: string): string {
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleTimeString("en-GB", {
        timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false,
      });
}
