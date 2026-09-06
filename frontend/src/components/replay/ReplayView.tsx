import { useEffect, useMemo, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RegionSummary, ReplayRegionStep } from "../../api/types";
import { useModelStatus, useReplay, useReplayCycles } from "../../hooks/useDashboardData";
import { EmptyState, ErrorState, LoadingState } from "../common/States";
import { IndiaChoroplethMap } from "../map/IndiaChoroplethMap";
import { MapLegend } from "../map/MapLegend";
import { ReplayFocusChart } from "./ReplayFocusChart";
import { ReplayProbabilityChart } from "./ReplayProbabilityChart";

const STEP_MS = 2200;

export function ReplayView({ topology }: { topology: Topology | null }) {
  const [selectedInit, setSelectedInit] = useState<string | undefined>(undefined);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const [focusRegionId, setFocusRegionId] = useState<string | null>(null);

  const cyclesQuery = useReplayCycles(true);
  const statusQuery = useModelStatus();
  const replayQuery = useReplay(selectedInit, true);
  const replay = replayQuery.data;
  const steps = replay?.steps ?? [];

  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
    setFocusRegionId(null);
  }, [replay?.init_date]);

  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || steps.length === 0) return;
    timer.current = window.setInterval(() => {
      setStepIdx((i) => {
        if (i >= steps.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, STEP_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [playing, steps.length]);

  const step = steps[Math.min(stepIdx, Math.max(steps.length - 1, 0))];
  const mapRegions: RegionSummary[] = useMemo(
    () => (step?.regions ?? []).map(toRegionSummary),
    [step],
  );
  const focusOptions = replay?.focus_options ?? [];
  const shownFocus =
    focusOptions.find((o) => o.region_id === focusRegionId) ?? replay?.focus ?? null;
  const chartableRegionIds = new Set(focusOptions.map((o) => o.region_id));

  if (replayQuery.isLoading || cyclesQuery.isLoading) {
    return <LoadingState label="Scoring the historical cycle…" />;
  }
  if (replayQuery.error) return <ErrorState error={replayQuery.error} />;
  if (replay && !replay.model_trained) {
    return <EmptyState title="No trained model yet" message={replay.message} />;
  }
  if (!replay || !steps.length) {
    return <EmptyState title="Nothing to replay" message={replay?.message ?? "No scoreable cycle in the store."} />;
  }

  const cycles = cyclesQuery.data ?? replay.available_cycles;
  const scrub = (i: number) => {
    setPlaying(false);
    setStepIdx(i);
  };

  return (
    <div className="replay">
      <div className="replay__intro">
        <div className="replay__pick">
          <label htmlFor="replay-cycle" className="muted small">Forecast cycle</label>
          <select
            id="replay-cycle"
            value={selectedInit ?? replay.init_date ?? ""}
            onChange={(e) => setSelectedInit(e.target.value || undefined)}
          >
            {cycles.map((c) => (
              <option key={c.init_date} value={c.init_date}>
                {c.init_date}
                {c.verified
                  ? ` · outcome known for ${c.verified_lead_days} day${c.verified_lead_days === 1 ? "" : "s"}`
                  : " · outcome not yet known"}
                {c.peak_bust_probability != null
                  ? ` · peak risk ${(c.peak_bust_probability * 100).toFixed(0)}%`
                  : ""}
              </option>
            ))}
          </select>
        </div>
        {replay.summary_narration ? (
          <p className="replay__summary">{replay.summary_narration}</p>
        ) : null}
      </div>

      <div className="replay__controls">
        <button
          type="button"
          className="replay__play"
          onClick={() => {
            if (stepIdx >= steps.length - 1) setStepIdx(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button type="button" onClick={() => scrub(Math.max(0, stepIdx - 1))} disabled={stepIdx === 0}>
          ‹ Prev
        </button>
        <button
          type="button"
          onClick={() => scrub(Math.min(steps.length - 1, stepIdx + 1))}
          disabled={stepIdx >= steps.length - 1}
        >
          Next ›
        </button>
        <div className="replay__ticks">
          {steps.map((s, i) => (
            <button
              key={s.lead_time_days}
              type="button"
              className={`replay__tick ${i === stepIdx ? "is-active" : ""} replay__tick--${bandOfStep(s)}`}
              aria-label={`Lead day ${s.lead_time_days}`}
              onClick={() => scrub(i)}
            >
              {s.lead_time_days}
            </button>
          ))}
        </div>
      </div>

      <div className="replay__body">
        <div className="replay__mapcol">
          {replay.risk_band_definitions ? (
            <MapLegend definitions={replay.risk_band_definitions} />
          ) : null}
          <IndiaChoroplethMap
            regions={mapRegions}
            selectedRegionId={shownFocus?.region_id ?? null}
            onSelect={(rid) => chartableRegionIds.has(rid) && setFocusRegionId(rid)}
            topology={topology}
          />
          <p className="muted small">Click a state to chart its forecast against what actually happened.</p>
        </div>

        <div className="replay__sidecol">
          <div className="replay__step">
            <div className="replay__stephead">
              <span className="replay__day">Day {step.lead_time_days}</span>
              {step.valid_date ? <span className="muted small">valid {step.valid_date}</span> : null}
            </div>
            <p className="replay__narration">{step.narration}</p>
            <div className="replay__stats">
              <span><b>{pct(step.mean_bust_probability)}</b> average bust risk</span>
              <span className="chip chip--high">{step.n_high} bust</span>
              <span className="chip chip--medium">{step.n_medium} watch</span>
            </div>
            <ol className="replay__toplist">
              {step.regions.slice(0, 5).map((r) => {
                const chartable = chartableRegionIds.has(r.region_id);
                const active = shownFocus?.region_id === r.region_id;
                return (
                  <li key={r.region_id}>
                    <button
                      type="button"
                      className={`replay__toprow ${active ? "is-active" : ""}`}
                      disabled={!chartable}
                      title={chartable ? "Show this region's forecast vs observed" : undefined}
                      onClick={() => setFocusRegionId(r.region_id)}
                    >
                      <span className={`dot dot--${r.risk_band}`} />
                      {r.region_name ?? r.region_id}
                      <b>{pct(r.bust_probability)}</b>
                      {r.dominant_variable ? (
                        <span className="muted small">{r.dominant_variable.replace(/_/g, " ")}</span>
                      ) : null}
                    </button>
                  </li>
                );
              })}
            </ol>
          </div>

          {shownFocus ? (
            <>
              <ReplayFocusChart focus={shownFocus} currentLead={step.lead_time_days} />
              <ReplayProbabilityChart
                currentLead={step.lead_time_days}
                cuts={statusQuery.data?.thresholds?.risk_band_cuts}
                variable={shownFocus.variable}
                points={(replayQuery.data?.steps ?? []).map((st) => {

                  const fp = shownFocus.points.find(
                    (x) => x.lead_time_days === st.lead_time_days);
                  const thr = shownFocus.bust_threshold;
                  const busted =
                    fp && thr != null && fp.predicted_value != null && fp.observed_value != null
                      ? Math.abs(fp.predicted_value - fp.observed_value) >= thr
                      : null;
                  return {
                    lead: st.lead_time_days,
                    p: st.regions.find((r) => r.region_id === shownFocus.region_id)
                         ?.bust_probability ?? null,
                    busted,
                  };
                })}
              />
            </>
          ) : (

            <div className="replay-focus">
              <div className="replay-focus__head">
                <strong>No forecast-vs-observed chart for this run</strong>
              </div>
              <p className="muted small">
                This cycle is recent enough that the days it forecasts have not all
                happened yet, so there is nothing to compare its forecast against. The map,
                the risk numbers and the narration above are all scored and real — only the
                verification chart needs observations that do not exist yet.
              </p>
              <p className="muted small">
                Pick a cycle marked <b>outcome known</b> in the dropdown to see the model
                checked against what actually happened.
              </p>
            </div>
          )}
        </div>
      </div>

      <p className="muted small replay__foot">
        Every value is scored from the real archived GEFS forecast for this run and the
        ERA5 record of what happened; every sentence is generated from those numbers. No
        scripted or synthetic content.
      </p>
    </div>
  );
}

function toRegionSummary(r: ReplayRegionStep): RegionSummary {
  return {
    region_id: r.region_id,
    region_name: r.region_name,
    bust_probability: r.bust_probability,
    risk_band: r.risk_band,
    confidence: r.confidence,
    dominant_variable: r.dominant_variable,
    data_available: true,
  };
}

function bandOfStep(s: { n_high: number; n_medium: number }): string {
  if (s.n_high >= 3) return "high";
  if (s.n_high + s.n_medium >= 3) return "medium";
  return "low";
}

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(0)}%`;
}
