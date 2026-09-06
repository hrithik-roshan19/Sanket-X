import { useEffect, useRef, useState } from "react";
import type { Topology } from "topojson-specification";
import type { RiskBand } from "../api/types";
import { AlertsPage } from "../components/alerts/AlertsPage";
import { BustSummaryChart } from "../components/dashboard/BustSummaryChart";
import { FeedFreshness } from "../components/dashboard/FeedFreshness";
import { HeroDivergence } from "../components/dashboard/HeroDivergence";
import { KpiStrip } from "../components/dashboard/KpiStrip";
import { ModelPage } from "../components/model/ModelPage";
import { RiskTicker } from "../components/dashboard/RiskTicker";
import { RegionDetailPanel } from "../components/detail/RegionDetailPanel";
import { EmptyState, ErrorState, LoadingState } from "../components/common/States";
import { IndiaChoroplethMap, loadTopology } from "../components/map/IndiaChoroplethMap";
import { LeadDayRail } from "../components/map/LeadDayRail";
import { LeadDaySelector } from "../components/map/LeadDaySelector";
import { MapLegend } from "../components/map/MapLegend";
import { AboutPage } from "../components/about/AboutPage";
import { ReplayView } from "../components/replay/ReplayView";
import { useAllRegions, useEnsembleDivergence, useModelStatus } from "../hooks/useDashboardData";
import { useLiveSocket } from "../hooks/useLiveSocket";
import { useMediaQuery } from "../hooks/useMediaQuery";
import { useLiveStore } from "../store/liveStore";

type View = "live" | "alerts" | "model" | "replay" | "about";

const TABS: { id: View; label: string; tail?: string }[] = [
  { id: "live", label: "Operations" },
  { id: "alerts", label: "Alerts" },
  { id: "model", label: "Model" },
  { id: "replay", label: "Replay", tail: " a real bust" },

  { id: "about", label: "About" },
];

export function DashboardPage() {
  useLiveSocket();

  const [view, setView] = useState<View>("live");
  const [leadDay, setLeadDay] = useState(1);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [alertFilter, setAlertFilter] = useState<RiskBand | undefined>(undefined);
  const [topology, setTopology] = useState<Topology | null>(null);
  const connectionStatus = useLiveStore((s) => s.connectionStatus);
  const [topoError, setTopoError] = useState<unknown>(null);

  const regionsQuery = useAllRegions();
  const statusQuery = useModelStatus();

  const ensembleQuery = useEnsembleDivergence();

  useEffect(() => {
    loadTopology().then(setTopology).catch(setTopoError);
  }, []);

  const allRegions = regionsQuery.data;
  const regions =
    allRegions?.days.find((d) => d.lead_time_days === leadDay) ?? allRegions?.days[0];
  const riskCuts = statusQuery.data?.thresholds?.risk_band_cuts;

  const autoLeadPicked = useRef(false);
  const available = regions?.available_lead_days;
  useEffect(() => {
    if (autoLeadPicked.current || !available?.length) return;
    if (!available.includes(leadDay)) setLeadDay(available[0]);
    autoLeadPicked.current = true;
  }, [available, leadDay]);

  const highCount = regions?.regions.filter((r) => r.risk_band === "high").length ?? 0;

  const heroFills = Boolean(ensembleQuery.data?.model_trained);

  const showRail = useMediaQuery("(min-width: 1440px) and (min-height: 700px)");

  const selectAndReveal = (regionId: string) => {
    setSelectedRegion(regionId);
    scrollToOperations();
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__inner">
          <button
            type="button"
            className="brand"
            aria-label="Sanket - back to the opening screen"
            onClick={() => {
              setView("live");
              window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
            }}
          >
            <img className="brand__glyph" src="/logo.png" alt="" width={40} height={32} />
            <img className="brand__wordmark" src="/wordmark.png" alt="Sanket" />
            <span className="brand__div" aria-hidden="true" />
            <span className="brand__sub">Forecast Bust Detection</span>
          </button>

          <nav className="viewtabs" role="tablist" aria-label="View">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={view === t.id}
                className={view === t.id ? "viewtab is-active" : "viewtab"}
                onClick={() => setView(t.id)}
              >
                {t.label}
                {t.tail ? <span className="viewtab__tail">{t.tail}</span> : null}
              </button>
            ))}
          </nav>

          <div className="topbar__right">
            <span className="pill pill--quiet" title="Real-time event connection; polling remains the safety fallback">
              <i aria-hidden="true" />
              Live {connectionStatus === "open" ? "connected" : connectionStatus === "connecting" ? "connecting" : "polling fallback"}
            </span>
            {view === "live" && regions?.model_trained && regions.init_date ? (
              <span className="pill pill--quiet" title="The forecast cycle currently in the store">
                <i aria-hidden="true" />
                {regions.init_date}
                {regions.valid_date ? ` → ${regions.valid_date}` : ""}
              </span>
            ) : null}
            {view === "live" && highCount > 0 ? (
              <span className="pill pill--alarm" title={`Regions in the bust band at lead day ${leadDay}`}>
                <i aria-hidden="true" />
                {highCount} bust · D{leadDay}
              </span>
            ) : null}
          </div>
        </div>
      </header>

      {view === "about" ? (
        <AboutPage onReplay={() => setView("replay")} />
      ) : view === "replay" ? (
        <main className="app__body app__body--replay">
          <ReplayView topology={topology} />
        </main>
      ) : view === "alerts" ? (
        <AlertsPage
          filter={alertFilter}
          onFilter={setAlertFilter}

          onSelect={(regionId, lead) => {
            setSelectedRegion(regionId);
            setLeadDay(lead);
            setView("live");

            requestAnimationFrame(scrollToOperations);
          }}
        />
      ) : view === "model" ? (
        <ModelPage />
      ) : (
        <>
          <section className={heroFills ? "screen1" : undefined}>
            <HeroDivergence data={ensembleQuery.data} />
            <KpiStrip all={allRegions} day={regions} />
            {heroFills ? <OpeningCues onReplay={() => setView("replay")} /> : null}
            {regions?.regions.length ? (
              <RiskTicker
                regions={regions.regions}
                leadDay={regions.lead_time_days}
                onSelect={selectAndReveal}
              />
            ) : null}
          </section>

          <FeedFreshness />

          <main
            className={showRail ? "app__body app__body--ops app__body--rail" : "app__body app__body--ops"}
            id="operations"
          >
            <section className="map-column">
              <div className="map-toolbar">
                {showRail ? null : (
                  <LeadDaySelector
                    value={leadDay}
                    onChange={setLeadDay}
                    disabled={!regions?.model_trained}
                    available={regions?.available_lead_days}
                  />
                )}
                {regions ? <MapLegend definitions={regions.risk_band_definitions} /> : null}
              </div>

              {regionsQuery.isLoading ? <LoadingState label="Scoring the current cycle…" /> : null}
              {regionsQuery.error ? <ErrorState error={regionsQuery.error} /> : null}
              {topoError ? <ErrorState error={topoError} /> : null}

              {regions && !regions.model_trained ? (
                <EmptyState
                  title="No trained model yet"
                  message={regions.message}
                  action={<p className="muted small">Upload a dataset with forecasts and matching observations to begin.</p>}
                />
              ) : null}

              {regions?.model_trained ? (
                regions.regions.length ? (
                  <IndiaChoroplethMap
                    regions={regions.regions}
                    selectedRegionId={selectedRegion}
                    onSelect={setSelectedRegion}
                    topology={topology}
                  />
                ) : (
                  <EmptyState title="Nothing to show for this lead day" message={regions.message} />
                )
              ) : null}
            </section>

            {showRail ? (
              <LeadDayRail all={allRegions} value={leadDay} onChange={setLeadDay} />
            ) : null}

            <RegionDetailPanel
              regionId={selectedRegion}
              onClose={() => setSelectedRegion(null)}
              riskCuts={riskCuts}
            />
          </main>

          {regions?.regions.length ? (
            <section className="app__below">
              <BustSummaryChart regions={regions.regions} onSelect={setSelectedRegion} />
            </section>
          ) : null}
        </>
      )}
    </div>
  );
}

function scrollToOperations() {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document
    .getElementById("operations")
    ?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
}

function OpeningCues({ onReplay }: { onReplay: () => void }) {
  return (
    <div className="cuerow">
      <button type="button" className="cuerow__primary" onClick={onReplay}>
        Watch it call a real bust →
      </button>
      <ScrollCue />
    </div>
  );
}

function ScrollCue() {
  return (
    <button type="button" className="scrollcue" onClick={scrollToOperations}>
      <span>The national map</span>
      <svg viewBox="0 0 16 10" aria-hidden="true" width="16" height="10">
        <path d="M1 1l7 7 7-7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    </button>
  );
}

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
