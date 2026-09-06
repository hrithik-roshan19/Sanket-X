# Sanket — dashboard

React + Vite + TypeScript. Talks only to the FastAPI backend; it holds no data of its own
and fabricates nothing.

```bash
# macOS / Linux
npm install
cp .env.example .env          # VITE_API_BASE_URL, VITE_WS_URL
npm run dev                   # http://localhost:5173  (backend must be on :8000)
```

```powershell
# Windows (PowerShell)
npm install
Copy-Item .env.example .env
npm run dev
```

## Layout

```
src/
  api/            client.ts (fetch + typed errors), types.ts (mirrors the backend schemas),
                  regions.ts / alerts.ts / modelStatus.ts / upload.ts,
                  regionCodes.ts  ← GENERATED, do not hand-edit
  hooks/          useDashboardData.ts (TanStack Query), useLiveSocket.ts (WS → cache invalidation)
  store/          liveStore.ts (Zustand: connection status, last event, training flag)
  components/
    map/          IndiaChoroplethMap, LeadDaySelector, MapLegend
    detail/       RegionDetailPanel, BustProbabilityCurve, VariableTrajectoryChart, ShapFactorsList
    dashboard/    AlertsPanel, BustSummaryChart, ModelStatusCard
    upload/       UploadPanel, ColumnMappingConfirmModal
    common/       States.tsx (Empty / Loading / Error / badges)
  pages/          DashboardPage.tsx
```

## Region ↔ map join

The API keys regions by ISO 3166-2:IN `region_id` (`IN-MH`); the vendored topojson keys
features by the 2011 census `st_code` (`27`). `src/api/regionCodes.ts` is the only place
those meet, and it is **generated** from the backend's single source of truth:

```bash
python backend/scripts/gen_frontend_region_codes.py
```

Re-run it if `backend/app/utils/india_state_codes.py` ever changes, rather than editing
the TypeScript by hand.

## Live updates

`useLiveSocket` opens `/ws` and, on each event, invalidates the affected TanStack Query
keys so the data is refetched over REST. WS payloads are never merged into UI state — that
keeps the socket and REST shapes independent. The socket reconnects with capped
exponential backoff, so a stopped backend does not spin the browser.

## The honesty rules, in UI terms

- A region the API did not return is drawn in the explicit **“No data”** grey, never in a
  risk colour — an absent prediction can't be misread as low risk.
- With no trained model the map renders **no coloured regions at all**; the map, alerts and
  model cards each show an empty state carrying the backend's own explanation.
- A variable with no trained model shows “too few paired rows”, not an empty chart.
- The factors list states whether the numbers are SHAP or the `feature_importances_`
  fallback.
- Analog cases say they're not implemented instead of showing invented neighbours.
- Model metrics always name the split (`test` / `val` / `train`) they came from.
- Observed values are only drawn where the forecast has actually verified; unverified lead
  days leave a gap.

## Styling

`src/styles.css` is deliberately plain — Phase 4b is the wiring pass. The design pass
(palette, typography, layout) is Phase 5.
