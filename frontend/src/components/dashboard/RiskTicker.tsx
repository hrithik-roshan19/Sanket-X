import { useMemo, useState } from "react";
import type { RegionSummary } from "../../api/types";

export function RiskTicker({ regions, leadDay, onSelect }: {
  regions: RegionSummary[];
  leadDay: number;
  onSelect: (regionId: string) => void;
}) {
  const [paused, setPaused] = useState(false);

  const items = useMemo(
    () =>
      regions
        .filter((r) => r.bust_probability != null)
        .sort((a, b) => (b.bust_probability as number) - (a.bust_probability as number)),
    [regions],
  );

  if (items.length < 2) return null;

  const half = (hidden: boolean) => (
    <div className="ticker__half" aria-hidden={hidden || undefined}>
      {items.map((r) => (
        <button
          key={r.region_id}
          type="button"
          className="ticker__item"
          onClick={() => onSelect(r.region_id)}
          tabIndex={hidden ? -1 : 0}
        >
          <i className={`ticker__dot ticker__dot--${r.risk_band ?? "none"}`} aria-hidden="true" />
          <span className="ticker__name">{r.region_name ?? r.region_id}</span>
          <span className="ticker__value">{((r.bust_probability as number) * 100).toFixed(0)}%</span>
          <span className="ticker__unit">bust risk · day {leadDay}</span>
        </button>
      ))}
    </div>
  );

  return (
    <div
      className="ticker"
      role="region"
      aria-label={`Bust risk by region, lead day ${leadDay}`}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}

      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className="ticker__track" style={{ animationPlayState: paused ? "paused" : "running" }}>
        {half(false)}
        {half(true)}
      </div>
    </div>
  );
}
