import { geoMercator, geoPath } from "d3-geo";
import { useMemo, useState } from "react";
import { feature } from "topojson-client";
import type { FeatureCollection, Geometry } from "geojson";
import type { Topology } from "topojson-specification";
import topoData from "../../assets/geo/india_states.topojson?url";
import { REGION_ID_BY_ST_CODE } from "../../api/regionCodes";
import type { RegionSummary } from "../../api/types";
import { bandLabel } from "../../theme";

const WIDTH = 620;
const HEIGHT = 680;

interface StateProps {
  st_nm: string;
  st_code: string;
}

export function IndiaChoroplethMap({
  regions,
  selectedRegionId,
  onSelect,
  topology,
}: {
  regions: RegionSummary[];
  selectedRegionId: string | null;
  onSelect: (regionId: string) => void;
  topology: Topology | null;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; region: RegionSummary | null; name: string } | null>(null);

  const byRegionId = useMemo(() => {
    const m = new Map<string, RegionSummary>();
    regions.forEach((r) => m.set(r.region_id, r));
    return m;
  }, [regions]);

  const { features, pathFor } = useMemo(() => {
    if (!topology) return { features: [], pathFor: () => "" };
    const fc = feature(topology, topology.objects.states) as unknown as FeatureCollection<Geometry, StateProps>;
    const projection = geoMercator().fitSize([WIDTH, HEIGHT], fc);
    const path = geoPath(projection);
    return { features: fc.features, pathFor: (g: Geometry) => path(g) ?? "" };
  }, [topology]);

  if (!topology) return <div className="state state--loading">Loading map…</div>;

  return (
    <div className="map-wrap">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="map" role="img" aria-label="India forecast bust risk by state">
        <g>
          {features.map((f, i) => {
            const stCode = String(f.properties.st_code).padStart(2, "0");
            const regionId = REGION_ID_BY_ST_CODE[stCode];
            const region = regionId ? byRegionId.get(regionId) : undefined;
            const band = region?.risk_band ?? null;
            const selected = regionId != null && regionId === selectedRegionId;
            return (
              <path
                key={`${stCode}-${i}`}
                d={pathFor(f.geometry)}
                className={[
                  "region",
                  band ? `region--${band}` : "region--nodata",
                  selected ? "region--selected" : "",
                ].join(" ")}
                tabIndex={region ? 0 : -1}
                role={region ? "button" : undefined}
                aria-label={`${f.properties.st_nm}${band ? `, ${bandLabel(band)} risk` : ", no data"}`}
                onClick={() => regionId && onSelect(regionId)}
                onKeyDown={(e) => {
                  if (regionId && (e.key === "Enter" || e.key === " ")) {
                    e.preventDefault();
                    onSelect(regionId);
                  }
                }}
                onMouseMove={(e) => {
                  const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                  setHover({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top,
                    region: region ?? null,
                    name: f.properties.st_nm,
                  });
                }}
                onMouseLeave={() => setHover(null)}
              />
            );
          })}
        </g>
      </svg>

      {hover ? (
        <div className="map-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <strong>{hover.region?.region_name ?? hover.name}</strong>
          {hover.region ? (
            <>
              <div>
                Bust probability:{" "}
                <b>{((hover.region.bust_probability ?? 0) * 100).toFixed(1)}%</b> ({bandLabel(hover.region.risk_band)})
              </div>
              {hover.region.dominant_variable ? <div>Driver: {hover.region.dominant_variable}</div> : null}
              {hover.region.confidence != null ? (
                <div>Mean confidence: {(hover.region.confidence * 100).toFixed(0)}%</div>
              ) : null}
            </>
          ) : (
            <div className="muted">No forecast data for this region</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

export async function loadTopology(): Promise<Topology> {
  const res = await fetch(topoData);
  if (!res.ok) throw new Error(`could not load India topojson (${res.status})`);
  return (await res.json()) as Topology;
}
