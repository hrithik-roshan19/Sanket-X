import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { BustProbabilityPoint, RiskBand } from "../../api/types";
import { CHART, bandLabel } from "../../theme";

const BAND_COLOR: Record<RiskBand, string> = {
  low: CHART.low,
  medium: CHART.medium,
  high: CHART.high,
};

export function BustProbabilityCurve({ points, cuts }: {
  points: BustProbabilityPoint[];
  cuts?: { medium: number; high: number };
}) {
  if (!points.length) return <p className="muted">No bust-probability curve for this region.</p>;
  const data = points.map((p) => ({
    lead: p.lead_time_days,
    probability: Number((p.bust_probability * 100).toFixed(2)),
    band: p.risk_band,
    driver: p.dominant_variable,
  }));

  const watch = cuts ? cuts.medium * 100 : null;
  const bust = cuts ? cuts.high * 100 : null;

  return (
    <>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 8, right: 10, bottom: 4, left: -8 }}>
          <defs>
            <linearGradient id="bustFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={CHART.high} stopOpacity={0.2} />
              <stop offset="100%" stopColor={CHART.high} stopOpacity={0} />
            </linearGradient>
          </defs>

          {watch != null && bust != null ? (
            <>
              <ReferenceArea y1={0} y2={watch} fill={CHART.low} fillOpacity={0.05} strokeOpacity={0} />
              <ReferenceArea y1={watch} y2={bust} fill={CHART.medium} fillOpacity={0.07} strokeOpacity={0} />
              <ReferenceArea y1={bust} y2={100} fill={CHART.high} fillOpacity={0.07} strokeOpacity={0} />
              <ReferenceLine y={watch} stroke={CHART.medium} strokeDasharray="3 3" strokeOpacity={0.6} />
              <ReferenceLine y={bust} stroke={CHART.high} strokeDasharray="3 3" strokeOpacity={0.6} />
            </>
          ) : null}

          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} stroke={CHART.axis} tickLine={false} />
          <YAxis
            domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} width={46}
            stroke={CHART.axis} tickLine={false}
          />
          <Tooltip
            formatter={(v: number, _n, item) =>
              [`${v}%  (${bandLabel(item.payload.band)})`, item.payload.driver ? `mostly driven by: ${item.payload.driver}` : "Bust risk"]
            }
            labelFormatter={(l) => `Lead day ${l}`}
          />
          <Area
            type="monotone" dataKey="probability" stroke="none" fill="url(#bustFill)"
            // tooltipType="none": Area and Line share a dataKey, so the tooltip doubles without it.
            isAnimationActive={false} legendType="none" tooltipType="none"
          />
          <Line
            type="monotone" dataKey="probability" stroke={CHART.observed} strokeWidth={2}
            dot={<BandDot />} activeDot={{ r: 5 }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {watch != null && bust != null ? (
        <p className="bandkey">
          <span><i style={{ background: CHART.low }} />low &lt; {watch.toFixed(0)}%</span>
          <span><i style={{ background: CHART.medium }} />watch {watch.toFixed(0)}–{bust.toFixed(0)}%</span>
          <span><i style={{ background: CHART.high }} />bust ≥ {bust.toFixed(0)}%</span>
          <em>band edges set from this model&rsquo;s own training run, not chosen by hand</em>
        </p>
      ) : null}
    </>
  );
}

function BandDot(props: { cx?: number; cy?: number; payload?: { band: RiskBand } }) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null || !payload) return null;
  return (
    <circle
      cx={cx} cy={cy} r={3.6}
      fill={BAND_COLOR[payload.band] ?? CHART.axis}
      stroke="#fff" strokeWidth={1.4}
    />
  );
}
