import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { CHART } from "../../theme";

export function ReplayProbabilityChart({
  points,
  currentLead,
  cuts,
  variable,
}: {
  points: { lead: number; p: number | null; busted: boolean | null }[];
  currentLead: number;
  cuts?: { medium?: number; high?: number };
  variable?: string;
}) {
  const any = points.some((d) => d.p != null);
  if (!any) return null;
  const pct = (v: number) => `${Math.round(v * 100)}%`;

  return (
    <div className="replay-focus">
      <div className="replay-focus__head">
        <strong>Predicted bust risk</strong>
        <span className="muted small">what the model said, before the outcome</span>
      </div>
      <ResponsiveContainer width="100%" height={132}>
        <LineChart data={points} margin={{ top: 8, right: 40, bottom: 4, left: -6 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} stroke={CHART.axis} />
          <YAxis width={58} domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
                 tickFormatter={pct} stroke={CHART.axis} />
          <Tooltip
            labelFormatter={(l) => `Lead day ${l}`}
            formatter={(v: number) => (v == null ? "—" : pct(v))}
          />
          {typeof cuts?.medium === "number" ? (
            <ReferenceLine y={cuts.medium} stroke={CHART.medium} strokeDasharray="4 4"
              label={{ value: "watch", position: "right", fontSize: 9, fill: CHART.axis, dy: -4, dx: -34 }} />
          ) : null}
          {typeof cuts?.high === "number" ? (
            <ReferenceLine y={cuts.high} stroke={CHART.high} strokeDasharray="4 4"
              label={{ value: "bust", position: "right", fontSize: 9, fill: CHART.axis, dy: -4, dx: -30 }} />
          ) : null}
          <ReferenceLine x={currentLead} stroke={CHART.marker} strokeWidth={2} />
          <Line type="monotone" dataKey="p" name="Bust risk" stroke={CHART.forecast}
            strokeWidth={2} activeDot={{ r: 5 }} connectNulls={false}
            dot={(props: any) => {
              const { cx, cy, payload, key } = props;
              if (cx == null || cy == null) return <g key={key} />;
              const hit = payload?.busted === true;
              return (
                <circle key={key} cx={cx} cy={cy} r={hit ? 4.5 : 3}
                  fill={hit ? CHART.high : "#fff"}
                  stroke={hit ? CHART.high : CHART.forecast} strokeWidth={2} />
              );
            }} />
        </LineChart>
      </ResponsiveContainer>
      <p className="muted small">
        <span style={{ color: CHART.high }}>●</span> observed bust
        {variable ? <> — {variable.replace(/_/g, " ")} error exceeded its threshold</> : null}
        {"  ·  "}
        <span style={{ color: CHART.forecast }}>○</span> threshold not exceeded for this
        variable
      </p>
      <p className="muted small">
        The line is scored by the deployed model from this cycle&apos;s forecast alone — no
        observation is used to produce it. The markers are the outcome, added afterwards.
      </p>
    </div>
  );
}
