import { useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer,
  Scatter, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import type { EnsembleDivergenceResponse } from "../../api/types";
import { CHART } from "../../theme";

type Tab = "national" | "skill";

export function HeroDivergence({ data }: { data?: EnsembleDivergenceResponse }) {
  const [tab, setTab] = useState<Tab>("national");

  if (!data?.model_trained) return null;

  const mean = data.mean_bust_probability;
  const prior = data.prior_mean_bust_probability;
  const delta = mean != null && prior != null ? mean - prior : null;
  const apart = data.n_high_regions > 0;

  return (
    <section className="hero">
      <div className="hero__grid">
        <div className="hero__lead">
          <div className="hero__kick">
            <i aria-hidden="true" />
            <span>{data.eyebrow}</span>
          </div>

          <h1 className="hero__head">
            {apart ? (
              <>Where the forecast comes <em>apart.</em></>
            ) : (
              <>Where the forecast <em>holds.</em></>
            )}
          </h1>

          <p className="hero__sub">{data.national_note ?? data.headline_note}</p>

          <p className="hero__why">
            Averaged across every region scored in this forecast run — one run is all ten
            days issued at a single hour.
          </p>

          {mean != null ? (
            <div className="hero__gauge">
              <Ring value={mean} />
              <div className="hero__gaugemeta">
                <span className="hero__gaugelabel">Mean bust risk · 0% holds, 100% busts</span>
                <strong className="hero__gaugebig">
                  {data.n_high_regions} of {data.n_scored_regions} regions in the top band
                </strong>
                {delta != null ? (
                  <span className={deltaClass(delta)}>
                    {delta >= 0 ? "▲" : "▼"} {delta >= 0 ? "+" : ""}{(delta * 100).toFixed(1)} points vs run {data.prior_init_date}
                  </span>
                ) : (
                  <span className="hero__delta hero__delta--none">{data.prior_note}</span>
                )}
              </div>
            </div>
          ) : null}
        </div>

        <figure className="hero__panel">
          <figcaption className="hero__panelhead">
            <div className="hero__tabs" role="tablist" aria-label="Hero chart">
              <button
                type="button" role="tab" aria-selected={tab === "national"}
                className={tab === "national" ? "hero__tab is-active" : "hero__tab"}
                onClick={() => setTab("national")}
              >
                Risk by lead day
              </button>
              <button
                type="button" role="tab" aria-selected={tab === "skill"}
                className={tab === "skill" ? "hero__tab is-active" : "hero__tab"}
                onClick={() => setTab("skill")}
              >
                Were we right?
              </button>
            </div>
          </figcaption>

          <div className="hero__chart">
            {tab === "national" ? <NationalChart data={data} /> : <SkillChart data={data} />}
          </div>

          {tab === "national" ? (
            <div className="hero__key">
              <b><i style={{ background: CHART.forecast }} />Mean across regions</b>
              <b><i style={{ background: CHART.member }} />Calmest to worst region</b>
            </div>
          ) : (
            <div className="hero__key">
              <b><i style={{ background: CHART.forecast }} />What actually happened</b>
              <b><i style={{ background: CHART.axis }} />Where dots would sit if always right</b>
            </div>
          )}

          <p className="hero__source">
            {tab === "national"
              ? `${data.source ?? ""} · cycle ${data.init_date}`
              : skillCaption(data)}
          </p>
        </figure>
      </div>
    </section>
  );
}

function NationalChart({ data }: { data: EnsembleDivergenceResponse }) {
  const rows = data.national.map((n) => ({
    lead: n.lead_time_days,
    mean: Number((n.mean_bust_probability * 100).toFixed(1)),

    band: [n.min_bust_probability * 100, n.max_bust_probability * 100] as [number, number],
    high: n.n_high_regions,
    n: n.n_regions,
  }));
  if (!rows.length) return <Empty>No scored regions in this cycle.</Empty>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 10, right: 14, bottom: 6, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} vertical={false} />
        <XAxis dataKey="lead" tickFormatter={(d) => `D${d}`} stroke={CHART.axis} tickLine={false} />
        <YAxis
          domain={[0, 100]} width={46} stroke={CHART.axis} tickLine={false}
          tickFormatter={(v: number) => `${v}%`}
        />
        <Tooltip
          labelFormatter={(l) => `Lead day ${l}`}
          formatter={(v: number | number[], _n: string, item) => {
            if (Array.isArray(v)) {
              return [`${v[0].toFixed(0)}% – ${v[1].toFixed(0)}%`, "Calmest to worst region"];
            }
            return [
              `${v.toFixed(1)}%  ·  ${item.payload.high}/${item.payload.n} in the top band`,
              "Mean bust risk",
            ];
          }}
        />
        <Area
          dataKey="band" stroke="none" fill={CHART.member} fillOpacity={0.4}
          isAnimationActive={false} name="Calmest to worst region"
        />
        <Line
          type="monotone" dataKey="mean" name="Mean bust risk"
          stroke={CHART.forecast} strokeWidth={2.6} dot={{ r: 2.5 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function SkillChart({ data }: { data: EnsembleDivergenceResponse }) {
  const skill = data.skill;
  if (!skill) return <Empty>This run recorded no accuracy metrics.</Empty>;

  if (!skill.calibration.length) {
    return (
      <div className="hero__skillfallback">
        <SkillNumbers skill={skill} />
        {skill.note ? <p className="muted small">{skill.note}</p> : null}
      </div>
    );
  }

  const rows = skill.calibration.map((b) => ({
    predicted: Number((b.predicted_mean * 100).toFixed(1)),
    observed: Number((b.observed_rate * 100).toFixed(1)),
    n: b.n,
  }));

  return (
    <>
      <SkillNumbers skill={skill} />
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 6, right: 14, bottom: 6, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis
            type="number" dataKey="predicted" domain={[0, 100]} stroke={CHART.axis}
            tickLine={false} tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            type="number" dataKey="observed" domain={[0, 100]} width={46} stroke={CHART.axis}
            tickLine={false} tickFormatter={(v: number) => `${v}%`}
          />
          <ZAxis type="number" dataKey="n" range={[60, 400]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<CalibrationTip />} />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
            stroke={CHART.axis} strokeDasharray="5 5" strokeWidth={1.5}
          />
          <Scatter dataKey="observed" fill={CHART.forecast} fillOpacity={0.75} />
        </ComposedChart>
      </ResponsiveContainer>
    </>
  );
}

function SkillNumbers({ skill }: { skill: NonNullable<EnsembleDivergenceResponse["skill"]> }) {
  return (
    <dl className="hero__skill">
      <div><dt>Ranking (ROC-AUC)</dt><dd>{num(skill.roc_auc, 3)}</dd></div>
      <div><dt>Warnings right</dt><dd>{num(skill.precision, 2)}</dd></div>
      <div><dt>Busts caught</dt><dd>{num(skill.recall, 2)}</dd></div>
      <div><dt>Probability error</dt><dd>{num(skill.brier, 3)}</dd></div>
    </dl>
  );
}

function CalibrationTip({ active, payload }: {
  active?: boolean;
  payload?: { payload: { predicted: number; observed: number; n: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="hero__tip">
      <strong>Sanket said {p.predicted.toFixed(0)}%</strong>
      <span>it actually busted {p.observed.toFixed(0)}% of the time</span>
      <span className="hero__tipmeta">across {p.n.toLocaleString()} forecasts</span>
    </div>
  );
}

function Ring({ value }: { value: number }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value));
  const colour = pct >= 0.57 ? CHART.high : pct >= 0.36 ? CHART.medium : CHART.low;
  return (
    <div className="hero__ring">
      <svg viewBox="0 0 128 128" aria-hidden="true">
        <circle cx="64" cy="64" r={r} fill="none" stroke="#dfe4ec" strokeWidth="11" />
        <circle
          cx="64" cy="64" r={r} fill="none" stroke={colour} strokeWidth="11"
          strokeLinecap="round" strokeDasharray={`${c * pct} ${c}`}
          transform="rotate(-90 64 64)"
        />
      </svg>
      <span className="hero__ringvalue">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="muted small hero__empty">{children}</p>;
}

function skillCaption(data: EnsembleDivergenceResponse): string {
  const s = data.skill;
  if (!s?.n) return "No accuracy metrics recorded for this run";
  return `${s.n.toLocaleString()} forecasts on the held-out ${s.split} split — data this model never trained on`;
}

function deltaClass(delta: number): string {
  if (Math.abs(delta) < 0.005) return "hero__delta hero__delta--none";
  return delta > 0 ? "hero__delta hero__delta--up" : "hero__delta hero__delta--down";
}

function num(v: number | null | undefined, digits: number): string {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

