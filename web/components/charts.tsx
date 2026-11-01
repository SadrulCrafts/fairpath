"use client";
import * as React from "react";
import {
  Bar, BarChart, CartesianGrid, Label, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis, ZAxis, LabelList,
} from "recharts";
import { pct } from "@/lib/format";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

type TipRow = { name: string; value: string; color?: string };
export function TipBox({ title, rows }: { title?: string; rows: TipRow[] }) {
  return (
    <div className="rounded-md border bg-popover px-3 py-2 text-[12px] shadow-lg">
      {title && <div className="mb-1 font-medium">{title}</div>}
      {rows.map((r) => (
        <div key={r.name} className="flex items-center justify-between gap-4 tnum">
          <span className="flex items-center gap-1.5 text-muted-foreground">
            {r.color && <span className="h-2 w-2 rounded-sm" style={{ background: r.color }} />}{r.name}
          </span>
          <span className="font-medium">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

const axisProps = { tickLine: false, axisLine: false, fontSize: 11 } as const;

/** Grouped bars: one cluster per group, one bar per series (e.g. baseline vs deployed). */
export function GroupedBars({ data, series, height = 240, format = pct, domain = [0, 1] as [number, number], colors = SERIES }: {
  data: Record<string, string | number | null>[]; series: { key: string; label: string }[]; height?: number;
  format?: (v: number) => string; domain?: [number, number]; colors?: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} barGap={2} barCategoryGap="28%" margin={{ top: 18, right: 8, left: -8, bottom: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="label" {...axisProps} />
        <YAxis domain={domain} tickFormatter={(v) => format(v)} {...axisProps} width={44} />
        <Tooltip
          cursor={{ fill: "var(--grid)", opacity: 0.4 }}
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <TipBox title={String(label)} rows={payload.map((p, i) => ({ name: String(p.name), value: format(Number(p.value)), color: colors[i] }))} />
            ) : null
          }
        />
        {series.length > 1 && <Legend iconType="square" iconSize={9} wrapperStyle={{ fontSize: 12, paddingTop: 4 }} />}
        {series.map((s, i) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={colors[i]} radius={[4, 4, 0, 0]} maxBarSize={44}>
            <LabelList dataKey={s.key} position="top" formatter={(v: number) => (v == null ? "" : format(v))} style={{ fontSize: 10.5, fill: "var(--ink-2)" }} />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Dot-and-whisker: point estimate with a 95% CI for each row, two series side by side. */
export function ForestPlot({ rows, series, max, min = 0, colors = SERIES, digits = 2 }: {
  rows: { label: string; values: ({ value: number | null; lo: number | null; hi: number | null } | null)[] }[];
  series: string[]; max?: number; min?: number; colors?: string[]; digits?: number;
}) {
  const hi = max ?? Math.max(0.05, ...rows.flatMap((r) => r.values.map((v) => v?.hi ?? v?.value ?? 0))) * 1.1;
  const span = hi - min;
  const x = (v: number) => `${((Math.min(Math.max(v, min), hi) - min) / span) * 100}%`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => min + f * span);
  return (
    <div className="space-y-1">
      <div className="ml-[38%] flex justify-between pb-1 text-[10.5px] text-muted-foreground tnum">
        {ticks.map((t) => <span key={t}>{t.toFixed(digits)}</span>)}
      </div>
      {rows.map((r) => (
        <div key={r.label} className="grid grid-cols-[38%_1fr] items-center gap-0 border-t py-2 text-[12.5px]">
          <div className="pr-3">{r.label}</div>
          <div className="relative space-y-1.5">
            <div className="pointer-events-none absolute inset-0 flex justify-between">
              {ticks.map((t) => <span key={t} className="h-full w-px bg-[var(--grid)]" />)}
            </div>
            {r.values.map((v, i) => v && v.value != null && (
              <div key={i} className="relative h-3" title={`${series[i]}: ${v.value.toFixed(3)} (95% CI ${v.lo?.toFixed(3)} to ${v.hi?.toFixed(3)})`}>
                {v.lo != null && v.hi != null && (
                  <div className="absolute top-1/2 h-[2px] -translate-y-1/2 rounded" style={{ left: x(v.lo), width: `calc(${x(v.hi)} - ${x(v.lo)})`, background: colors[i], opacity: 0.55 }} />
                )}
                <div className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-card" style={{ left: x(v.value), background: colors[i] }} />
              </div>
            ))}
          </div>
        </div>
      ))}
      <div className="flex gap-4 pt-2 text-[11.5px] text-muted-foreground">
        {series.map((s, i) => (
          <span key={s} className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: colors[i] }} />{s}</span>
        ))}
        <span>Lines show 95% bootstrap intervals</span>
      </div>
    </div>
  );
}

export interface ParetoPoint { method: string; label: string; n: number; x: number; y: number; pareto: boolean; needsSensitive: boolean; isBaseline: boolean }
export const PARETO_COLORS = { baseline: "var(--series-2)", blind: "var(--series-1)", group: "var(--series-3)" };
export const paretoColor = (p: { isBaseline: boolean; needsSensitive: boolean }) =>
  p.isBaseline ? PARETO_COLORS.baseline : p.needsSensitive ? PARETO_COLORS.group : PARETO_COLORS.blind;

/** Numbered points (the number matches the table next to the chart), so labels can never collide. */
export function ParetoChart({ points, xLabel, height = 320 }: { points: ParetoPoint[]; xLabel: string; height?: number }) {
  const frontier = points.filter((p) => p.pareto).sort((a, b) => a.x - b.x);
  const Dot = (props: { cx?: number; cy?: number; payload?: ParetoPoint }) => {
    const { cx = 0, cy = 0, payload } = props;
    if (!payload) return <g />;
    return (
      <g>
        <circle cx={cx} cy={cy} r={10} fill={paretoColor(payload)} stroke="var(--surface-1)" strokeWidth={2} />
        <text x={cx} y={cy} dy={3.5} textAnchor="middle" fontSize={10} fontWeight={600} fill="#fff" style={{ pointerEvents: "none" }}>{payload.n}</text>
      </g>
    );
  };
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 16, right: 24, left: 0, bottom: 18 }}>
          <CartesianGrid />
          <XAxis type="number" dataKey="x" name={xLabel} {...axisProps} tickFormatter={(v) => v.toFixed(2)} domain={["auto", "auto"]} padding={{ left: 16, right: 16 }}>
            <Label value={`${xLabel} (lower is fairer) →`} position="insideBottom" offset={-12} style={{ fontSize: 11, fill: "var(--ink-muted)" }} />
          </XAxis>
          <YAxis type="number" dataKey="y" name="F1" {...axisProps} tickFormatter={(v) => v.toFixed(2)} domain={["auto", "auto"]} width={48} padding={{ top: 12, bottom: 12 }}>
            <Label value="F1 ↑" angle={-90} position="insideLeft" offset={14} style={{ fontSize: 11, fill: "var(--ink-muted)" }} />
          </YAxis>
          <ZAxis range={[300, 300]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              const p = payload?.[0]?.payload as ParetoPoint | undefined;
              return active && p ? (
                <TipBox title={`${p.n}. ${p.label}`} rows={[{ name: xLabel, value: p.x.toFixed(3) }, { name: "F1", value: p.y.toFixed(3) }, { name: "On the frontier", value: p.pareto ? "yes" : "no" }]} />
              ) : null;
            }}
          />
          <Scatter data={frontier} line={{ stroke: "var(--ink-muted)", strokeDasharray: "4 4", strokeWidth: 1.4 }} shape={() => <g />} isAnimationActive={false} />
          <Scatter data={points} shape={<Dot />} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-x-4 gap-y-1 pl-12 text-[11.5px] text-muted-foreground">
        {[["Baseline", PARETO_COLORS.baseline], ["Score never depends on group", PARETO_COLORS.blind], ["Uses the group at prediction", PARETO_COLORS.group]].map(([l, c]) => (
          <span key={l} className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: c }} />{l}</span>
        ))}
        <span className="inline-flex items-center gap-1.5"><span className="w-4 border-t border-dashed border-muted-foreground" />Pareto frontier</span>
      </div>
    </div>
  );
}

export function ReliabilityChart({ series, height = 280 }: { series: { name: string; rows: { mean_pred: number; frac_pos: number; n: number }[] }[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 10, right: 16, left: -6, bottom: 14 }}>
        <CartesianGrid />
        <XAxis type="number" dataKey="mean_pred" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v) => pct(v)} {...axisProps}>
          <Label value="Predicted risk" position="insideBottom" offset={-8} style={{ fontSize: 11, fill: "var(--ink-muted)" }} />
        </XAxis>
        <YAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v) => pct(v)} {...axisProps} width={44} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--axis)" strokeDasharray="3 3" />
        <Tooltip
          content={({ active, payload }) =>
            active && payload?.length ? (
              <TipBox rows={payload.map((p, i) => ({ name: String(p.name), value: `${pct((p.payload as { mean_pred: number }).mean_pred)} → ${pct(Number(p.value))} (n=${(p.payload as { n: number }).n})`, color: SERIES[i] }))} />
            ) : null
          }
        />
        <Legend iconType="plainline" verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
        {series.map((s, i) => (
          <Line key={s.name} name={s.name} data={s.rows} dataKey="frac_pos" stroke={SERIES[i]} strokeWidth={2} dot={{ r: 3.5, strokeWidth: 0, fill: SERIES[i] }} activeDot={{ r: 5 }} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ThresholdLines({ data, groups, marker, yLabel, height = 220 }: {
  data: Record<string, number>[]; groups: { key: string; label: string }[]; marker: number[]; yLabel: string; height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 14 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="t" type="number" domain={[0.05, 0.95]} ticks={[0.1, 0.3, 0.5, 0.7, 0.9]} tickFormatter={(v) => v.toFixed(1)} {...axisProps}>
          <Label value="Threshold" position="insideBottom" offset={-8} style={{ fontSize: 11, fill: "var(--ink-muted)" }} />
        </XAxis>
        <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps} width={44} />
        {marker.map((m, i) => <ReferenceLine key={i} x={m} stroke={marker.length > 1 ? SERIES[i] : "var(--ink-2)"} strokeDasharray="3 3" />)}
        <Tooltip
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <TipBox title={`${yLabel} at threshold ${Number(label).toFixed(2)}`} rows={payload.map((p, i) => ({ name: String(p.name), value: pct(Number(p.value), 1), color: SERIES[i] }))} />
            ) : null
          }
        />
        <Legend iconType="plainline" verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
        {groups.map((g, i) => (
          <Line key={g.key} type="monotone" dataKey={g.key} name={g.label} stroke={SERIES[i]} strokeWidth={2} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
