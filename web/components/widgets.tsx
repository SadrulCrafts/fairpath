"use client";
import * as React from "react";
import { ArrowDownRight, ArrowRight, CheckCircle2, CircleHelp, HandHeart, Scale, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { InfoTip } from "@/components/ui/tooltip";
import { Callout } from "@/components/ui/misc";
import { GLOSSARY } from "@/lib/glossary";
import { bandMeta, pct } from "@/lib/format";
import type { Band, Contribution, Prediction } from "@/lib/types";
import { cn } from "@/lib/utils";

export function Glossed({ k, children }: { k: keyof typeof GLOSSARY | string; children?: React.ReactNode }) {
  const g = GLOSSARY[k];
  return (
    <span className="inline-flex items-center gap-1">
      {children ?? g?.title ?? k}
      {g && <InfoTip><div className="font-medium text-foreground">{g.title}</div><div className="mt-0.5 text-muted-foreground">{g.body}</div></InfoTip>}
    </span>
  );
}

export function KpiCard({ label, value, sub, glossary, tone, className }: {
  label: string; value: React.ReactNode; sub?: React.ReactNode; glossary?: string; tone?: "good" | "warn"; className?: string;
}) {
  return (
    <Card className={cn("p-4", className)}>
      <div className="flex items-center gap-1 text-[12.5px] text-muted-foreground">{glossary ? <Glossed k={glossary}>{label}</Glossed> : label}</div>
      <div className={cn("mt-1.5 text-[26px] font-semibold leading-none tracking-tight", tone === "good" && "text-[#006300] dark:text-good")}>{value}</div>
      {sub && <div className="mt-2 text-[12px] leading-5 text-muted-foreground">{sub}</div>}
    </Card>
  );
}

/** Semicircular gauge; the band is always spelled out next to the colour. */
export function RiskGauge({ risk, band, bands, threshold }: { risk: number; band: Band; bands: { low: number; high: number }; threshold: number }) {
  const r = 80, cx = 100, cy = 96;
  const pt = (v: number) => {
    const a = Math.PI * (1 - v);
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  };
  const arc = (a: number, b: number) => {
    const [x1, y1] = pt(a), [x2, y2] = pt(b);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  const [mx, my] = pt(Math.min(Math.max(risk, 0.005), 0.995));
  const [tx, ty] = pt(threshold);
  const m = bandMeta[band];
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 112" className="w-full max-w-[260px]" role="img" aria-label={`Predicted risk ${pct(risk)}, ${m.label}`}>
        <path d={arc(0, bands.low - 0.004)} stroke="var(--status-good)" strokeWidth="12" fill="none" opacity="0.85" />
        <path d={arc(bands.low + 0.004, bands.high - 0.004)} stroke="var(--status-serious)" strokeWidth="12" fill="none" opacity="0.85" />
        <path d={arc(bands.high + 0.004, 1)} stroke="var(--status-critical)" strokeWidth="12" fill="none" opacity="0.85" />
        <line x1={tx} y1={ty} x2={cx + (tx - cx) * 0.78} y2={cy + (ty - cy) * 0.78} stroke="var(--ink-2)" strokeWidth="1.5" strokeDasharray="2 2" />
        <circle cx={mx} cy={my} r="9" fill="var(--surface-1)" stroke="var(--ink-1)" strokeWidth="3" style={{ transition: "all .4s ease" }} />
      </svg>
      <div className="-mt-10 text-center">
        <div className="text-[44px] font-semibold leading-none tracking-tight tnum">{pct(risk)}</div>
        <div className="mt-2"><Badge variant={m.variant}>{m.label}</Badge></div>
      </div>
    </div>
  );
}

export function ShapBars({ items, rest }: { items: Contribution[]; rest?: number }) {
  const max = Math.max(...items.map((c) => Math.abs(c.shap)), Math.abs(rest ?? 0), 1e-6);
  return (
    <div className="space-y-2.5">
      {items.map((c) => {
        const up = c.shap > 0;
        const w = (Math.abs(c.shap) / max) * 50;
        return (
          <div key={c.feature} className="group grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-center gap-3 text-[13px]">
            <div className="min-w-0 text-right">
              <div className="truncate font-medium">{c.label}</div>
              <div className="truncate text-[12px] text-muted-foreground">{c.display_value}</div>
            </div>
            <div className="relative h-5">
              <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
              <div
                className="absolute top-1 h-3 rounded-[3px] transition-all"
                style={{
                  left: up ? "50%" : `${50 - w}%`, width: `${Math.max(w, 0.8)}%`,
                  background: up ? "var(--status-critical)" : "var(--series-1)",
                }}
                title={`${up ? "+" : "−"}${Math.abs(c.shap).toFixed(3)} log-odds`}
              />

            </div>
          </div>
        );
      })}
      <div className="flex justify-between pt-1 text-[11.5px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: "var(--series-1)" }} /> lowers risk</span>
        <span className="inline-flex items-center gap-1.5">raises risk <span className="h-2 w-2 rounded-sm" style={{ background: "var(--status-critical)" }} /></span>
      </div>
    </div>
  );
}

export function ConfidenceBadge({ level }: { level: "confident" | "uncertain" | "atypical" }) {
  if (level === "confident") return <Badge variant="good"><CheckCircle2 /> Confident</Badge>;
  if (level === "uncertain") return <Badge variant="warn"><CircleHelp /> Uncertain</Badge>;
  return <Badge variant="serious"><CircleHelp /> Unusual profile</Badge>;
}

export function PredictionPanel({ p, bands, compact = false }: { p: Prediction; bands: { low: number; high: number }; compact?: boolean }) {
  return (
    <div className="space-y-5">
      <Card>
        <CardContent className="grid gap-6 p-6 sm:grid-cols-[1fr_1.2fr] sm:items-center">
          <RiskGauge risk={p.risk} band={p.band} bands={bands} threshold={p.threshold} />
          <div className="space-y-3 text-[13.5px]">
            <p className="leading-6">{p.band_text}</p>
            <div className="flex flex-wrap items-center gap-2">
              <ConfidenceBadge level={p.confidence.level} />
              {p.flagged ? <Badge variant="critical">Suggested for outreach</Badge> : <Badge variant="outline">Below the outreach threshold</Badge>}
            </div>
            <p className="text-[12.5px] leading-5 text-muted-foreground">
              <Glossed k="confidence">Why this confidence?</Glossed> {p.confidence.explanation}
            </p>
            <p className="text-[12px] text-muted-foreground tnum">
              <Glossed k="threshold">Outreach threshold</Glossed>: {pct(p.threshold)} calibrated risk
            </p>
          </div>
        </CardContent>
      </Card>

      {p.explanation && (
        <Card>
          <CardHeader>
            <CardTitle><Glossed k="shap">What drove this score</Glossed></CardTitle>
            <CardDescription>The pieces of information that moved this student&apos;s risk most, compared with a typical student.</CardDescription>
          </CardHeader>
          <CardContent><ShapBars items={p.explanation.top.slice(0, compact ? 5 : 8)} /></CardContent>
        </Card>
      )}

      {p.counterfactuals && !compact && <PathToSuccess cf={p.counterfactuals} />}
      {!compact && <FairnessNote note={p.fairness_note} />}
    </div>
  );
}

export function PathToSuccess({ cf }: { cf: NonNullable<Prediction["counterfactuals"]> }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><HandHeart className="h-4 w-4 text-brand" /> Path to lower risk</CardTitle>
        <CardDescription>
          Support the institution could offer that the model thinks would help most. Only things that can realistically change are
          considered; gender, age, family background and past grades are never touched.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {cf.options.length === 0 && <Callout>{cf.message ?? "Nothing actionable to suggest for this profile."}</Callout>}
        {cf.message && cf.options.length > 0 && <Callout tone="info">{cf.message}</Callout>}
        {cf.options.map((o, i) => (
          <div key={i} className="rounded-lg border p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-[13px] font-medium">
                <Sparkles className="h-3.5 w-3.5 text-brand" /> Option {i + 1}
                <span className="font-normal text-muted-foreground">· {o.n_changes} change{o.n_changes > 1 ? "s" : ""}</span>
              </div>
              <div className="flex items-center gap-1.5 text-[13px] tnum">
                <span className="text-muted-foreground">{pct(cf.current_risk)}</span>
                <ArrowRight className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="font-semibold">{pct(o.new_risk)}</span>
                <Badge variant="good" className="ml-1"><ArrowDownRight />{pct(o.risk_reduction)}</Badge>
              </div>
            </div>
            <ul className="space-y-2">
              {o.changes.map((c) => (
                <li key={c.feature} className="text-[13px] leading-5">
                  <div><span className="font-medium">{c.action}</span></div>
                  <div className="text-[12px] text-muted-foreground">{c.label}: {c.from_label} → {c.to_label}</div>
                </li>
              ))}
            </ul>
          </div>
        ))}
        <p className="text-[12px] leading-5 text-muted-foreground">
          These are model-based what-ifs, not causal promises. They show what the model responds to, which is a good starting point
          for a conversation with the student, not a prescription.
        </p>
      </CardContent>
    </Card>
  );
}

export function FairnessNote({ note }: { note: Prediction["fairness_note"] }) {
  if (!note?.text) return null;
  return (
    <Card className="bg-muted/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Scale className="h-4 w-4" /> Fairness context</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-[13.5px] leading-6 text-muted-foreground">{note.text}</p>
        <div className="grid gap-3 sm:grid-cols-2">
          {note.groups.map((g) => (
            <div key={g.group} className="rounded-md border bg-card p-3 text-[12.5px]">
              <div className="mb-1.5 font-medium">{g.label} <span className="font-normal text-muted-foreground">· {g.n} test students</span></div>
              <div className="grid grid-cols-2 gap-y-1 tnum text-muted-foreground">
                <span>Flagged</span><span className="text-right text-foreground">{pct(g.flag_rate)}</span>
                <span>At-risk caught</span><span className="text-right text-foreground">{pct(g.tpr)}</span>
                <span>Actually at risk</span><span className="text-right text-foreground">{pct(g.base_rate)}</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
