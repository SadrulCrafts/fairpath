"use client";
import * as React from "react";
import Link from "next/link";
import { ArrowRight, BookOpen, CheckCircle2, Scale, SlidersHorizontal, UserRound, XCircle, MinusCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Callout, Skeleton } from "@/components/ui/misc";
import { Glossed } from "@/components/widgets";
import { getJSON } from "@/lib/api";
import { useDataset } from "@/lib/dataset-context";
import { num, pct } from "@/lib/format";

interface GroupRow { group: string; label: string; n: number; n_at_risk: number; tpr: number; flag_rate: number }

interface DsOverview {
  dataset: string; title: string; short_title: string; n: number; n_test: number; model_label: string; threshold: number;
  test: Record<string, { value: number; lo?: number; hi?: number }>;
  baseline_test: Record<string, { value: number; lo?: number; hi?: number }>;
  calibration: { method: string; test_ece_before: number; test_ece_after: number };
  conformal: { alpha: number; test: { coverage: number; uncertain_share: number } };
  primary_label: string;
  fairness: { deployed: Record<string, { value: number; lo: number; hi: number; perm_p?: number }>; baseline: Record<string, { value: number; lo: number; hi: number }>; groups: GroupRow[]; baseline_groups?: GroupRow[] };
  preflight: Record<string, { label: string; v1: string; v1_passed: boolean; v2: string; v2_code: string }>;
  sensitive_rank: { feature: string; shap_rank: number; perm_rank: number; n_features: number };
  benchmark_best_label: string;
  recommendation?: { deployable_label: string; eodds_change: number; f1_change: number } | null;
  blind_effect?: Record<"baseline" | "blind" | "reweighing" | "blind_reweighing", { eodds: number; spd: number; f1: number }> | null;
  degenerate_methods?: string[];
}
interface Overview { datasets: Record<string, DsOverview>; thesis_reproduction: { match: number; close: number; differs: number; table_5_1_match: number; table_5_1_total: number } | null; synthetic: boolean }

function Verdict({ code }: { code: string }) {
  if (code === "mitigate") return <Badge variant="good"><CheckCircle2 /> Mitigate and measure</Badge>;
  if (code === "mitigate_coarse") return <Badge variant="good"><CheckCircle2 /> Mitigate (coarse)</Badge>;
  if (code === "monitor") return <Badge variant="outline"><MinusCircle /> Monitor</Badge>;
  return <Badge variant="warn"><XCircle /> Not enough data</Badge>;
}

export default function OverviewPage() {
  const { dataset, meta } = useDataset();
  const [ov, setOv] = React.useState<Overview | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  React.useEffect(() => { if (meta) getJSON<Overview>("/api/reports/overview").then(setOv).catch((e) => setErr(e.message)); }, [meta]);
  const d = ov?.datasets[dataset];
  const drop = ov?.datasets.dropout, small = ov?.datasets.student_mat;

  return (
    <div className="animate-fade-in">
      <section className="mb-10 grid items-stretch gap-6 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        <div className="flex flex-col justify-center py-2">
          <div className="eyebrow mb-4">An undergraduate thesis, rebuilt as a working tool</div>
          <h1 className="font-display text-[36px] font-medium leading-[1.06] tracking-[-0.018em] sm:text-[48px]">
            Early warning that is honest about who it helps
          </h1>
          <p className="mt-5 max-w-[60ch] text-[15.5px] leading-[1.7] text-muted-foreground">
            FairPath estimates which students may need support, explains each estimate, suggests help an institution can actually offer,
            and audits whether the model reaches every group equally. Before claiming any fairness fix, it asks whether the data is
            large enough to tell.
          </p>
          <div className="mt-7 flex flex-wrap gap-2.5">
            <Link href="/predict/" className="inline-flex h-10 items-center gap-2 rounded-lg bg-primary px-4 text-[14px] font-medium text-primary-foreground transition-colors hover:bg-primary/90">
              <UserRound className="h-4 w-4" /> Score a student
            </Link>
            <Link href="/fairness/" className="inline-flex h-10 items-center gap-2 rounded-lg border bg-card px-4 text-[14px] font-medium transition-colors hover:bg-accent">
              <Scale className="h-4 w-4" /> Open the fairness audit
            </Link>
          </div>
        </div>
        {d ? <GapHero d={d} /> : <Skeleton className="h-[340px] rounded-xl" />}
      </section>

      {err && <Callout tone="warn" title="Reports not available">{err}</Callout>}

      <section aria-label="Key results" className="grid grid-cols-2 overflow-hidden rounded-xl border bg-card lg:grid-cols-4">
        {!d ? Array.from({ length: 4 }).map((_, i) => <div key={i} className="p-5"><Skeleton className="h-[84px]" /></div>) : (
          <>
            <Stat label="Ranking quality" glossary="roc_auc" value={num(d.test.roc_auc.value, 3)}
              sub={<>ROC-AUC on {d.n_test} held-out students, 95% CI {num(d.test.roc_auc.lo, 3)}–{num(d.test.roc_auc.hi, 3)}</>} />
            <Stat label="At-risk students reached" glossary="recall" value={pct(d.test.recall.value)}
              sub={<>at a {pct(d.threshold)} risk threshold; precision {pct(d.test.precision.value)}</>} />
            <Stat label="Calibration error" glossary="ece" value={num(d.calibration.test_ece_after, 3)} good={d.calibration.test_ece_after < 0.05}
              sub={d.calibration.method === "none"
                ? <>already well calibrated, so no correction was applied</>
                : <>{d.calibration.method === "isotonic" ? "isotonic" : "Platt"} scaling chosen by cross-validation (uncorrected {num(d.calibration.test_ece_before, 3)})</>} />
            <Stat label="Confidence label coverage" glossary="confidence" value={pct(d.conformal.test.coverage)}
              sub={<>target {pct(1 - d.conformal.alpha)}; {pct(d.conformal.test.uncertain_share)} of students marked uncertain</>} />
          </>
        )}
      </section>

      <section className="mt-10">
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="font-display text-[24px] font-medium tracking-tight">What the project found</h2>
          <Link href="/methodology/" className="text-[13px] text-muted-foreground hover:text-foreground">Read the full methodology</Link>
        </div>
        {!drop || !small ? <Skeleton className="h-48" /> : (
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <div className="eyebrow">Finding one</div>
                <CardTitle className="font-display text-[19px] font-medium">Whether mitigation can work depends on the data, not the method</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-[13.5px] leading-6 text-muted-foreground">
                <p>
                  With {drop.n.toLocaleString()} university students, gender gaps are large and measurable, and the pre-flight check says go ahead.
                  With {small.n} school students, the same checks say the gaps can&apos;t be told apart from noise, so any &ldquo;fix&rdquo; would be impossible to verify.
                </p>
                <div className="flex flex-wrap gap-2 pt-1">
                  <span className="text-[12px]">{drop.short_title}:</span><Verdict code={drop.preflight[Object.keys(drop.preflight)[0]].v2_code} />
                  <span className="text-[12px]">{small.short_title}:</span><Verdict code={small.preflight[Object.keys(small.preflight)[0]].v2_code} />
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="eyebrow">Finding two</div>
                <CardTitle className="font-display text-[19px] font-medium">Deleting the gender column helps, but it doesn&apos;t delete the gap</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-[13.5px] leading-6 text-muted-foreground">
                {drop.blind_effect && small.blind_effect && (
                  <p>
                    On the dropout data, removing gender cut the equalized-odds gap from {num(drop.blind_effect.baseline.eodds, 2)} to{" "}
                    {num(drop.blind_effect.blind.eodds, 2)}. Most of what was left came through correlated features like programme, fees and age.
                    On the school data it changed almost nothing ({num(small.blind_effect.baseline.eodds, 2)} → {num(small.blind_effect.blind.eodds, 2)}),
                    where sex ranks {small.sensitive_rank.shap_rank} of {small.sensitive_rank.n_features} features anyway.
                    &ldquo;We don&apos;t use gender&rdquo; is a starting point, not a fairness guarantee.
                  </p>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="eyebrow">Finding three</div>
                <CardTitle className="font-display text-[19px] font-medium">A fairer model didn&apos;t have to be a worse one</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-[13.5px] leading-6 text-muted-foreground">
                <p>
                  The deployed dropout model never sees gender and is trained with Reweighing. Its equal-opportunity gap is{" "}
                  {num(drop.fairness.deployed.tpr_gap.value, 2)} against {num(drop.fairness.baseline.tpr_gap.value, 2)} for the unmitigated model,
                  with ROC-AUC {num(drop.test.roc_auc.value, 3)} vs {num(drop.baseline_test.roc_auc.value, 3)}.
                </p>
                {ov?.thesis_reproduction && (
                  <p className="text-[12.5px]">
                    Along the way, {ov.thesis_reproduction.table_5_1_match} of {ov.thesis_reproduction.table_5_1_total} headline performance numbers from the
                    thesis were re-computed from scratch to within 0.005.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </section>

      <section className="mt-10 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle><Glossed k="preflight">Pre-flight check, every audited group</Glossed></CardTitle>
            <CardDescription>Run before any mitigation. Version 1 is the thesis rule; version 2 separates &ldquo;fair enough&rdquo; from &ldquo;too small to tell&rdquo;.</CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            {!ov ? <Skeleton className="h-40" /> : (
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b text-left text-[12px] text-muted-foreground">
                    <th className="py-2 pr-3 font-medium">Dataset</th><th className="py-2 pr-3 font-medium">Group axis</th>
                    <th className="py-2 pr-3 font-medium">v1 (thesis)</th><th className="py-2 font-medium">v2</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.values(ov.datasets).flatMap((ds) =>
                    Object.entries(ds.preflight).map(([axis, v]) => (
                      <tr key={ds.dataset + axis} className="border-b last:border-0">
                        <td className="py-2 pr-3 text-muted-foreground">{ds.short_title}</td>
                        <td className="py-2 pr-3">{v.label}</td>
                        <td className="py-2 pr-3">{v.v1_passed ? <Badge variant="good">Attempt</Badge> : <Badge variant="outline">Do not mitigate</Badge>}</td>
                        <td className="py-2"><Verdict code={v.v2_code} /></td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
        <div className="grid content-start gap-3">
          {[
            { href: "/predict/", icon: UserRound, title: "Score one student", body: "Risk, confidence, what drove it, and a path to lower risk." },
            { href: "/what-if/", icon: SlidersHorizontal, title: "Try the what-if simulator", body: "Move the sliders and watch the risk respond." },
            { href: "/fairness/", icon: Scale, title: "Open the fairness audit", body: "Group gaps with intervals, mitigation trade-offs, threshold effects." },
            { href: "/methodology/", icon: BookOpen, title: "Read the model card", body: "Data, limits, and what this tool must never be used for." },
          ].map(({ href, icon: Icon, title, body }) => (
            <Link key={href} href={href} className="group flex items-start gap-3 rounded-lg border bg-card p-4 transition-colors hover:bg-accent/50">
              <div className="rounded-md bg-brand-soft p-2 text-brand"><Icon className="h-4 w-4" /></div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between text-[14px] font-medium">{title}<ArrowRight className="h-4 w-4 opacity-0 transition-opacity group-hover:opacity-60" /></div>
                <div className="text-[12.5px] text-muted-foreground">{body}</div>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <Callout className="mt-10" title="Please read before using any score">{meta?.disclaimer}</Callout>
    </div>
  );
}

function Stat({ label, value, sub, glossary, good }: { label: string; value: React.ReactNode; sub: React.ReactNode; glossary?: string; good?: boolean }) {
  return (
    <div className="border-b border-r p-5 last:border-r-0 lg:border-b-0 [&:nth-child(2)]:border-r-0 lg:[&:nth-child(2)]:border-r">
      <div className="text-[12.5px] text-muted-foreground">{glossary ? <Glossed k={glossary}>{label}</Glossed> : label}</div>
      <div className={good ? "display-num mt-2 text-[34px] leading-none text-[#006300] dark:text-good" : "display-num mt-2 text-[34px] leading-none"}>{value}</div>
      <div className="mt-2.5 text-[12px] leading-5 text-muted-foreground">{sub}</div>
    </div>
  );
}

/**
 * The signature element: recall for each group, before and after mitigation, as a dumbbell.
 * The deployed row starts where the unmitigated one is and settles into place once, on load.
 */
function GapHero({ d }: { d: DsOverview }) {
  const base = d.fairness.baseline_groups ?? [];
  const dep = d.fairness.groups;
  const [settled, setSettled] = React.useState(false);
  React.useEffect(() => {
    const t = window.setTimeout(() => setSettled(true), 350);
    return () => window.clearTimeout(t);
  }, [d.dataset]);
  React.useEffect(() => setSettled(false), [d.dataset]);
  if (base.length < 2 || dep.length < 2) return null;

  const all = [...base, ...dep].map((g) => g.tpr);
  const lo = Math.max(0, Math.floor((Math.min(...all) - 0.1) * 10) / 10);
  const hi = 1;
  const x = (v: number) => 4 + ((v - lo) / (hi - lo)) * 92;
  const colors = ["var(--series-1)", "var(--series-2)"];
  const pre = Object.values(d.preflight)[0]?.v2_code;
  const measurable = pre === "mitigate" || pre === "mitigate_coarse";
  const rows = [
    { key: "base", title: "Unmitigated model", sub: `sees ${d.primary_label.toLowerCase()}`, groups: base, gap: d.fairness.baseline.tpr_gap },
    { key: "dep", title: "Deployed model", sub: `never sees ${d.primary_label.toLowerCase()}`, groups: settled ? dep : base, gap: d.fairness.deployed.tpr_gap },
  ];
  const ticks = Array.from({ length: Math.round((hi - lo) * 10) + 1 }, (_, i) => lo + i * 0.1).filter((t) => t <= hi + 1e-9);

  return (
    <div className="relative flex flex-col overflow-hidden rounded-xl border bg-card p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-[15px] font-semibold tracking-tight">Who the model reaches, by {d.primary_label.toLowerCase()}</h2>
          <p className="mt-1 text-[12.5px] leading-5 text-muted-foreground">
            Share of at-risk students flagged, on {d.n_test} held-out students. Closer dots mean a fairer model.
          </p>
        </div>
        <Badge variant={measurable ? "good" : "warn"} className="shrink-0">{measurable ? "Measurable" : "Not enough data"}</Badge>
      </div>

      <div className="mt-6 flex-1 space-y-7" role="img"
        aria-label={`Equal-opportunity gap ${num(d.fairness.baseline.tpr_gap.value, 2)} for the unmitigated model and ${num(d.fairness.deployed.tpr_gap.value, 2)} for the deployed model`}>
        {rows.map((r) => {
          const [a, b] = r.groups;
          const left = Math.min(x(a.tpr), x(b.tpr)), right = Math.max(x(a.tpr), x(b.tpr));
          return (
            <div key={r.key}>
              <div className="mb-2 flex items-baseline justify-between gap-3">
                <div className="text-[13px]"><span className="font-medium">{r.title}</span> <span className="text-muted-foreground">· {r.sub}</span></div>
                <div className="text-right">
                  <span className="display-num text-[22px] leading-none">{num(r.gap.value, 3)}</span>
                  <span className="ml-1.5 text-[11.5px] text-muted-foreground">gap</span>
                </div>
              </div>
              <div className="relative h-9">
                <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-[var(--grid)]" />
                <div className="absolute top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-[var(--ink-muted)]/50 transition-all duration-[1100ms] ease-[cubic-bezier(.2,.8,.2,1)]"
                  style={{ left: `${left}%`, width: `${right - left}%` }} />
                {r.groups.map((g, i) => (
                  <div key={g.group} className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-[1100ms] ease-[cubic-bezier(.2,.8,.2,1)]" style={{ left: `${x(g.tpr)}%` }}>
                    <div className="h-4 w-4 rounded-full ring-[3px] ring-[var(--surface-1)]" style={{ background: colors[i] }} />
                    <div className={`absolute left-1/2 -translate-x-1/2 whitespace-nowrap text-[11.5px] tnum ${i === 0 ? "-top-5" : "top-5"}`}>
                      <span className="font-medium">{g.label}</span> <span className="text-muted-foreground">{pct(g.tpr)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
        <div className="relative h-4 text-[10.5px] text-muted-foreground tnum" aria-hidden>
          {ticks.map((t) => (
            <span key={t} className="absolute -translate-x-1/2" style={{ left: `${x(t)}%` }}>{Math.round(t * 100)}%</span>
          ))}
        </div>
      </div>

      <p className="mt-5 border-t pt-4 text-[12.5px] leading-5 text-muted-foreground">
        {measurable ? (
          <>Gap {num(d.fairness.baseline.tpr_gap.value, 2)} → {num(d.fairness.deployed.tpr_gap.value, 2)}, with ROC-AUC {num(d.baseline_test.roc_auc.value, 3)} → {num(d.test.roc_auc.value, 3)}. The pre-flight check says this difference can be measured on data this size.</>
        ) : (
          <>The two gaps have overlapping 95% intervals ({num(d.fairness.baseline.tpr_gap.lo, 2)}–{num(d.fairness.baseline.tpr_gap.hi, 2)} vs {num(d.fairness.deployed.tpr_gap.lo, 2)}–{num(d.fairness.deployed.tpr_gap.hi, 2)}). With {d.n} students the pre-flight check says no fairness claim can be made either way.</>
        )}
      </p>
    </div>
  );
}
