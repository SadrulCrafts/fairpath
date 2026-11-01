"use client";
import * as React from "react";
import Link from "next/link";
import { ArrowRight, BookOpen, CheckCircle2, Scale, SlidersHorizontal, UserRound, XCircle, MinusCircle } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Callout, Skeleton } from "@/components/ui/misc";
import { Glossed, KpiCard } from "@/components/widgets";
import { getJSON } from "@/lib/api";
import { useDataset } from "@/lib/dataset-context";
import { num, pct } from "@/lib/format";

interface DsOverview {
  dataset: string; title: string; short_title: string; n: number; n_test: number; model_label: string; threshold: number;
  test: Record<string, { value: number; lo?: number; hi?: number }>;
  baseline_test: Record<string, { value: number; lo?: number; hi?: number }>;
  calibration: { method: string; test_ece_before: number; test_ece_after: number };
  conformal: { alpha: number; test: { coverage: number; uncertain_share: number } };
  primary_label: string;
  fairness: { deployed: Record<string, { value: number; lo: number; hi: number; perm_p?: number }>; baseline: Record<string, { value: number; lo: number; hi: number }> };
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
      <PageHeader eyebrow="An undergraduate thesis, rebuilt as a working tool" title="Early warning that is honest about who it helps">
        Universities increasingly use models to spot students who might drop out. FairPath predicts that risk, explains it, suggests
        support that could help, and shows, with uncertainty, whether the model treats groups of students differently. It also asks a
        question most fairness tools skip: <em>is there even enough data to tell?</em>
      </PageHeader>

      {err && <Callout tone="warn" title="Reports not available">{err}</Callout>}

      <section aria-label="Key results" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {!d ? Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[112px]" />) : (
          <>
            <KpiCard label="Ranking quality" glossary="roc_auc" value={num(d.test.roc_auc.value, 3)}
              sub={<>ROC-AUC on {d.n_test} held-out students<br />95% CI {num(d.test.roc_auc.lo, 3)} – {num(d.test.roc_auc.hi, 3)}</>} />
            <KpiCard label="At-risk students caught" glossary="recall" value={pct(d.test.recall.value)}
              sub={<>at a {pct(d.threshold)} risk threshold<br />precision {pct(d.test.precision.value)}</>} />
            <KpiCard label="Calibration error" glossary="ece" value={num(d.calibration.test_ece_after, 3)} tone={d.calibration.test_ece_after < 0.05 ? "good" : undefined}
              sub={d.calibration.method === "none"
                ? <>already well calibrated, so no<br />post-hoc correction was applied</>
                : <>{d.calibration.method === "isotonic" ? "isotonic" : "Platt"} scaling chosen by cross-validation<br />(uncorrected: {num(d.calibration.test_ece_before, 3)})</>} />
            <KpiCard label={`Equal-opportunity gap (${d.primary_label.toLowerCase()})`} glossary="tpr_gap" value={num(d.fairness.deployed.tpr_gap.value, 3)}
              sub={<>unmitigated model: {num(d.fairness.baseline.tpr_gap.value, 3)}<br />95% CI {num(d.fairness.deployed.tpr_gap.lo, 2)} – {num(d.fairness.deployed.tpr_gap.hi, 2)}</>} />
          </>
        )}
      </section>

      <section className="mt-10">
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="text-lg font-semibold tracking-tight">What the project found</h2>
          <Link href="/methodology/" className="text-[13px] text-muted-foreground hover:text-foreground">Full methodology →</Link>
        </div>
        {!drop || !small ? <Skeleton className="h-48" /> : (
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <div className="eyebrow">Finding 1</div>
                <CardTitle>Whether mitigation can work depends on the data, not the method</CardTitle>
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
                <div className="eyebrow">Finding 2</div>
                <CardTitle>Deleting the gender column helps, but it doesn&apos;t delete the gap</CardTitle>
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
                <div className="eyebrow">Finding 3</div>
                <CardTitle>A fairer model didn&apos;t have to be a worse one</CardTitle>
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
