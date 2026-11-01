"use client";
import * as React from "react";
import { AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Callout, Label, Skeleton } from "@/components/ui/misc";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ForestPlot, GroupedBars, ParetoChart, ThresholdLines, paretoColor, type ParetoPoint } from "@/components/charts";
import { Glossed } from "@/components/widgets";
import { getJSON } from "@/lib/api";
import { useData, useDataset } from "@/lib/dataset-context";
import { groupsAt, summarise, type Sweep } from "@/lib/fairness";
import { MODEL_LABELS, num, pct, pval } from "@/lib/format";
import type { Gap, GroupRow, Stat } from "@/lib/types";

interface AxisFair { label: string; deployed: { groups: GroupRow[]; gaps: Record<string, Gap>; warnings: string[]; n: number }; baseline: { groups: GroupRow[]; gaps: Record<string, Gap> } }
interface Final {
  model_label: string; deployed: { threshold: number }; baseline: { threshold: number };
  fairness: Record<string, AxisFair>; threshold_sweeps: Record<string, { deployed: Sweep; baseline: Sweep; labels: Record<string, string> }>;
}
interface Method {
  method: string; label: string; stage: string; needs_sensitive_at_prediction: boolean; about: string; pareto: boolean;
  f1: Stat; roc_auc: Stat; spd: Stat; eodds: Stat; tpr_gap: Stat; fpr_gap: Stat; flag_rate: Stat; degenerate?: boolean; n_failed_folds?: number;
  tests?: Record<string, { p_value: number; p_holm?: number; rank_biserial: number; effect: string }>;
}
interface Study { design: string; summary: Record<string, Record<string, { methods: Method[]; frontier: string[]; model_label: string }>>; recommendations: Record<string, Record<string, { deployable_label: string; eodds_change: number; f1_change: number; lowest_gap_post: string | null }>> }
interface Check { id: number; name: string; criterion: string; value: number; passed: boolean }
interface Pre { axes: Record<string, { label: string; v1: { checks: Check[]; verdict: string; passed: boolean }; v2: { checks: Check[]; verdict: string; verdict_code: string } }> }

const GAP_ROWS: { key: string; label: string; g: string }[] = [
  { key: "spd", label: "Demographic parity gap", g: "spd" },
  { key: "tpr_gap", label: "Equal opportunity gap", g: "tpr_gap" },
  { key: "fpr_gap", label: "False positive gap", g: "fpr_gap" },
  { key: "eodds", label: "Equalized odds gap", g: "eodds" },
  { key: "ppv_gap", label: "Predictive parity gap", g: "ppv_gap" },
];
const STAGE_LABEL: Record<string, string> = { none: "–", pre: "Pre", in: "In", post: "Post" };

function CheckList({ checks }: { checks: Check[] }) {
  return (
    <ul className="space-y-2">
      {checks.map((c) => (
        <li key={c.id} className="flex items-start gap-2 text-[13px]">
          {c.passed ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-good" /> : <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-critical" />}
          <div className="flex-1">{c.name}<div className="text-[12px] text-muted-foreground tnum">value {c.value} · needs {c.criterion}</div></div>
        </li>
      ))}
    </ul>
  );
}

export default function FairnessPage() {
  const { dataset } = useDataset();
  const final = useData((ds) => getJSON<Final>(`/api/reports/${ds}/final_model`));
  const study = useData((ds) => getJSON<Study>(`/api/reports/${ds}/fairness_study`));
  const pre = useData((ds) => getJSON<Pre>(`/api/reports/${ds}/preflight`));
  const [axis, setAxis] = React.useState<string>("");
  const [model, setModel] = React.useState<string>("");
  const [metric, setMetric] = React.useState("tpr");
  const [perGroup, setPerGroup] = React.useState(false);
  const [thr, setThr] = React.useState<Record<string, number>>({});

  React.useEffect(() => { setAxis(""); setModel(""); }, [dataset]);
  const axes = final.data ? Object.keys(final.data.fairness) : [];
  const ax = axis && final.data?.fairness[axis] ? axis : axes[0];
  const fb = ax ? final.data?.fairness[ax] : undefined;
  const sweep = ax ? final.data?.threshold_sweeps[ax] : undefined;
  const studyAxis = ax && study.data?.summary[ax] ? study.data.summary[ax] : undefined;
  const models = studyAxis ? Object.keys(studyAxis) : [];
  const mdl = model && studyAxis?.[model] ? model : models[models.length - 1];
  const blk = mdl ? studyAxis?.[mdl] : undefined;
  const rec = ax && mdl ? study.data?.recommendations[ax]?.[mdl] : undefined;
  const pf = ax ? pre.data?.axes[ax] : undefined;
  const numbered = blk ? [...blk.methods].filter((m) => !m.degenerate).sort((a, b) => a.eodds.mean - b.eodds.mean) : [];
  const numberOf = (method: string) => numbered.findIndex((m) => m.method === method) + 1;
  const BASE_DEP_COLORS = ["var(--series-2)", "var(--series-1)"];

  React.useEffect(() => {
    if (sweep && final.data) setThr(Object.fromEntries(Object.keys(sweep.deployed.groups).map((g) => [g, final.data!.deployed.threshold])));
  }, [sweep, final.data]);

  const labelOf = (g: string) => sweep?.labels[g] ?? g;
  const live = sweep && Object.keys(thr).length ? groupsAt(sweep.deployed, thr) : null;
  const liveSum = live ? summarise(live) : null;
  const defaultSum = sweep && final.data ? summarise(groupsAt(sweep.deployed, Object.fromEntries(Object.keys(sweep.deployed.groups).map((g) => [g, final.data!.deployed.threshold])))) : null;

  const metricKey = { flag: "flag_rate", tpr: "tpr", fpr: "fpr", ppv: "ppv" }[metric] as keyof GroupRow;
  const barData = fb ? fb.deployed.groups.map((g) => ({
    label: `${g.label}${g.reliability !== "ok" ? " *" : ""}`,
    baseline: (fb.baseline.groups.find((b) => b.group === g.group)?.[metricKey] as number) ?? null,
    deployed: (g[metricKey] as number) ?? null,
  })) : [];

  const tprCurve = sweep ? sweep.deployed.thresholds.map((t, i) => {
    const row: Record<string, number> = { t };
    for (const [g, c] of Object.entries(sweep.deployed.groups)) {
      const pos = c.tp[i] + c.fn[i];
      row[g] = pos ? c.tp[i] / pos : 0;
    }
    return row;
  }) : [];

  const loading = final.loading || study.loading || pre.loading;
  return (
    <div className="animate-fade-in">
      <PageHeader title="Does the model treat groups differently?">
        Everything here is measured on students the model never saw during training, with 95% bootstrap intervals, so you can tell a
        real gap from noise. Pick a group axis to audit.
      </PageHeader>

      <Card className="mb-6">
        <CardContent className="grid gap-4 p-4 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label>Group axis</Label>
            {axes.length ? <Select ariaLabel="Group axis" value={ax!} onValueChange={setAxis} options={axes.map((a) => ({ value: a, label: final.data!.fairness[a].label }))} /> : <Skeleton className="h-9" />}
          </div>
          <div className="space-y-1.5">
            <Label>Model for the mitigation study</Label>
            {models.length ? <Select ariaLabel="Model" value={mdl!} onValueChange={setModel} options={models.map((m) => ({ value: m, label: MODEL_LABELS[m] ?? m }))} />
              : <div className="flex h-9 items-center text-[12.5px] text-muted-foreground">{loading ? "Loading…" : "Not studied for this axis"}</div>}
          </div>
          <div className="flex items-end text-[12.5px] leading-5 text-muted-foreground">
            Deployed: {final.data?.model_label}, blind to the primary attribute, trained with Reweighing.
          </div>
        </CardContent>
      </Card>

      {pre.data && Object.values(pre.data.axes)[0]?.v2.verdict_code === "insufficient" && (
        <Callout tone="warn" title="This dataset is too small for firm fairness conclusions" className="mb-6">
          The pre-flight check says a gap change of 0.05 can&apos;t be told apart from noise here. Every number on this page is shown with its
          interval for that reason: read the intervals, not the point estimates, and don&apos;t treat any &ldquo;improvement&rdquo; below as proven.
        </Callout>
      )}

      {fb?.deployed.warnings.length ? (
        <Callout tone="warn" title="Small groups: read with care" className="mb-6">
          {fb.deployed.warnings.join(" ")} Groups marked * below are in this situation.
        </Callout>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>How each group is treated</CardTitle>
            <CardDescription>Held-out test set, each model at its own calibrated threshold.</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs value={metric} onValueChange={setMetric}>
              <TabsList>
                <TabsTrigger value="tpr">At-risk caught</TabsTrigger>
                <TabsTrigger value="flag">Flag rate</TabsTrigger>
                <TabsTrigger value="fpr">False alarms</TabsTrigger>
                <TabsTrigger value="ppv">Precision</TabsTrigger>
              </TabsList>
              <TabsContent value={metric}>
                {fb ? <GroupedBars colors={BASE_DEP_COLORS} data={barData} series={[{ key: "baseline", label: "Unmitigated (aware)" }, { key: "deployed", label: "Deployed" }]} /> : <Skeleton className="h-60" />}
              </TabsContent>
            </Tabs>
            {fb && (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-[12.5px] tnum">
                  <thead><tr className="border-b text-left text-muted-foreground"><th className="py-1.5 font-medium">Group</th><th className="font-medium">n</th><th className="font-medium">Actually at risk</th><th className="font-medium">Mean predicted</th><th className="font-medium"><Glossed k="coverage">Coverage</Glossed></th></tr></thead>
                  <tbody>
                    {fb.deployed.groups.map((g) => (
                      <tr key={g.group} className="border-b last:border-0">
                        <td className="py-1.5">{g.label} {g.reliability !== "ok" && <Badge variant="warn" className="ml-1">{g.reliability}</Badge>}</td>
                        <td>{g.n}</td><td>{pct(g.base_rate)}</td><td>{pct(g.mean_score)}</td><td>{pct(g.coverage)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Gap metrics, with uncertainty</CardTitle>
            <CardDescription>Point estimate and 95% bootstrap interval. If an interval touches zero, the gap may be noise.</CardDescription>
          </CardHeader>
          <CardContent>
            {fb ? (
              <>
                <ForestPlot colors={BASE_DEP_COLORS} series={["Unmitigated", "Deployed"]}
                  rows={GAP_ROWS.map((r) => ({ label: r.label, values: [fb.baseline.gaps[r.key], fb.deployed.gaps[r.key]] }))} />
                <table className="mt-4 w-full text-[12.5px] tnum">
                  <thead><tr className="border-b text-left text-muted-foreground"><th className="py-1.5 font-medium">Gap</th><th className="font-medium">Deployed</th><th className="font-medium"><Glossed k="perm_p">p (permutation)</Glossed></th></tr></thead>
                  <tbody>
                    {GAP_ROWS.map((r) => {
                      const g = fb.deployed.gaps[r.key];
                      return (
                        <tr key={r.key} className="border-b last:border-0">
                          <td className="py-1.5"><Glossed k={r.g}>{r.label}</Glossed></td>
                          <td>{num(g.value)} <span className="text-muted-foreground">[{num(g.lo, 2)}, {num(g.hi, 2)}]</span></td>
                          <td>{g.perm_p != null ? pval(g.perm_p) : "–"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </>
            ) : <Skeleton className="h-72" />}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Mitigation methods compared</CardTitle>
          <CardDescription>{study.data?.design}. Every method is scored on the same folds; p-values compare each one with the baseline (Wilcoxon, Holm-corrected).</CardDescription>
        </CardHeader>
        <CardContent>
          {!blk ? (loading ? <Skeleton className="h-80" /> : <Callout>The mitigation study was run for the two main axes and their intersection. Pick one of those to see it.</Callout>) : (
            <div className="grid gap-6 xl:grid-cols-[1.1fr_1fr]">
              <ParetoChart xLabel="Equalized odds gap" points={numbered.map<ParetoPoint>((m) => ({
                method: m.method, label: m.label, n: numberOf(m.method), x: m.eodds.mean, y: m.f1.mean, pareto: m.pareto,
                needsSensitive: m.needs_sensitive_at_prediction, isBaseline: m.method === "baseline",
              }))} />
              <div className="overflow-x-auto">
                <table className="w-full text-[12.5px] tnum [&_td]:px-1 [&_th]:px-1 [&_td]:whitespace-nowrap [&_td:nth-child(2)]:whitespace-normal [&_td:nth-child(n+4)]:text-right [&_th:nth-child(n+4)]:text-right">
                  <thead><tr className="border-b text-left text-muted-foreground"><th className="w-7 py-1.5 font-medium">#</th><th className="py-1.5 font-medium">Method</th><th className="font-medium">Stage</th><th className="font-medium">F1</th><th className="font-medium">EOdds</th><th className="font-medium">SPD</th><th className="font-medium">Flagged</th><th className="font-medium" title="Holm-corrected Wilcoxon p-value for the equalized-odds change">p</th></tr></thead>
                  <tbody>
                    {blk.methods.map((m) => {
                      const p = m.tests?.eodds?.p_holm;
                      return (
                        <tr key={m.method} className="border-b last:border-0" title={m.about}>
                          <td className="py-1.5">{numberOf(m.method) > 0 ? (
                            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-semibold text-white" style={{ background: paretoColor({ isBaseline: m.method === "baseline", needsSensitive: m.needs_sensitive_at_prediction }) }}>{numberOf(m.method)}</span>
                          ) : <span className="text-muted-foreground">–</span>}</td>
                          <td className="py-1.5 pr-2">{m.label}{m.pareto && <span className="ml-1 text-brand" title="On the Pareto frontier">◆</span>}{m.needs_sensitive_at_prediction && <AlertTriangle className="ml-1 inline h-3 w-3 text-muted-foreground" aria-label="uses the group at prediction time" />}
                            {m.degenerate && <Badge variant="warn" className="ml-1">degenerate</Badge>}{!!m.n_failed_folds && <Badge variant="outline" className="ml-1">{m.n_failed_folds} fold failed</Badge>}</td>
                          <td>{STAGE_LABEL[m.stage]}</td>
                          <td>{num(m.f1.mean)}</td><td>{num(m.eodds.mean)}</td><td>{num(m.spd.mean)}</td><td>{pct(m.flag_rate.mean)}</td>
                          <td className={p != null && p < 0.05 ? "font-medium" : "text-muted-foreground"}>{p == null ? "–" : pval(p)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="mt-2 text-[11.5px] leading-5 text-muted-foreground">◆ on the fairness–accuracy frontier · <AlertTriangle className="inline h-3 w-3" /> the score depends on the student&apos;s group (as an input or through a group-specific rule) · <em>degenerate</em>: flags almost everyone or no one, so its error rates are equal by construction. It is not a fair model, and it is left off the chart.</p>
                {rec && (
                  <Callout tone="good" className="mt-4" title="What I would ship">
                    Among methods whose score never depends on the student&apos;s group, <strong>{rec.deployable_label}</strong> gives the best
                    trade-off here: equalized-odds gap {rec.eodds_change <= 0 ? "down" : "up"} {num(Math.abs(rec.eodds_change), 3)}, F1 {rec.f1_change >= 0 ? "up" : "down"} {num(Math.abs(rec.f1_change), 3)}.
                    {rec.lowest_gap_post && <> Methods that use the group directly (as an input or through group-specific thresholds) can push the gap lower still, but then two students with the same record can be scored differently because of their group, which many institutions, and some laws, won&apos;t accept.</>}
                  </Callout>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Threshold explorer</CardTitle>
            <CardDescription>Move the outreach threshold and watch the group metrics change. Recomputed in your browser from the test-set results.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {!sweep || !live || !liveSum || !defaultSum ? <Skeleton className="h-72" /> : (
              <>
                <div className="flex items-center gap-2">
                  <Switch id="pergroup" checked={perGroup} onCheckedChange={(v) => { setPerGroup(v); if (!v) { const t = Object.values(thr)[0]; setThr(Object.fromEntries(Object.keys(thr).map((g) => [g, t]))); } }} />
                  <Label htmlFor="pergroup">Separate threshold per group</Label>
                </div>
                {perGroup && <Callout tone="warn">Group-specific thresholds are what post-processing does. They can close gaps, but they mean two students with the same score are treated differently because of their group. Use this to understand the trade-off, not as a recommendation.</Callout>}
                <div className="space-y-4">
                  {(perGroup ? Object.keys(thr) : [Object.keys(thr)[0]]).map((g) => (
                    <div key={g} className="grid grid-cols-[120px_1fr_48px] items-center gap-3 text-[13px]">
                      <span className="truncate">{perGroup ? labelOf(g) : "All groups"}</span>
                      <Slider min={0.05} max={0.95} step={0.01} value={[thr[g]]} aria-label={`Threshold for ${perGroup ? labelOf(g) : "all groups"}`}
                        onValueChange={([v]) => setThr(perGroup ? { ...thr, [g]: v } : Object.fromEntries(Object.keys(thr).map((k) => [k, v])))} />
                      <span className="text-right font-medium tnum">{thr[g].toFixed(2)}</span>
                    </div>
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {[
                    { k: "eodds", label: "Equalized odds gap", v: liveSum.eodds, d: defaultSum.eodds, lowerBetter: true },
                    { k: "spd", label: "Demographic parity gap", v: liveSum.spd, d: defaultSum.spd, lowerBetter: true },
                    { k: "f1", label: "F1", v: liveSum.f1, d: defaultSum.f1, lowerBetter: false },
                    { k: "recall", label: "Recall", v: liveSum.recall, d: defaultSum.recall, lowerBetter: false },
                  ].map((m) => {
                    const diff = m.v - m.d, good = m.lowerBetter ? diff < -0.001 : diff > 0.001, bad = m.lowerBetter ? diff > 0.001 : diff < -0.001;
                    return (
                      <div key={m.k} className="rounded-md border p-3">
                        <div className="text-[11.5px] text-muted-foreground">{m.label}</div>
                        <div className="mt-1 text-lg font-semibold tnum">{m.v.toFixed(3)}</div>
                        <div className={`text-[11.5px] tnum ${good ? "text-[#006300] dark:text-good" : bad ? "text-critical" : "text-muted-foreground"}`}>{diff >= 0 ? "+" : "−"}{Math.abs(diff).toFixed(3)} vs deployed</div>
                      </div>
                    );
                  })}
                </div>
                <ThresholdLines data={tprCurve} groups={Object.keys(sweep.deployed.groups).map((g) => ({ key: g, label: labelOf(g) }))} marker={Object.values(perGroup ? thr : { a: Object.values(thr)[0] })} yLabel="At-risk students caught" />
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle><Glossed k="preflight">Pre-flight check for this axis</Glossed></CardTitle>
            <CardDescription>Is there enough data, and a clear enough gap, for a mitigation to be measurable at all?</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {!pf ? <Skeleton className="h-64" /> : (
              <>
                <div>
                  <div className="mb-2 flex items-center justify-between"><span className="text-[13px] font-medium">Version 2 (revised)</span>
                    <Badge variant={pf.v2.verdict_code.startsWith("mitigate") ? "good" : pf.v2.verdict_code === "monitor" ? "outline" : "warn"}>{pf.v2.verdict}</Badge></div>
                  <CheckList checks={pf.v2.checks} />
                </div>
                <div className="border-t pt-5">
                  <div className="mb-2 flex items-center justify-between"><span className="text-[13px] font-medium">Version 1 (thesis)</span>
                    <Badge variant={pf.v1.passed ? "good" : "outline"}>{pf.v1.verdict}</Badge></div>
                  <CheckList checks={pf.v1.checks} />
                </div>
                <p className="text-[12px] leading-5 text-muted-foreground">v1 can&apos;t tell a fair dataset from a tiny one: both fail its variance check. v2 first asks whether the data is precise enough, then whether the gap is real, then whether there&apos;s room to improve.</p>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
