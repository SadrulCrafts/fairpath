"use client";
import * as React from "react";
import { Trophy } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Callout, Skeleton } from "@/components/ui/misc";
import { ForestPlot, ReliabilityChart } from "@/components/charts";
import { Glossed } from "@/components/widgets";
import { getJSON } from "@/lib/api";
import { useData } from "@/lib/dataset-context";
import { MODEL_LABELS, num, pval } from "@/lib/format";
import type { Stat } from "@/lib/types";

interface Row { model: string; label: string; params: Record<string, unknown>; tuning_cv_auc: number | null; [k: string]: unknown }
interface Bench {
  cv: string; n_dev: number; best_model: string; threshold_note: string; summary: Row[];
  comparisons: { model: string; vs: string; corrected_t: { p_value: number; p_holm: number; mean_diff: number }; wilcoxon: { p_value: number; p_holm: number } }[];
  calibration: Record<string, { mean_pred: number; frac_pos: number; n: number }[]>;
}

const COLS: { key: string; label: string; g?: string; lower?: boolean }[] = [
  { key: "roc_auc", label: "ROC-AUC", g: "roc_auc" },
  { key: "pr_auc", label: "PR-AUC", g: "pr_auc" },
  { key: "brier", label: "Brier", g: "brier", lower: true },
  { key: "ece", label: "ECE", g: "ece", lower: true },
  { key: "f1_rm", label: "F1", g: "f1" },
  { key: "recall_rm", label: "Recall", g: "recall" },
  { key: "spd_rm", label: "SPD", g: "spd", lower: true },
  { key: "eodds_rm", label: "EOdds", g: "eodds", lower: true },
];

export default function ModelsPage() {
  const bench = useData((ds) => getJSON<Bench>(`/api/reports/${ds}/benchmark`));
  const b = bench.data;
  const real = b?.summary.filter((r) => !r.model.startsWith("dummy")) ?? [];
  const best: Record<string, number> = {};
  if (b) for (const c of COLS) {
    const vals = real.map((r) => (r[c.key] as Stat)?.mean).filter((v) => v != null);
    best[c.key] = c.lower ? Math.min(...vals) : Math.max(...vals);
  }
  const [showCal, setShowCal] = React.useState<string[]>([]);
  React.useEffect(() => { if (b) setShowCal([b.best_model, "logistic_regression"].filter((v, i, a) => a.indexOf(v) === i)); }, [b]);

  return (
    <div className="animate-fade-in">
      <PageHeader eyebrow="Model comparison" title="Five model families, tuned and compared fairly">
        Each family was tuned with Optuna on the development split, then scored with {b?.cv ?? "5x5 repeated cross-validation"}.
        Intervals use the Nadeau–Bengio correction, because overlapping CV folds make naive intervals far too narrow.
      </PageHeader>

      {!b ? <Skeleton className="h-96" /> : (
        <>
          <Card>
            <CardContent className="overflow-x-auto p-0">
              <table className="w-full min-w-[760px] text-[13px] tnum">
                <thead>
                  <tr className="border-b text-left text-[12px] text-muted-foreground">
                    <th className="px-4 py-3 font-medium">Model</th>
                    {COLS.map((c) => <th key={c.key} className="px-2 py-3 font-medium">{c.g ? <Glossed k={c.g}>{c.label}</Glossed> : c.label}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {b.summary.map((r) => (
                    <tr key={r.model} className={`border-b last:border-0 ${r.model.startsWith("dummy") ? "text-muted-foreground" : ""}`}>
                      <td className="px-4 py-2.5 font-medium">
                        <span className="inline-flex items-center gap-2">{r.label}{r.model === b.best_model && <Badge variant="brand"><Trophy /> best AUC</Badge>}</span>
                      </td>
                      {COLS.map((c) => {
                        const s = r[c.key] as Stat;
                        const isBest = !r.model.startsWith("dummy") && s && Math.abs(s.mean - best[c.key]) < 1e-9;
                        return (
                          <td key={c.key} className={`px-2 py-2.5 ${isBest ? "font-semibold text-foreground" : ""}`} title={s?.lo != null ? `95% CI ${num(s.lo)} – ${num(s.hi)}` : undefined}>
                            {num(s?.mean)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
          <p className="mt-2 text-[12px] leading-5 text-muted-foreground">
            Mean over folds on {b.n_dev.toLocaleString()} development students; hover a cell for its interval. Threshold-based columns (F1, recall, SPD, EOdds)
            use a rate-matched cut-off chosen on each training fold, so every model flags about the same number of students. The dummy rows are the noise floor.
          </p>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Ranking quality with honest intervals</CardTitle>
                <CardDescription>ROC-AUC, 95% corrected intervals. Heavy overlap means the models are statistically tied.</CardDescription>
              </CardHeader>
              <CardContent>
                <ForestPlot series={["ROC-AUC"]} digits={3}
                  min={Math.floor((Math.min(...real.map((r) => ((r.roc_auc as Stat).lo ?? 0.5))) - 0.01) * 100) / 100}
                  max={Math.ceil((Math.max(...real.map((r) => ((r.roc_auc as Stat).hi ?? 1))) + 0.01) * 100) / 100}
                  rows={real.map((r) => { const s = r.roc_auc as Stat; return { label: r.label, values: [{ value: s.mean, lo: s.lo ?? null, hi: s.hi ?? null }] }; })} />
                <div className="mt-5 space-y-1.5">
                  <div className="text-[12.5px] font-medium">Is the best model really better? (vs {MODEL_LABELS[b.best_model]})</div>
                  <table className="w-full text-[12.5px] tnum">
                    <thead><tr className="border-b text-left text-muted-foreground"><th className="py-1.5 font-medium">Model</th><th className="font-medium">ΔAUC</th><th className="font-medium">Corrected t, p (Holm)</th><th className="font-medium">Naive Wilcoxon, p (Holm)</th></tr></thead>
                    <tbody>
                      {b.comparisons.map((c) => (
                        <tr key={c.model} className="border-b last:border-0">
                          <td className="py-1.5">{MODEL_LABELS[c.model]}</td>
                          <td>{num(-c.corrected_t.mean_diff, 4)}</td>
                          <td className={c.corrected_t.p_holm < 0.05 ? "font-medium" : ""}>{pval(c.corrected_t.p_holm)}</td>
                          <td className="text-muted-foreground">{pval(c.wilcoxon.p_holm)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="pt-2 text-[12px] leading-5 text-muted-foreground">
                    The naive test treats 25 overlapping folds as independent and often calls tiny differences &ldquo;significant&rdquo;. The corrected test is the one to believe.
                  </p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle><Glossed k="ece">Calibration before post-hoc correction</Glossed></CardTitle>
                <CardDescription>Out-of-fold predictions on the development split. The deployed model is recalibrated afterwards.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-3 flex flex-wrap gap-1.5">
                  {real.map((r) => (
                    <button key={r.model} onClick={() => setShowCal((s) => s.includes(r.model) ? s.filter((x) => x !== r.model) : [...s, r.model].slice(-3))}
                      className={`rounded-full border px-2.5 py-0.5 text-[12px] ${showCal.includes(r.model) ? "border-foreground/40 bg-accent font-medium" : "text-muted-foreground"}`}>
                      {r.label}
                    </button>
                  ))}
                </div>
                <ReliabilityChart series={showCal.map((m) => ({ name: MODEL_LABELS[m], rows: b.calibration[m] ?? [] }))} />
                <p className="text-[11.5px] text-muted-foreground">Up to three models at a time.</p>
              </CardContent>
            </Card>
          </div>

          <Card className="mt-6">
            <CardHeader><CardTitle>Tuned hyper-parameters</CardTitle><CardDescription>Best Optuna trial (TPE sampler, median pruning) by mean 5-fold ROC-AUC.</CardDescription></CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {real.map((r) => (
                <div key={r.model} className="rounded-md border p-3 text-[12px]">
                  <div className="mb-1.5 flex justify-between font-medium"><span>{r.label}</span><span className="text-muted-foreground tnum">{r.tuning_cv_auc ? num(r.tuning_cv_auc) : ""}</span></div>
                  <div className="space-y-0.5 font-mono text-[11.5px] text-muted-foreground">
                    {Object.entries(r.params).map(([k, v]) => <div key={k} className="flex justify-between gap-2"><span>{k}</span><span className="truncate">{typeof v === "number" ? Number(v.toPrecision(4)) : String(v)}</span></div>)}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
          <Callout className="mt-6">
            Model choice matters less than people expect on tabular data like this: the five families sit within a few thousandths of AUC of each other.
            What changes outcomes for students is the threshold, the calibration and the fairness treatment, which is where the rest of this dashboard focuses.
          </Callout>
        </>
      )}
    </div>
  );
}
