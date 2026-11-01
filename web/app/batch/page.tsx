"use client";
import * as React from "react";
import { Download, FileDown, FileSpreadsheet, Loader2, ShieldCheck, Upload } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Callout, EmptyState } from "@/components/ui/misc";
import { ConfidenceBadge, KpiCard } from "@/components/widgets";
import { api, postFile, useSlow } from "@/lib/api";
import { useDataset } from "@/lib/dataset-context";
import { bandMeta, pct } from "@/lib/format";
import type { Band, ValidationIssue } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Row { row: number; student_id: string | null; risk: number; band: Band; flagged: boolean; confidence: "confident" | "uncertain" | "atypical"; top_factors: string[] }
interface Result { n: number; summary: { bands: Partial<Record<Band, number>>; flagged: number; uncertain: number; mean_risk: number; threshold: number }; rows: Row[]; disclaimer: string }

function download(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function BatchPage() {
  const { dataset, current } = useDataset();
  const [file, setFile] = React.useState<File | null>(null);
  const [result, setResult] = React.useState<Result | null>(null);
  const [issues, setIssues] = React.useState<ValidationIssue[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState<"score" | "pdf" | null>(null);
  const [drag, setDrag] = React.useState(false);
  const [filter, setFilter] = React.useState<"all" | "flagged" | "uncertain">("all");
  const slow = useSlow(busy !== null);

  React.useEffect(() => { setResult(null); setIssues(null); setError(null); setFile(null); }, [dataset]);

  const score = async (f: File) => {
    setFile(f); setBusy("score"); setError(null); setIssues(null); setResult(null);
    try {
      const res = await postFile(`/api/datasets/${dataset}/batch`, f);
      const body = await res.json();
      if (res.status === 422) { setIssues(body.errors); return; }
      if (!res.ok) throw new Error(body.detail ?? `Request failed (${res.status})`);
      setResult(body);
    } catch (e) { setError((e as Error).message); } finally { setBusy(null); }
  };

  const pdf = async () => {
    if (!file) return;
    setBusy("pdf");
    try {
      const res = await postFile(`/api/datasets/${dataset}/batch/report.pdf`, file);
      if (!res.ok) throw new Error(`PDF failed (${res.status})`);
      download(`fairpath_${dataset}_report.pdf`, await res.blob());
    } catch (e) { setError((e as Error).message); } finally { setBusy(null); }
  };

  const csv = () => {
    if (!result) return;
    const head = "row,student_id,risk,band,flagged,confidence,top_factors\n";
    const body = result.rows.map((r) => [r.row + 1, r.student_id ?? "", r.risk.toFixed(4), r.band, r.flagged ? 1 : 0, r.confidence, `"${r.top_factors.join("; ")}"`].join(",")).join("\n");
    download(`fairpath_${dataset}_scores.csv`, new Blob([head + body], { type: "text/csv" }));
  };

  const rows = (result?.rows ?? []).filter((r) => filter === "all" || (filter === "flagged" ? r.flagged : r.confidence === "uncertain")).sort((a, b) => b.risk - a.risk);

  return (
    <div className="animate-fade-in">
      <PageHeader title="Score a whole cohort"
        actions={<a href={api.url(`/api/datasets/${dataset}/template.csv`)} download><Button variant="outline" size="sm"><FileSpreadsheet /> Download template</Button></a>}>
        Upload a CSV with one student per row. The columns are checked against the model&apos;s schema before anything is scored.
        An optional <code className="rounded bg-muted px-1 text-[13px]">student_id</code> column is passed through so you can match results back.
      </PageHeader>

      <label
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) score(f); }}
        className={cn("flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed bg-card px-6 py-10 text-center transition-colors", drag ? "border-primary bg-brand-soft" : "hover:bg-accent/40")}
      >
        <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f) score(f); e.target.value = ""; }} />
        {busy === "score" ? <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /> : <Upload className="h-6 w-6 text-muted-foreground" />}
        <div className="text-[14px] font-medium">{file ? file.name : "Drop a CSV here, or click to choose"}</div>
        <div className="text-[12.5px] text-muted-foreground">Up to 5,000 rows · 2 MB · comma or semicolon separated</div>
        <div className="mt-1 inline-flex items-center gap-1.5 text-[12px] text-muted-foreground"><ShieldCheck className="h-3.5 w-3.5" /> Processed in memory and discarded. Nothing is stored.</div>
      </label>

      {slow && <Callout className="mt-4" title="Still working…">If the demo server was asleep it can take up to a minute to wake up. Large files also take a few seconds to explain row by row.</Callout>}
      {error && <Callout tone="warn" className="mt-4" title="Something went wrong">{error}</Callout>}
      {issues && (
        <Callout tone="warn" className="mt-4" title={`The file doesn't match the ${current?.short_title} schema`}>
          <ul className="mt-1 max-h-48 list-disc overflow-auto pl-5">
            {issues.slice(0, 25).map((i, k) => <li key={k}>{i.row != null && <>Row {i.row + 1}, </>}<code>{i.field}</code>: {i.message}</li>)}
          </ul>
          {issues.length > 25 && <p className="mt-1">…and {issues.length - 25} more. Start from the template to get the columns right.</p>}
        </Callout>
      )}

      {!result && !issues && !error && busy !== "score" && (
        <EmptyState icon={FileSpreadsheet} title="No file scored yet" className="mt-6">Download the template to see the expected columns and value formats.</EmptyState>
      )}

      {result && (
        <div className="mt-8 space-y-5 animate-fade-in">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <KpiCard label="Students scored" value={result.n.toLocaleString()} sub={`mean risk ${pct(result.summary.mean_risk)}`} />
            <KpiCard label="Suggested for outreach" value={result.summary.flagged} sub={`risk at or above ${pct(result.summary.threshold)}`} />
            <KpiCard label="High / moderate / low" value={<span className="tnum">{result.summary.bands.high ?? 0} / {result.summary.bands.moderate ?? 0} / {result.summary.bands.low ?? 0}</span>} />
            <KpiCard label="Uncertain predictions" glossary="confidence" value={result.summary.uncertain} sub="review these by hand" />
          </div>
          <Card>
            <div className="flex flex-col gap-3 border-b p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex gap-1 rounded-lg bg-muted p-1 text-[12.5px]">
                {(["all", "flagged", "uncertain"] as const).map((f) => (
                  <button key={f} onClick={() => setFilter(f)} className={cn("rounded-md px-3 py-1 capitalize", filter === f ? "bg-card font-medium shadow-sm" : "text-muted-foreground")}>{f}</button>
                ))}
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={csv}><Download /> CSV</Button>
                <Button variant="outline" size="sm" onClick={pdf} disabled={busy === "pdf"}>{busy === "pdf" ? <Loader2 className="animate-spin" /> : <FileDown />} PDF report</Button>
              </div>
            </div>
            <CardContent className="max-h-[560px] overflow-auto p-0">
              <table className="w-full text-[13px]">
                <thead className="sticky top-0 bg-card text-left text-[12px] text-muted-foreground shadow-[0_1px_0_hsl(var(--border))]">
                  <tr><th className="px-4 py-2.5 font-medium">#</th><th className="px-2 py-2.5 font-medium">Student</th><th className="px-2 py-2.5 text-right font-medium">Risk</th><th className="px-3 py-2.5 font-medium">Band</th><th className="px-2 py-2.5 font-medium">Confidence</th><th className="px-4 py-2.5 font-medium">Main factors raising risk</th></tr>
                </thead>
                <tbody>
                  {rows.slice(0, 500).map((r) => (
                    <tr key={r.row} className="border-b last:border-0 hover:bg-accent/30">
                      <td className="px-4 py-2 text-muted-foreground tnum">{r.row + 1}</td>
                      <td className="px-2 py-2">{r.student_id ?? "–"}</td>
                      <td className="px-2 py-2 text-right font-medium tnum">{pct(r.risk)}</td>
                      <td className="px-3 py-2"><Badge variant={bandMeta[r.band].variant}>{bandMeta[r.band].label.replace(" risk", "")}</Badge></td>
                      <td className="px-2 py-2"><ConfidenceBadge level={r.confidence} /></td>
                      <td className="px-4 py-2 text-muted-foreground">{r.top_factors.join(", ") || "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length === 0 && <div className="p-8 text-center text-[13px] text-muted-foreground">No rows match this filter.</div>}
            </CardContent>
          </Card>
          <p className="text-[11.5px] leading-5 text-muted-foreground">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
