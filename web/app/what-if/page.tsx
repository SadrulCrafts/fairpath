"use client";
import * as React from "react";
import { ArrowDownRight, ArrowUpRight, Loader2, RotateCcw } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Callout, Skeleton } from "@/components/ui/misc";
import { Select } from "@/components/ui/select";
import { FieldInput, defaultsFrom, type Values } from "@/components/student-form";
import { ConfidenceBadge, Glossed, RiskGauge, ShapBars } from "@/components/widgets";
import { getJSON, postJSON } from "@/lib/api";
import { useData, useDataset } from "@/lib/dataset-context";
import { pct } from "@/lib/format";
import type { Example, Field, Prediction, Schema } from "@/lib/types";

function pickFields(schema: Schema): { levers: Field[]; context: Field[] } {
  const inModel = schema.fields.filter((f) => f.in_model);
  const levers = inModel.filter((f) => f.actionable);
  // protected characteristics (age, special needs, nationality...) are deliberately not adjustable here
  const context = inModel.filter((f) => !f.actionable && !f.protected && f.type !== "categorical" && !["unemployment_rate", "inflation_rate", "gdp"].includes(f.name)).slice(0, 8);
  return { levers, context };
}

export default function WhatIfPage() {
  const { dataset, current } = useDataset();
  const schema = useData((ds) => getJSON<Schema>(`/api/datasets/${ds}/schema`));
  const examples = useData((ds) => getJSON<Example[]>(`/api/datasets/${ds}/examples`));
  const [start, setStart] = React.useState<string>("typical");
  const [base, setBase] = React.useState<Values>({});
  const [values, setValues] = React.useState<Values>({});
  const [baseRisk, setBaseRisk] = React.useState<number | null>(null);
  const [pred, setPred] = React.useState<Prediction | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const abort = React.useRef<AbortController | null>(null);

  const call = React.useCallback(async (v: Values) => {
    if (!schema.data) return null;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    const features = Object.fromEntries(Object.entries(v).filter(([k]) => schema.data!.fields.find((f) => f.name === k)?.in_model));
    return postJSON<Prediction>(`/api/datasets/${dataset}/whatif`, { features, explain: true, counterfactual: false }, ctrl.signal);
  }, [dataset, schema.data]);

  // choose a starting student
  React.useEffect(() => {
    if (!schema.data) return;
    let v = defaultsFrom(schema.data);
    const ex = examples.data?.find((e) => e.name === start);
    if (ex) v = { ...v, ...Object.fromEntries(Object.entries(ex.record).filter(([k]) => k in v)) };
    setBase(v); setValues(v); setError(null);
    call(v).then((p) => { if (p) { setBaseRisk(p.risk); setPred(p); } }).catch((e) => e.name !== "AbortError" && setError(e.message));
  }, [schema.data, examples.data, start, call]);

  // live updates, debounced
  React.useEffect(() => {
    if (!schema.data || values === base) { setBusy(false); return; }
    setBusy(true);
    const t = setTimeout(() => {
      call(values).then((p) => p && setPred(p)).catch((e) => e.name !== "AbortError" && setError(e.message)).finally(() => setBusy(false));
    }, 220);
    return () => { clearTimeout(t); setBusy(false); };
  }, [values, base, call, schema.data]);

  const fields = schema.data ? pickFields(schema.data) : null;
  const delta = pred && baseRisk != null ? pred.risk - baseRisk : 0;
  const changed = Object.keys(values).filter((k) => values[k] !== base[k]);

  return (
    <div className="animate-fade-in">
      <PageHeader eyebrow="What-if simulator" title="See how the risk responds"
        actions={<Button variant="outline" size="sm" onClick={() => setValues(base)} disabled={!changed.length}><RotateCcw /> Undo changes</Button>}>
        Start from a student, then move the levers an institution can actually influence. The score updates as you go. Treat this as a way
        to understand the model, not as a forecast of what an intervention will achieve.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          <Card>
            <CardContent className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
              <span className="text-[13px] font-medium">Starting student</span>
              <div className="sm:w-72">
                <Select ariaLabel="Starting student" value={start} onValueChange={setStart}
                  options={[{ value: "typical", label: "A typical student" }, ...(examples.data ?? []).map((e) => ({ value: e.name, label: `${e.name} (synthetic)` }))]} />
              </div>
            </CardContent>
          </Card>
          {!fields ? <Skeleton className="h-96" /> : (
            <>
              <Card>
                <CardHeader>
                  <CardTitle>Things support could change</CardTitle>
                  <CardDescription>Only features marked as realistically changeable in the dataset configuration.</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-5 sm:grid-cols-2">
                  {fields.levers.map((f) => (
                    <FieldInput key={f.name} field={f} value={values[f.name] ?? f.default} slider onChange={(v) => setValues({ ...values, [f.name]: v })} />
                  ))}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Context</CardTitle>
                  <CardDescription>Facts about the student&apos;s record. Useful for exploring the model, but not things to &ldquo;fix&rdquo;. Personal characteristics such as age, gender, nationality or special needs are intentionally not adjustable.</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-5 sm:grid-cols-2">
                  {fields.context.map((f) => (
                    <FieldInput key={f.name} field={f} value={values[f.name] ?? f.default} slider={f.type === "numeric"} onChange={(v) => setValues({ ...values, [f.name]: v })} />
                  ))}
                </CardContent>
              </Card>
            </>
          )}
        </div>

        <div className="space-y-4 lg:sticky lg:top-4 lg:self-start">
          {error && <Callout tone="warn">{error}</Callout>}
          <Card>
            <CardContent className="p-6">
              {!pred || !current ? <Skeleton className="h-48" /> : (
                <>
                  <div className="relative">
                    <RiskGauge risk={pred.risk} band={pred.band} bands={current.bands} threshold={pred.threshold} />
                    {busy && <Loader2 className="absolute right-0 top-0 h-4 w-4 animate-spin text-muted-foreground" />}
                  </div>
                  <div className="mt-5 flex items-center justify-between border-t pt-4 text-[13px]">
                    <span className="text-muted-foreground">Starting risk {pct(baseRisk)}</span>
                    {Math.abs(delta) < 0.005 ? <Badge variant="outline">No change</Badge> : delta < 0
                      ? <Badge variant="good"><ArrowDownRight />{pct(-delta, 1)} lower</Badge>
                      : <Badge variant="critical"><ArrowUpRight />{pct(delta, 1)} higher</Badge>}
                  </div>
                  <div className="mt-3 flex items-center gap-2 text-[12.5px] text-muted-foreground">
                    <ConfidenceBadge level={pred.confidence.level} /> <Glossed k="confidence">at 90% coverage</Glossed>
                  </div>
                  {changed.length > 0 && <p className="mt-3 text-[12px] text-muted-foreground">{changed.length} field{changed.length > 1 ? "s" : ""} changed from the starting student.</p>}
                </>
              )}
            </CardContent>
          </Card>
          {pred?.explanation && (
            <Card>
              <CardHeader><CardTitle className="text-[14px]">Top factors right now</CardTitle></CardHeader>
              <CardContent><ShapBars items={pred.explanation.top.slice(0, 5)} /></CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
