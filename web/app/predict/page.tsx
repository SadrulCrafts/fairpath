"use client";
import * as React from "react";
import { Loader2, RotateCcw, UserRound, Wand2 } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Callout, EmptyState, Skeleton } from "@/components/ui/misc";
import { StudentForm, defaultsFrom, type Values } from "@/components/student-form";
import { PredictionPanel } from "@/components/widgets";
import { ApiError, getJSON, postJSON } from "@/lib/api";
import { useData, useDataset } from "@/lib/dataset-context";
import type { Example, Prediction, Schema, ValidationIssue } from "@/lib/types";
import { pct } from "@/lib/format";

export default function PredictPage() {
  const { current, dataset } = useDataset();
  const schema = useData((ds) => getJSON<Schema>(`/api/datasets/${ds}/schema`));
  const examples = useData((ds) => getJSON<Example[]>(`/api/datasets/${ds}/examples`));
  const [values, setValues] = React.useState<Values>({});
  const [result, setResult] = React.useState<Prediction | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<{ message: string; issues?: ValidationIssue[] } | null>(null);
  const resultRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (schema.data) { setValues(defaultsFrom(schema.data)); setResult(null); setError(null); }
  }, [schema.data]);

  const run = async (v: Values = values) => {
    setLoading(true); setError(null);
    try {
      const features = Object.fromEntries(Object.entries(v).filter(([k]) => schema.data?.fields.find((f) => f.name === k)?.in_model));
      setResult(await postJSON<Prediction>(`/api/datasets/${dataset}/predict`, { features, explain: true, counterfactual: true }));
      // on small screens the result sits below the form: bring it into view
      if (window.innerWidth < 1024) setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e) {
      const err = e as ApiError;
      setError({ message: err.message, issues: err.details as ValidationIssue[] | undefined });
    } finally { setLoading(false); }
  };

  const loadExample = (ex: Example) => {
    const v = { ...values };
    for (const [k, val] of Object.entries(ex.record)) if (k in v) v[k] = val;
    setValues(v);
    run(v);
  };

  return (
    <div className="animate-fade-in">
      <PageHeader eyebrow="Single student" title="Score one student">
        Fill in what you know at enrolment. Missing details default to a typical student. Nothing you enter is stored.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[12.5px] text-muted-foreground">Load an example:</span>
            {examples.data?.map((ex) => (
              <Button key={ex.name} variant="outline" size="sm" onClick={() => loadExample(ex)} title={`Synthetic profile, risk about ${pct(ex.risk)}`}>
                {ex.name.replace(" example", "")}
              </Button>
            ))}
            <Button variant="ghost" size="sm" onClick={() => schema.data && (setValues(defaultsFrom(schema.data)), setResult(null))}>
              <RotateCcw /> Reset
            </Button>
          </div>
          {schema.loading || !schema.data ? <Skeleton className="h-[520px]" /> : (
            <StudentForm schema={schema.data} values={values} onChange={setValues} />
          )}
          <div className="sticky bottom-4 z-10">
            <Button variant="brand" size="lg" className="w-full shadow-lg" onClick={() => run()} disabled={loading || !schema.data}>
              {loading ? <Loader2 className="animate-spin" /> : <Wand2 />} {loading ? "Scoring…" : "Estimate risk"}
            </Button>
          </div>
          <p className="text-[11.5px] leading-5 text-muted-foreground">Example profiles are synthetic, generated to look like the {current?.short_title.toLowerCase()} data. They are not real students.</p>
        </div>

        <div className="min-w-0 scroll-mt-16" ref={resultRef}>
          {error && (
            <Callout tone="warn" title="Couldn't score this student" className="mb-4">
              {error.message}
              {error.issues && (
                <ul className="mt-2 list-disc pl-5">
                  {error.issues.slice(0, 6).map((i, k) => <li key={k}><code>{i.field}</code>: {i.message}</li>)}
                </ul>
              )}
            </Callout>
          )}
          {loading && !result && <div className="space-y-4"><Skeleton className="h-56" /><Skeleton className="h-72" /><Skeleton className="h-64" /></div>}
          {!loading && !result && !error && (
            <EmptyState icon={UserRound} title="No prediction yet" className="lg:mt-11">
              Pick an example or fill in the form, then press <strong>Estimate risk</strong>. You&apos;ll see the risk, how sure the model is, what drove the score and what support might help.
            </EmptyState>
          )}
          {result && current && <div className={loading ? "opacity-60 transition-opacity" : "transition-opacity"}><PredictionPanel p={result} bands={current.bands} /></div>}
          {result && <p className="mt-4 text-[11.5px] leading-5 text-muted-foreground">{result.disclaimer}</p>}
        </div>
      </div>
    </div>
  );
}
