"use client";
import * as React from "react";
import { PageHeader } from "@/components/app-shell";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Callout, Skeleton } from "@/components/ui/misc";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getJSON } from "@/lib/api";
import { useDataset } from "@/lib/dataset-context";
import { MODEL_LABELS, num } from "@/lib/format";

interface Repro { note: string; comparison: { table: string; dataset: string; model: string; metric: string; thesis: number; reproduced: number; diff: number; status: "match" | "close" | "differs" }[] }

const STATUS_VARIANT = { match: "good", close: "outline", differs: "warn" } as const;

export default function MethodologyPage() {
  const { meta } = useDataset();
  const [repro, setRepro] = React.useState<Repro | null>(null);
  const [tab, setTab] = React.useState("story");
  React.useEffect(() => { if (meta) getJSON<Repro>("/api/reports/thesis").then(setRepro).catch(() => setRepro(null)); }, [meta]);
  const t51 = repro?.comparison.filter((r) => r.table === "5.1") ?? [];

  return (
    <div className="animate-fade-in">
      <PageHeader eyebrow="Methodology & model card" title="How this works, and where it doesn't">
        The research behind FairPath, the decisions that went into the deployed model, an honest account of what was reproduced from
        the original thesis, and the limits anyone using it should know about.
      </PageHeader>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="flex-wrap">
          <TabsTrigger value="story">The research</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
          <TabsTrigger value="card">Model card</TabsTrigger>
          <TabsTrigger value="data">Datasheet</TabsTrigger>
          <TabsTrigger value="repro">Thesis reproduction</TabsTrigger>
        </TabsList>

        <TabsContent value="story">
          <div className="grid gap-8 lg:grid-cols-[1fr_300px]">
            <div className="prose-fp max-w-3xl">
              <p>
                This started as my undergraduate thesis at Gannan Normal University, <strong>&ldquo;Investigating Algorithmic Bias in Student Success
                Prediction Models: A Comparative Study&rdquo;</strong>. Most fairness papers ask which mitigation method lowers which metric on which dataset.
                I wanted to ask the question that comes before that: <strong>under what conditions of data size and signal strength can mitigation be expected to work at all?</strong>
              </p>
              <h2>Four research questions</h2>
              <ul>
                <li><strong>RQ1.</strong> How well do standard model families predict student success once leakage is removed, compared with trivial baselines?</li>
                <li><strong>RQ2.</strong> How do pre-processing and post-processing mitigation trade fairness against accuracy? (FairPath adds in-processing.)</li>
                <li><strong>RQ3.</strong> When and why does mitigation fail to produce a statistically reliable improvement? This is the main contribution.</li>
                <li><strong>RQ4.</strong> Is bias driven by the sensitive attribute itself, or by correlated proxy features?</li>
              </ul>
              <h2>What I found</h2>
              <p>
                On 3,630 university students, gender gaps are large, well above the noise a random predictor produces, and every mitigation family
                shrinks them measurably. On 395 school students, the same gaps sit at the noise floor. No method produced an improvement that survived
                resampling, and two very different families (Reweighing and group thresholds) failed in the same way. That points at the data, not the methods.
              </p>
              <p>
                I turned that into a <strong>pre-flight check</strong>: three numbers you can compute before spending time on mitigation. While rebuilding the
                project I found a weakness in my own rule (its variance check divides by the gap, so a genuinely fair dataset fails it just like a tiny one),
                and FairPath ships a revised version that separates <em>&ldquo;nothing to fix&rdquo;</em> from <em>&ldquo;too little data to tell&rdquo;</em>.
              </p>
              <p>
                Finally, the source of the bias. On the school data, sex sits in the lower half of the feature ranking and removing it
                changes nothing. On the university data gender matters more, and removing it narrows the gap, but more than half of the gap
                remains, carried by correlated features such as programme, tuition status and age. Removing the column does not remove the
                bias, so &ldquo;we don&apos;t use gender&rdquo; is not, on its own, a fairness guarantee.
              </p>
              <h2>Why it matters for the people involved</h2>
              <p>
                An early-warning score decides who gets an advisor&apos;s call. If the model misses at-risk students from one group more often, those students
                get less help. If it over-flags another group, they get unnecessary scrutiny and possibly a label they didn&apos;t earn. Neither failure shows up in
                an overall accuracy number, which is why everything here is broken down by group and reported with intervals.
              </p>
            </div>
            <Card className="h-fit">
              <CardHeader><CardTitle className="text-[14px]">At a glance</CardTitle></CardHeader>
              <CardContent className="space-y-3 text-[13px]">
                {[["Author", "MD SADRUL ANAM"], ["Institution", "Gannan Normal University"], ["Year", "2026"],
                  ["Datasets", "UCI Dropout (n = 3,630) · UCI Student-mat (n = 395)"], ["Models", "LR, RF, HGB, LightGBM, XGBoost"],
                  ["Mitigation", "Reweighing · Exponentiated Gradient · group thresholds · ThresholdOptimizer"], ["Licence", "MIT (code) · CC BY 4.0 (data)"]].map(([k, v]) => (
                  <div key={k}><div className="text-[11.5px] text-muted-foreground">{k}</div><div>{v}</div></div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="pipeline">
          <div className="prose-fp max-w-3xl">
            <h2>1. Data and leakage</h2>
            <p>
              For the dropout data, students still <em>enrolled</em> at the cut-off are removed, because their outcome is unknown, not negative. The twelve
              &ldquo;curricular units&rdquo; columns are dropped: they record end-of-semester results and push AUC to about 0.92, a number no real early-warning
              system could achieve at enrolment. For the school data, the in-year grades G1 and G2 are dropped for the same reason.
            </p>
            <h2>2. Splits</h2>
            <p>
              Each dataset is split once into a development set (80%) and a held-out test set (20%), stratified on outcome and group. All tuning,
              model selection, calibration and threshold choice happen inside the development set. The test set is used once, for the numbers on this site.
              Preprocessing lives inside scikit-learn pipelines, so it is refitted in every fold.
            </p>
            <h2>3. Models</h2>
            <p>
              Five families are tuned with Optuna (30 trials, 5-fold CV) and compared with 5×5 repeated stratified CV. Differences are tested with the
              Nadeau–Bengio corrected t-test, because ordinary tests on overlapping folds are badly over-confident.
            </p>
            <h2>4. The deployed model</h2>
            <p>
              The best family is retrained <strong>without the primary sensitive attribute</strong> and with <strong>Reweighing</strong>, so the institution never applies a
              different rule to a student because of their group. Probabilities are calibrated on out-of-fold predictions (isotonic for the larger dataset,
              Platt scaling for the small one), the outreach threshold maximises out-of-fold F1, and split-conformal prediction sets give the confidence indicator.
            </p>
            <h2>5. Fairness measurement</h2>
            <p>
              Per group: flag rate, recall (TPR), false-positive rate, precision and calibration. Gaps use max minus min across groups. Every gap gets a
              stratified bootstrap interval (1,000 resamples) and a permutation-test p-value (2,000 permutations). Groups under 100 students are marked
              &ldquo;caution&rdquo;, under 30 &ldquo;unreliable&rdquo;.
            </p>
            <h2>6. Mitigation study</h2>
            <p>
              Nine conditions inside one leak-free 5×2 CV loop: baseline, blind, Reweighing, blind + Reweighing, Exponentiated Gradient (demographic parity and
              equalized odds), the thesis&apos; group-threshold search, and Fairlearn&apos;s ThresholdOptimizer (both constraints). Post-processing is tuned on an
              inner validation split that never overlaps the test fold.
            </p>
            <h2>7. Explanations</h2>
            <p>
              SHAP values are computed on the model&apos;s transformed inputs and summed back to the original features (exact, because SHAP is additive).
              Counterfactuals search only features marked as realistically changeable (tuition support, scholarships, tutoring, study time, attendance)
              and never touch gender, age, nationality, family background or past grades.
            </p>
          </div>
        </TabsContent>

        <TabsContent value="card">
          <div className="grid gap-4 lg:grid-cols-2">
            {[
              { t: "Intended use", b: "Help advisors decide whom to contact first with an offer of support, at the start of a programme or school year. A person who knows the student reviews every flag." },
              { t: "Out of scope", b: "Admissions, scholarships, discipline, grading, or any decision that withholds an opportunity. Any fully automated action. Use on populations very different from Portuguese higher or secondary education without re-validation." },
              { t: "Users", b: "Student-success teams, advisors and researchers. Not students themselves: an individual risk score should not be shown to the student it describes." },
              { t: "Factors", b: "Audited by gender / sex, debtor status, age band, home address and parental education, plus intersections where groups are large enough." },
              { t: "Limitations", b: "Two Portuguese datasets from 2005–2019. Binary outcomes hide degrees of success. Labels reflect past institutional practice. Debtor and tuition status may be recorded after enrolment. The school dataset is too small for reliable group conclusions." },
              { t: "Ethical considerations", b: "False negatives mean missed help; false positives mean unwarranted attention and possible self-fulfilling labels. Students should be told a score exists, be able to ask for a human-only review, and get its main reasons in plain language." },
            ].map((c) => (
              <Card key={c.t}><CardHeader><CardTitle>{c.t}</CardTitle></CardHeader><CardContent className="text-[13.5px] leading-6 text-muted-foreground">{c.b}</CardContent></Card>
            ))}
          </div>
          <Callout className="mt-4" title="Privacy by design">The API is stateless: inputs and uploaded files are processed in memory and discarded. Example students on this site are synthetic.</Callout>
        </TabsContent>

        <TabsContent value="data">
          <div className="grid gap-4 lg:grid-cols-2">
            {meta?.datasets.map((d) => (
              <Card key={d.name}>
                <CardHeader><CardTitle>{d.title}</CardTitle></CardHeader>
                <CardContent className="space-y-3 text-[13.5px] leading-6 text-muted-foreground">
                  <p>{d.description}</p>
                  <div><span className="font-medium text-foreground">Outcome:</span> {d.at_risk_label} vs {d.success_label.toLowerCase()}.</div>
                  <div><span className="font-medium text-foreground">Licence:</span> {d.licence}. <a className="underline underline-offset-4" href={d.source_url} target="_blank" rel="noreferrer">UCI repository</a></div>
                  <div className="rounded-md bg-muted/60 p-3 text-[12px] leading-5">{d.citation}</div>
                  <div className="text-[12px]">Raw data is not redistributed in the repository; a download script fetches it from UCI, and a clearly labelled synthetic copy lets anyone run the code without it.</div>
                </CardContent>
              </Card>
            )) ?? <Skeleton className="h-64" />}
          </div>
        </TabsContent>

        <TabsContent value="repro">
          <div className="prose-fp mb-6 max-w-3xl">
            <p>
              The original thesis code was lost, so everything was rebuilt from the thesis text. To check the rebuild, a separate harness re-runs the
              <strong> original protocol</strong> (success as the positive class, the original hyper-parameters, 5×2 CV, seed 42) and compares the result with
              every number printed in the thesis.
              {t51.length > 0 && <> {t51.filter((r) => r.status === "match").length} of {t51.length} values in the main performance table (Table 5.1) are reproduced to within 0.005.</>}
            </p>
            <p>
              The post-processing rows of Table 6.1 differ more. The thesis ran that table with only 5 folds and a coarse 10-point threshold grid on validation
              splits of around 60 students, and that is exactly the instability the thesis itself diagnoses. Re-running with 10 folds moves those numbers by more
              than their own standard deviation, which is the point of RQ3.
            </p>
          </div>
          <Card>
            <CardContent className="max-h-[560px] overflow-auto p-0">
              {!repro ? <Skeleton className="m-4 h-64" /> : (
                <table className="w-full text-[12.5px] tnum">
                  <thead className="sticky top-0 bg-card text-left text-muted-foreground shadow-[0_1px_0_hsl(var(--border))]">
                    <tr><th className="px-4 py-2 font-medium">Table</th><th className="px-2 font-medium">Dataset</th><th className="px-2 font-medium">Model</th><th className="px-2 font-medium">Metric</th><th className="px-2 font-medium">Thesis</th><th className="px-2 font-medium">Re-computed</th><th className="px-4 font-medium">Status</th></tr>
                  </thead>
                  <tbody>
                    {repro.comparison.map((r, i) => (
                      <tr key={i} className="border-b last:border-0">
                        <td className="px-4 py-1.5">{r.table}</td><td className="px-2">{r.dataset}</td><td className="px-2">{MODEL_LABELS[r.model] ?? r.model}</td>
                        <td className="px-2">{r.metric}</td><td className="px-2">{num(r.thesis)}</td><td className="px-2">{num(r.reproduced)}</td>
                        <td className="px-4"><Badge variant={STATUS_VARIANT[r.status]}>{r.status}</Badge></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardContent>
          </Card>
          <p className="mt-3 text-[12px] text-muted-foreground">{repro?.note} A full list of corrections to the thesis text is in docs/REPRODUCIBILITY.md in the repository.</p>
        </TabsContent>
      </Tabs>
    </div>
  );
}
