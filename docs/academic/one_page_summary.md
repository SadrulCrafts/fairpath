# FairPath: when can fairness mitigation work in student-success prediction?

**MD SADRUL ANAM** · Gannan Normal University · Undergraduate thesis (2026), extended into an open-source system
Code and live demo: `https://github.com/SadrulCrafts/fairpath` · `https://<your-demo-url>`

### The problem
Universities increasingly use predictive models to decide which students an advisor should contact first. These models are judged on
overall accuracy, which hides *who* they fail. A model that misses struggling students from one group more often than another quietly
directs less help to that group. The fairness literature offers many mitigation methods, but it mostly reports that "method M reduced
metric X on dataset D". In education, datasets are often small: one cohort, one department. That raises a question that comes first.

### Research question
**Under what conditions of data size and signal strength can fairness mitigation measurably work at all?**

### What I did
* Built one leak-free evaluation protocol and applied it to two public Portuguese datasets that differ in size: university dropout
  (n = 3,630) and secondary-school maths (n = 395). Removed the end-of-semester grade variables, which leak the outcome: with them, cross-validated AUC jumps from 0.845 to 0.954 (`scripts/leakage_check.py`).
* Compared five tuned model families and nine mitigation conditions across pre-, in- and post-processing inside one cross-validation loop,
  with bootstrap intervals, permutation tests, Holm-corrected Wilcoxon tests and the Nadeau–Bengio corrected t-test.
* Decomposed mitigation failure into four hypotheses (group size, threshold resolution, noise floor, estimate variance) and turned them into
  a **pre-flight check** that practitioners can run before attempting mitigation. When rebuilding the project I found that my original
  rule could not distinguish a fair dataset from a tiny one, and designed a revised version (v2) based on confidence-interval precision,
  a permutation test and an interval bound.
* Released it as a reproducible package with a decision-support dashboard: calibrated risk, conformal confidence, SHAP explanations,
  counterfactual "paths to lower risk" restricted to support an institution can offer, and a full fairness audit.

### Results
* **Large data, clear signal:** on the dropout data, every mitigation family significantly reduced the equalized-odds gap. The deployed
  model, which never sees gender and is trained with Reweighing, halved the gap on held-out students (0.328 → 0.150) and nearly closed the
  equal-opportunity gap (0.184 → 0.036), for a test ROC-AUC of 0.861 vs 0.864.
* **Small data:** on 395 students, the gaps sit at the level a random predictor produces; no non-degenerate method improved them, and
  the methods that reached "zero gap" did it by flagging everyone or no one. The pre-flight check flags this in advance.
* **Proxies:** removing gender reduced the dropout gap by about 40% but left most of it, carried by programme, fees and age.
* **Reproducibility:** with the original code lost, a re-implementation from the thesis text reproduced 31 of 32 headline numbers
  within 0.005, and surfaced 12 corrections that are now documented.

### Why it matters
Fairness work in education should start by asking whether a fairness claim can be *measured* on the data at hand. A cheap check
before mitigation prevents institutions from claiming improvements that are noise, and points them to where more data, or a different
kind of intervention, is needed.

### Next
Fairness under distribution shift across cohorts, causal analysis of proxy pathways, Bayesian estimation for small groups, and
student-facing explanations (see `research_directions.md`).
