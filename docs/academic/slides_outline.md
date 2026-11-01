# Talk outline (10 slides, about 10 minutes)

The deck itself is published as a Slides artifact. This file is the script behind it: what each slide shows and what to say.

| # | Slide | What to say (roughly a minute each) |
|---|---|---|
| 1 | **Title.** When can fairness mitigation work? | Who I am, one sentence on the thesis, one sentence on what I built after it. |
| 2 | **The problem.** Early-warning models decide who gets help first. | Accuracy is an average. If a model misses struggling students from one group more often, that group quietly receives less support. |
| 3 | **The gap in the literature.** | Papers report that method M reduced metric X on dataset D. Educational data is often small. Before asking *which* method works, ask whether a fairness change can be *measured* at all. |
| 4 | **Data and protocol.** Dropout (3,630) vs student-mat (395). | Two public datasets from Portugal that differ mainly in size. I removed the end-of-semester grade columns, which leak the outcome (AUC 0.954 with them vs 0.845 without). Five tuned model families, nine mitigation conditions, one cross-validation loop, proper paired tests. |
| 5 | **Result 1: on large data, mitigation works.** Pareto chart. | On dropout, all eight mitigation conditions cut the equalized-odds gap significantly versus the baseline (Holm-corrected p = 0.016 each). Some cost a lot of F1 (thesis thresholds: 0.558), some very little (blind + reweighing: 0.720 vs 0.734). |
| 6 | **Result 2: on small data, it can't be shown.** | On 395 students no non-degenerate method changed the gap. The methods that reached a gap of exactly zero did it by flagging no one or everyone. A "perfectly fair" model that helps nobody is a warning sign, so FairPath detects and excludes these. |
| 7 | **Why: four hypotheses → a pre-flight check.** | Group size, threshold resolution, noise floor and estimate variance. I turned them into three questions: is the estimate precise, is the gap real, is there room to improve. Dropout × gender: mitigate. Student-mat × sex: insufficient data. A gap that is clearly real but measured on small groups gets a fourth verdict, "coarse": worth mitigating, but only large changes can be confirmed. |
| 8 | **The deployed model.** Held-out numbers. | Never sees gender, trained with Reweighing. Equalized-odds gap 0.328 → 0.150, equal-opportunity gap 0.184 → 0.036, ROC-AUC 0.864 → 0.861. Removing gender alone cuts the gap by about 40%; the rest is carried by proxies. |
| 9 | **The system.** Dashboard screenshots. | Calibrated risk with a conformal confidence label, SHAP explanation, counterfactual "support options" restricted to things an institution can change, batch scoring, and a full public fairness audit. Designed as decision support: it suggests who to talk to, never what to decide. |
| 10 | **What I'd do next.** | Fairness under cohort shift, causal proxy pathways, Bayesian estimates for small groups, and a user study with advisors. These map directly to MSc/PhD proposals. Thank you, and the link. |

## Timing tips

* Spend the most time on slides 5–7. They hold the contribution.
* On slide 6, pause after "flagging everyone". It is the moment people remember.
* If you have only 3 minutes: slides 2, 5, 6, 7, 10.
