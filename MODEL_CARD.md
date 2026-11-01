# Model card: FairPath early-warning models

Format follows Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). Every number below comes from
`reports/<dataset>/final_model.json` (held-out test set) unless stated otherwise. Version 1.0.0.

## Model details

| | Dropout model | Student-mat model |
|---|---|---|
| Task | Probability that a student drops out of their degree | Probability that a student fails maths (G3 < 10) |
| Algorithm | XGBoost (tuned with Optuna, 30 trials) | L2-regularised logistic regression (C tuned) |
| Selected because | Highest mean CV ROC-AUC of five families (differences not significant under the corrected t-test) | Same rule |
| Inputs | 23 enrolment-time features. **Gender is not an input** | 29 features. **Sex is not an input** |
| Fairness treatment | Reweighing on gender during training | Reweighing on sex during training |
| Calibration | None: cross-validation showed the raw scores were already the best calibrated | Platt scaling (chosen by cross-validated Brier score) |
| Outreach threshold | 0.37 calibrated risk (max out-of-fold F1) | 0.30 |
| Risk bands | low < 0.25 ≤ moderate < 0.50 ≤ high | same |
| Confidence | Split-conformal prediction sets, 90% target coverage | same |
| Artifact | `artifacts/dropout_model.joblib` (SHA-256 in `artifacts/registry.json`) | `artifacts/student_mat_model.joblib` |
| Author | MD SADRUL ANAM, Gannan Normal University. Licence: MIT | |

## Intended use

* **Primary use:** helping a student-success team decide whom to contact first with an *offer of support* (advising, tutoring,
  financial counselling) at the start of a programme or school year.
* **Primary users:** advisors and student-success staff, and researchers studying fairness in early-warning systems.
* **How:** a person who knows the student reviews every flag and decides what, if anything, to do. The score is one input to that
  conversation, alongside the explanation, the confidence level and the fairness context the dashboard shows.

## Out of scope (do not use for)

* admissions, scholarships, financial-aid decisions, discipline, grading or placement;
* any automated action, or any decision that withholds an opportunity from a student;
* showing a student their own risk score, or ranking students publicly;
* populations that differ substantially from Portuguese higher or secondary education, without re-validating on local data.

## Metrics (held-out test set)

| Metric | Dropout (n = 726) | Student-mat (n = 79) |
|---|---|---|
| ROC-AUC [95% CI] | 0.861 [0.833, 0.888] | 0.665 [0.546, 0.780] |
| PR-AUC | 0.828 [0.792, 0.862] | 0.508 [0.380, 0.657] |
| Brier score | 0.142 | 0.205 |
| Calibration error (ECE) | 0.030 | 0.077 |
| Recall at threshold | 0.768 [0.718, 0.817] | 0.577 [0.385, 0.769] |
| Precision at threshold | 0.688 [0.650, 0.728] | 0.429 [0.314, 0.545] |
| Conformal coverage (target 0.90) | 0.912 | 0.924 |
| Share of "uncertain" predictions | 27.5% | 55.7% |

The student-mat model is weak (the AUC interval reaches down to 0.55) and uncertain for more than half of students. It is included as
a research case study of the small-data regime, not as something to deploy.

## Fairness evaluation

Gaps are max minus min across groups, at each model's own threshold, with 95% stratified-bootstrap intervals and permutation p-values.

**Dropout, by gender** (unmitigated aware model vs deployed model):

| Gap | Unmitigated | Deployed | p (deployed) |
|---|---|---|---|
| Demographic parity (flag rate) | 0.389 [0.322, 0.451] | 0.231 [0.163, 0.298] | < 0.001 |
| Equal opportunity (recall) | 0.184 [0.086, 0.274] | 0.036 [0.001, 0.127] | 0.50 |
| False positive rate | 0.328 [0.228, 0.425] | 0.150 [0.056, 0.243] | 0.001 |
| Equalized odds | 0.328 | 0.150 [0.062, 0.243] | 0.006 |

How to read this: the model now catches at-risk women and men at almost the same rate (75% vs 79%). Men are still flagged more often
(59% vs 36%), partly because men in this data really do drop out more often (56% vs 30% in the test set), and partly because more men who
would have graduated are flagged anyway (the remaining false-positive gap). That remaining gap means more unnecessary outreach for male
students. Whether it's acceptable is a policy decision; the dashboard's threshold explorer shows what changing it would cost.

**Other axes and intersections.** Debtor status, age band, gender × debtor and gender × age band are all audited
(`reports/dropout/final_model.json`). Their flag-rate gaps are larger (0.49 for debtor status, 0.56 for age band) and clearly real, but the groups are small, so the
pre-flight check returns *clear gap, coarse measurement*: only changes larger than about 0.08 (single axes) to 0.15 (intersections)
could be confirmed. The deployed model's mitigation targets gender only; these axes are reported and monitored. Note that debt and
tuition status are also among the strongest predictors, so blinding them would cost accuracy (AUC 0.851 → 0.804 without both). Within individual programmes, the enrolment-weighted male-minus-female
flag-rate gap is 0.14 (programme audit, development split).

**Student-mat.** All gaps have intervals spanning most of the plausible range (e.g. equal opportunity 0.42 [0.10, 0.73]); the
pre-flight check returns *insufficient data* for sex, home address and their intersection. For parental education it finds a clear
flag-rate gap (0.36, permutation p < 0.001) but could only confirm changes larger than about 0.16. No fairness claim is made for this model.

## Training and evaluation data

See [DATASHEET.md](DATASHEET.md). An 80/20 split stratified on outcome × primary group; everything that learns (tuning, model selection,
calibration, threshold, conformal quantile) uses only the 80% development split. The test split was evaluated once.

## Explanations

SHAP values (path-dependent TreeSHAP for XGBoost, exact linear SHAP for logistic regression), summed back to the original features.
Counterfactual "paths to lower risk" change only features marked as actionable support in `configs/datasets/*.yaml` and never touch
protected or immutable features. They describe what the model responds to, not the causal effect of an intervention.

## Ethical considerations

* **Harms of errors.** A missed at-risk student loses an offer of help; a wrongly flagged student gets unneeded attention and possibly a
  self-fulfilling label. Both are measured per group above.
* **Proxies.** Removing gender doesn't remove gender information; programme, fees and age carry it. That's why mitigation and monitoring
  are needed on top of blinding.
* **Possible timing leakage.** `debtor` and `tuition_up_to_date` are the strongest predictors and may be recorded after enrolment.
  Without them the out-of-fold AUC falls from 0.851 to 0.804. An institution deploying this should use their enrolment-time values only.
* **Financial levers.** On the dropout data the actionable features are mostly financial, so suggested support is mostly financial. That
  is a property of the data, and it says something about where institutions can help.
* **Contestability.** Students should be told that a score exists, be able to ask for a human-only review, and get its main reasons in
  plain language (the dashboard's top factors are designed for that).
* **Binary gender.** The data records gender as binary; non-binary students are invisible to this audit.

## Caveats and recommendations

Re-validate and re-audit on local data before any use; re-run the pre-flight check for every group you care about; report intervals,
not point estimates; review the false-positive gap with the people affected; and retrain when cohorts change, because fairness measured
on one cohort may not carry over to the next.
