# FairPath 1.0.0

The first public release: my undergraduate thesis on algorithmic bias in student-success prediction, rebuilt as a reproducible
pipeline and a working decision-support dashboard.

## Highlights

* **Leak-free pipeline** on two UCI datasets (n = 3,630 and n = 395): five tuned model families, 5x5 cross-validation, a held-out test
  set used once, corrected significance tests.
* **Fairness audit with uncertainty**: per-group rates, six gap metrics, bootstrap intervals, permutation tests, small-group warnings,
  intersectional axes.
* **Nine mitigation conditions** across pre-, in- and post-processing, compared inside one CV loop, with a Pareto frontier and detection of
  degenerate "fair by flagging everyone" solutions.
* **Pre-flight check** v1 (from the thesis) and v2 (revised), also usable on any CSV: `fairpath preflight data.csv --label y --group g --score p`.
* **Deployed models** that never take gender/sex as input, trained with Reweighing, calibrated, with conformal confidence, SHAP
  explanations and actionable counterfactual support options.
* **Dashboard** (Next.js) with seven pages, and a stateless **FastAPI** service with batch scoring and PDF reports.
* **Thesis reproduction**: 31 of 32 headline performance numbers reproduced within 0.005; 12 corrections to the thesis text documented.

## Key results (held-out test set)

* Dropout: ROC-AUC 0.861 [0.833, 0.888]; equalized-odds gap by gender 0.150 (unmitigated: 0.328); equal-opportunity gap 0.036 (0.184).
* Student-mat: ROC-AUC 0.665 [0.546, 0.780]; every pre-flight check says the data is too small for fairness conclusions.

## Known limitations

Two Portuguese datasets, binary gender, offline evaluation only. See MODEL_CARD.md.
