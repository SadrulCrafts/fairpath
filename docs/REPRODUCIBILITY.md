# Reproducibility, and an honest account of the rebuild

## Reproduce every number

```bash
make setup
make data          # downloads both UCI datasets into data/raw (not committed)
make reproduce     # re-runs the original thesis protocol, ~2 minutes
make train         # full upgraded pipeline, ~25 minutes on 2 CPU cores
make report        # regenerates reports/overview.json, docs/figures/*, docs/RESULTS.md
```

Everything is seeded (`seed: 42` in `configs/default.yaml`): CV splits, Optuna's TPE sampler, every model's
`random_state`, bootstrap and permutation draws. Package versions are pinned in `requirements.txt`.
Gradient-boosting libraries can differ in the last decimal across CPU architectures; no conclusion depends on that.

Each report embeds its provenance: data source (`real`/`synthetic`), a SHA-256 fingerprint of the input CSV,
the git commit and a timestamp. Model bundles in `artifacts/` are listed with their SHA-256 in `artifacts/registry.json`.

## What happened to the original code

The code that produced the thesis results was lost. What survived was an earlier snapshot of the project.
Before rebuilding anything I audited that snapshot against the thesis text, and two problems came up:

* the surviving dropout script did **not** apply the cleaning that the thesis describes in section 4.2 (it kept the
  `Enrolled` students as negatives and kept the 12 end-of-semester columns), so its stored results show an AUC of 0.92,
  exactly the leakage the thesis warns about;
* the dashboard's group thresholds (0.14 / 0.10) had been tuned on the test split in a notebook.

The printed thesis numbers do **not** have these problems. When the thesis protocol is re-implemented from the text,
the main tables come back almost exactly (next section). So the results were real; the code that produced them was simply
not the code that survived. FairPath is a clean rebuild from the thesis description, with the protocol re-run as a check.

## Re-running the thesis protocol

`fairpath reproduce-thesis` copies the original setup: success as the positive class, aware models, the hyper-parameters
the original code used, the original preprocessing, 5x2 `RepeatedStratifiedKFold` with seed 42, and the inner 80/20 split
for threshold tuning. The full side-by-side table is generated in [RESULTS.md](RESULTS.md#thesis-reproduction).

| What | Result |
|---|---|
| Table 5.1, predictive performance (32 values) | 31 within 0.005, one within 0.01 (HGB macro-F1 on student-mat, 0.594 vs 0.601) |
| Stratified-dummy noise floor | dropout 0.028 vs 0.029, student-mat 0.087 vs 0.087 |
| Table 6.1, Reweighing rows | dropout within 0.01; student-mat LR exact, tree models within 0.03 |
| Table 6.1, post-processing rows | differ by up to 0.07 |
| Table 6.2, student-mat (SPD 0.084 -> 0.120, p = 0.43) | exact |

The post-processing rows are the expected exception. The thesis computed Table 6.1 with only five folds and a ten-point
threshold grid tuned on validation slices of 60-ish students. Re-running with ten folds moves those numbers by more than their own
fold-to-fold standard deviation. That instability is precisely what the thesis' RQ3 is about, so I read it as support for the
thesis' argument rather than a failure to reproduce it.

## Corrections to the thesis text

These came out of the audit. None of them changes the thesis' conclusions, but anyone citing the thesis should know them.

1. **Hyper-parameters (section 3.3).** The text says logistic regression used `class_weight=None`, random forest had unlimited
   depth and HistGradientBoosting used `learning_rate=0.1, max_iter=100`. The results were actually produced with
   `class_weight="balanced"`, `max_depth=8`, and `learning_rate=0.05, max_depth=6, max_iter=300`. I confirmed this by
   reproducing Table 5.1 with each setting; only the second matches.
2. **Bootstrap CV in H4 (section 6.4.1).** The text gives a point estimate of 0.084 and a bootstrap SD of 0.063, which is a
   CV of 75%, not 337%. The 337% in the stored output is 0.0627 / 0.0186, where 0.0186 is the |SPD| of one 64-student
   validation split, with 500 resamples rather than 1,000. The conclusion (student-mat's estimate is uninformative) still holds.
3. **Table 6.3 mixes two versions of the data.** The dropout column (minimum group 245, |SPD| 0.251, ratio 8.6x, CV 17%) was
   computed on the uncleaned n = 4,424 version; Table 6.2 uses the cleaned data (200, 0.305) and reports a CV of 12.6%.
   FairPath recomputes both checks on the cleaned data (see RESULTS.md).
4. **Standard error behind Check 1.** With n_g = 100 per group the standard error of *one* group's rate is 0.05, but SPD is a
   difference of two rates, so its standard error is about sqrt(2) x 0.05 = 0.071. Section 2.4.2 says n_g of about 200 is needed for
   0.05, which contradicts section 6.4.3. Pre-flight v2 replaces this with an explicit confidence-interval width.
5. **"Classifier-free" pre-flight.** Checks 2 and 3 use the trained model's |SPD|, so only Check 1 can be computed without a model.
6. **Proxy explanation (section 8.2).** The claim that male students are older is not supported by the data (mean age 16.73 for
   female and 16.66 for male students). Male students have slightly more past failures (0.37 vs 0.30). The proxy-mediation
   conclusion stands on the feature ranks and the aware-vs-blind comparison, not on this sentence.
7. **Intersectional analysis (section 6.5).** The surviving code analysed Gender x Scholarship, not Gender x Debtor, and its
   numbers differ from the text. FairPath runs Gender x Debtor (and Gender x Age band) from scratch.
8. **Different fold budgets side by side.** Table 6.1 (5x1 folds) gives student-mat HGB a baseline |SPD| of 0.110, while
   Tables 5.1/6.2 (5x2 folds) give 0.084 for the same cell.
9. **Calibration recommendation.** Section 7.3.1 says HGB on dropout needs no recalibration; section 7.4 recommends it.
10. **Effect size and Platt scaling.** The text defines the rank-biserial r; the surviving code computed r = Z / sqrt(N) from the
    p-value. Platt scaling is described with 5 internal folds; the code used 3.
11. **Permutation importance.** Described as computed on the full dataset for all three models with 10 repeats; the surviving
    code used HGB only, a 79-student test split and 20 repeats (sex ranked 16th rather than 14th).
12. **Artefact claims (appendices A and B).** The 44 unit tests, `src/data.py`, `src/mitigation.py`, `src/baselines.py` and the
    `docs/` folder described there were part of the lost version. FairPath's test suite replaces them.

## Thesis protocol vs. upgraded pipeline

| | Thesis | FairPath |
|---|---|---|
| Positive class | success | at risk (the student an advisor needs to find) |
| Evaluation | 5x2 CV on all data (5x1 for some tables) | 80/20 development/test split; 5x5 CV inside development; one final test evaluation |
| Categorical codes (dropout) | treated as numbers | one-hot encoded, rare levels pooled |
| Models | LR, RF, HGB, fixed hyper-parameters | + LightGBM, XGBoost; all tuned with Optuna |
| Thresholds | 0.5 | rate-matched for comparisons; calibrated, F1-optimal for the deployed model |
| Statistics | Wilcoxon, effect size from p | Wilcoxon + rank-biserial, Holm correction, Nadeau-Bengio corrected t-test, bootstrap CIs, permutation tests |
| Mitigation | Reweighing, group thresholds | + Exponentiated Gradient, ThresholdOptimizer, blind and blind + Reweighing |
| Pre-flight | v1 | v1 and v2 |

Because of the different positive class, test protocol and tuning, upgraded numbers are not supposed to equal the thesis numbers.
Where they tell a different story, RESULTS.md says so.
