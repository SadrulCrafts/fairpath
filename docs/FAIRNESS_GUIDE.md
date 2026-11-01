# A plain-language guide to the fairness numbers

This is written for advisors, administrators and anyone reviewing the tool, not only for ML people.

## The setting

The model gives each student a risk score. Students above a threshold are **flagged** for an offer of support. Some of them really
were going to struggle (a correct flag); some would have been fine (a false alarm). Some students who were going to struggle are not
flagged (a miss). Fairness questions are about how these outcomes are distributed across groups of students.

## The metrics

| Metric | Question it answers | Zero means |
|---|---|---|
| **Demographic parity gap** (SPD) | Are groups flagged at different rates? | every group is flagged equally often |
| **Disparate impact ratio** | Lowest flag rate ÷ highest | 1 = parity; < 0.8 fails the "four-fifths" rule of thumb |
| **Equal opportunity gap** (TPR gap) | Of the students who really struggled, does the model catch some groups less often? | struggling students are caught at the same rate in every group |
| **False positive gap** | Of the students who would have been fine, are some groups flagged more often anyway? | false alarms fall equally on every group |
| **Equalized odds gap** | The larger of the two gaps above | the model makes the same kinds of mistakes for everyone |
| **Predictive parity gap** | Does a flag mean the same thing for every group (precision)? | a flag is equally reliable for everyone |
| **Calibration gap** | Does "40% risk" mean 40% for every group? | scores mean the same thing for everyone |

All gaps are *largest minus smallest* across groups, so they show the worst case.

## Which one matters most here?

For an early-warning system that offers help, the **equal opportunity gap** comes first: it measures whether some students who need help
are systematically missed. The **false positive gap** comes second: it measures who carries the cost of unnecessary attention.

A **demographic parity gap is not automatically unfair**. In the dropout data, 56% of men and 30% of women in the test set went on to drop
out. A model that flagged both groups equally often would either miss many struggling men or flag many women who were fine. Mathematically,
when base rates differ you cannot equalise flag rates, error rates and calibration all at once (Kleinberg et al., 2017; Chouldechova, 2017),
so a choice has to be made, and it should be made openly.

## Reading the uncertainty

* **95% interval.** If you could re-draw the test students many times, the gap would land in this range about 95% of the time. A wide
  interval (as on the 395-student dataset) means the point estimate tells you little.
* **Permutation p-value.** How often a gap this large would appear if group membership had nothing to do with the predictions. Below 0.05:
  unlikely to be noise.
* **Small-group flags.** Groups under 100 students are marked *caution*, under 30 *unreliable*.

## The mitigation methods

| Stage | Method | What it does | Score depends on the student's group? |
|---|---|---|---|
| none | Baseline (aware) | gender is an ordinary input | yes (as an input) |
| pre | Fairness through unawareness | drop the gender column | no |
| pre | Reweighing | weight training rows so group and outcome look independent | only if gender is also an input |
| pre | Blind + Reweighing (**deployed**) | both of the above | **no** |
| in | Exponentiated Gradient | constrained training for demographic parity or equalized odds; predictions are randomised | yes |
| post | Group thresholds (thesis) / ThresholdOptimizer | a different cut-off per group | yes |

Methods that use the group at prediction time can close gaps further, but two students with identical records can then be treated
differently because of their group. Many institutions, and some legal frameworks, won't accept that, which is why the deployed model doesn't.

**Degenerate solutions.** A method that flags everyone (or no one) has identical error rates for all groups, so every gap is zero. That
isn't a fair model, it's no model at all. FairPath marks these and leaves them off the trade-off chart.

## The pre-flight check

Before trying to fix a gap, ask whether you could even measure the fix.

* **v1 (thesis):** smallest group in a validation split ≥ 100; model gap ≥ 3x the gap of a random predictor; bootstrap coefficient of
  variation ≤ 50%.
* **v2 (revised):** (1) *precision* — is the 95% interval of a gap narrower than ±0.05 with these group sizes? (2) *signal* — is the
  observed gap distinguishable from chance (permutation test)? (3) *room* — is the gap's lower confidence bound at least 0.05?
  Verdicts: **mitigate and measure**; **clear gap, coarse measurement** (signal and room pass, and the gap's lower bound is larger
  than the smallest change the data could confirm, but the interval is wider than ±0.05: worth mitigating, but only large changes can
  be shown); **monitor** (enough data, nothing worth fixing); **insufficient data** (report intervals; claim nothing).

v2 exists because v1's third check divides by the gap itself. A genuinely fair dataset has a gap near zero, so v1 calls it "do not
mitigate" for the same reason it rejects a tiny dataset. v2 tells the two apart.

The "coarse" verdict came from running v2 on the real data. Debtor status on the dropout data has a flag-rate gap of 0.49
(permutation p < 0.001), but debtors are only about 11% of students, so the precision check failed by 0.006. The first version of
v2 called that "insufficient data", which is plainly wrong: the gap isn't in doubt, only how finely a *change* in it can be measured.
