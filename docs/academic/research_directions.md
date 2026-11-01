# Research directions

Each of these grows out of something the thesis or the rebuild found. They are written as starting points for a Master's or PhD
proposal: a question, why it is open, and a first study I could run.

## 1. Fairness under distribution shift between cohorts

**Question.** Does a model that is fair on one cohort stay fair on the next, and can we tell in advance when it won't?
**Why open.** Early-warning models are trained on past cohorts and applied to new ones. Admission policy, programme mix and economic
conditions change (the dropout data includes the 2008–2013 crisis years). Most fairness audits are a single snapshot.
**First study.** Train on early enrolment years, test on later ones, and track how each gap and the pre-flight verdict move. Compare
mitigation methods by how *stable* their fairness is, not just how large the gain is on one split. Extend with importance weighting or
group-DRO and ask whether they buy robustness.

## 2. Causal structure of proxy-mediated bias

**Question.** Through which pathways does gender information reach the prediction, and which of those pathways are legitimate?
**Why open.** The thesis shows that blinding leaves most of the gap. A gap carried by programme choice might be acceptable to an
institution; one carried by debt status might point to a structural problem the institution should address instead.
**First study.** Elicit a causal graph with domain experts, estimate path-specific effects (Nabi & Shpitser; Chiappa), and compare
path-specific counterfactual fairness with the group metrics used here.

## 3. Bayesian fairness estimation for small groups

**Question.** Can a hierarchical Bayesian model give useful fairness estimates where bootstrap intervals are uninformative?
**Why open.** The pre-flight check says "insufficient data" for sex, address and their intersection in the school dataset. That is honest, but
leaves the people in those groups unaudited.
**First study.** Beta-binomial (or hierarchical logistic) models of per-group error rates, partially pooled across intersectional cells,
with posterior probabilities such as P(gap > 0.05). Optimise thresholds against that posterior instead of point estimates, and test
calibration of the posteriors in simulation.

## 4. Student-facing and advisor-facing explanations

**Question.** How do advisors and students interpret risk scores, SHAP explanations and counterfactual "support" suggestions, and do
these explanations change decisions in the intended direction?
**Why open.** The dashboard makes design choices (framing as support, never showing a student their own score, highlighting
uncertainty) that are plausible but untested.
**First study.** A within-subjects study with advisors comparing score-only, score + explanation and score + explanation + uncertainty,
measuring decision quality, over-reliance and perceived fairness. Collaborate with a learning-analytics group with access to advisors.

## 5. Privacy-preserving and multi-institution prediction

**Question.** Can institutions that each hold too little data to audit fairness pool their evidence without pooling student records?
**Why open.** Small institutions fail the pre-flight check on their own. Federated learning and differentially private statistics could
let them audit together.
**First study.** Simulate several institutions by splitting the dropout data by programme, compare federated training plus a secure
aggregate fairness audit with local-only models, and quantify how much privacy budget the audit itself consumes.

## 6. A multi-country test of the pre-flight check

**Question.** Do the pre-flight thresholds generalise beyond Portugal?
**Why open.** The check was derived from two datasets from one country.
**First study.** Apply it to public education datasets from other countries (e.g. Open University Learning Analytics, US and Canadian
open data where available) and check whether its verdicts predict mitigation success out of sample.
