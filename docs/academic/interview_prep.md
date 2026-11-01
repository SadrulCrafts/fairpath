# Interview preparation

Read this aloud a few times before an interview. The aim is not to memorise answers but to be able to explain every choice
in your own words, simply, and to say "I don't know, but here is how I'd find out" when that is the truth. Professors
are testing whether you understand your own work and whether you'd be good to work with, not whether you know everything.

All numbers are from `reports/` (real UCI data). If you quote one, know where it came from.

---

## Part 1. Every design decision, in plain words

### The question itself
**What:** "Under what data conditions can fairness mitigation measurably work?" rather than "which method is best?"
**Why:** In education, datasets are often one cohort or one department. If a fairness difference can't even be measured on
your data, comparing methods on it is meaningless. Asking the measurability question first is the contribution.

### Two datasets that differ mainly in size
**What:** UCI dropout (3,630 students after cleaning) and UCI student-mat (395).
**Why:** Same country, similar kind of features, but a 9x difference in size. That makes them a natural pair for asking how
sample size affects what you can measure. Both are public and CC BY 4.0, so anyone can rerun everything.
**Limitation to admit:** two datasets from one country is a small basis for general thresholds. That is research direction 6.

### Removing leakage
**What:** Dropped the 12 "Curricular units" columns (grades and credits at the end of semesters 1 and 2) and removed students
still enrolled at the cut-off.
**Why:** The end-of-semester grades are nearly the outcome. With them, cross-validated AUC is 0.954; without them, 0.845
(`scripts/leakage_check.py`). A model that only works once a student has already failed their semester is useless as an *early*
warning. "Enrolled" students have no known outcome yet, so labelling them as "not dropped out" would be wrong.

### At-risk as the positive class
**What:** The model predicts risk of dropping out / failing, not success.
**Why:** The thesis predicted success. For a support tool the action is "who should an advisor contact", so the errors that
matter are missed at-risk students (false negatives) and unnecessary flags (false positives). Making "at risk" the positive class
makes the true-positive rate mean "share of struggling students we reach", which is much easier to explain.

### A test set used once
**What:** 80/20 split, stratified on outcome × group. All tuning, threshold choice and calibration happen on the 80%.
The 20% is scored once at the end.
**Why:** If you look at the test set while making choices, its numbers stop being an honest estimate. The original dashboard
had tuned its group thresholds on the test split; this fixes that.

### Five model families, tuned the same way
**What:** Logistic regression, random forest, HistGradientBoosting, LightGBM and XGBoost, each tuned with Optuna (30 trials,
5-fold CV), then compared with 5x5 repeated cross-validation.
**Why:** Fairness results can depend on the model. Tuning all of them the same way avoids comparing a tuned XGBoost to an
untuned baseline.

### The corrected t-test
**What:** The Nadeau–Bengio corrected resampled t-test alongside the Wilcoxon test.
**Why:** Cross-validation folds share training data, so they are not independent. Ordinary tests then become far too
confident. With the naive Wilcoxon, XGBoost looks significantly better than logistic regression (Holm p < 0.001); with the
correction, it isn't (p = 0.347). The AUC difference is about 0.005. I chose XGBoost because it was marginally best, not because
it is clearly better, and I say so.

### Rate-matched thresholds for comparisons
**What:** When comparing models or methods, each model's threshold is set so it flags the same share of students.
**Why:** Some tuned models (e.g. random forest) rarely output probabilities above 0.5, so at a 0.5 threshold they'd flag almost no
one and look "fair". Matching the flag rate compares like with like.

### Nine mitigation conditions
**What:** Baseline, blind (gender removed), Reweighing, blind + Reweighing, Exponentiated Gradient (demographic parity and
equalized odds), the thesis' group-threshold search, and Fairlearn's ThresholdOptimizer (two constraints).
**Why:** Covers pre-processing, in-processing and post-processing, which was the thesis' comparison, plus the "just remove the
column" approach that practitioners often use.

### Detecting degenerate solutions
**What:** A method is flagged "degenerate" if it flags more than 95% or fewer than 5% of students. Such methods are excluded
from the Pareto frontier and from recommendations.
**Why:** On student-mat, three methods reached a gap of exactly zero, by flagging everyone or no one. A metric can be gamed
this way, and a naive "lowest gap wins" rule would recommend them.

### The deployed model: blind + Reweighing
**What:** The model the dashboard serves never receives gender; during training, rows are reweighted so group and outcome are
independent.
**Why:** Post-processing methods (group thresholds, ThresholdOptimizer) need the student's gender *at prediction time*, which
means treating students differently by gender at the moment of decision. That raises legal and ethical issues and many
institutions would not accept it. Among methods that don't need gender at prediction, blind + Reweighing had the best
fairness–accuracy balance on dropout. Held-out: equalized-odds gap 0.328 → 0.150, equal-opportunity gap 0.184 → 0.036, AUC
0.864 → 0.861.
**Subtlety:** gender is still used *during training* to compute the weights. Be ready to say this clearly.

### The pre-flight check, v1 and v2
**What v1 was (thesis):** three thresholds: at least 100 students per group, the model's gap at least 3x what a random
predictor produces, and bootstrap coefficient of variation at most 50%.
**Problem I found:** v1 can't distinguish "this dataset is genuinely fair" from "this dataset is too small to tell", because a
small true gap fails the noise-floor rule either way. The coefficient of variation also explodes when the gap is near zero.
**What v2 asks:** (1) Is the estimate precise? 95% CI half-width ≤ 0.05. (2) Is the gap real? permutation test p < 0.05.
(3) Is there room to improve? CI lower bound ≥ 0.05. Verdicts: mitigate, monitor, insufficient data, and a fourth one I added
after the first real run: *clear gap, coarse measurement*.
**Why the fourth verdict:** v2 first called debtor status "insufficient data", even though its flag-rate gap was 0.49 with
permutation p < 0.001. Debtors are a small group, so the precision check failed by 0.006. The gap was never in doubt; only the
resolution for measuring a *change* was. So if the gap is clearly real and its lower bound is larger than the smallest change the data
could confirm, the verdict is "mitigate, but only changes larger than X can be confirmed".
**Result:** Dropout × gender → mitigate. Dropout's smaller axes (debtor, age, intersections) → clear gap, coarse measurement.
Student-mat × sex, address and sex × address → insufficient data; student-mat × parental education → coarse (gap 0.36, only
changes above about 0.16 confirmable).

### Calibration chosen by cross-validation
**What:** The calibration method (none, sigmoid, isotonic) is chosen by cross-validated Brier score.
**Why:** I first used isotonic by default and it made test calibration worse. Choosing by CV picked "none" for dropout (test
ECE 0.030) and sigmoid for student-mat.

### Conformal prediction for confidence
**What:** Split-conformal prediction sets at 90% target coverage. If the set contains both outcomes, the dashboard labels the
prediction "uncertain".
**Why:** A probability alone doesn't tell an advisor when the model is guessing. Conformal sets come with a coverage guarantee
that doesn't depend on the model being correct. Test coverage was 0.912 for a 0.90 target; about 28% of students were labelled
uncertain.

### SHAP aggregated to original features
**What:** SHAP values computed per one-hot column, then summed back to the original feature, using the tree path-dependent
method.
**Why:** An advisor wants "course" not "course_9500". Summing keeps the additivity property (values add up to the prediction),
which I test. I switched from the interventional method because it violated additivity with LightGBM.

### Counterfactuals restricted to actionable features
**What:** The "Path to lower risk" panel searches only over things an institution can support (tuition payment plans, a
scholarship, debt help; for the school data, study time, absences, extra support). Never age, gender, nationality or
parents' education.
**Why:** Suggesting a student "be younger" is useless and offensive. The output is phrased as support the institution can
offer, not as advice to the student.

### Human in the loop and privacy
**What:** A disclaimer on every page, a stateless API that stores nothing, no student ever sees their own score, synthetic
examples only in the demo.
**Why:** The tool decides who to *talk to*, not what happens to anyone. Risk scores can become self-fulfilling if shown to
students or used for admission.

### Rebuilding after the code was lost
**What:** Re-implemented the thesis protocol from the text, reproduced 31 of 32 Table 5.1 values within 0.005, and documented
12 corrections in `docs/REPRODUCIBILITY.md`.
**Why:** I had to confirm my own results were real before building on them. The corrections (e.g. the hyper-parameters
actually used, a misreported bootstrap CV) don't change the conclusions, but I'd rather point them out myself than have a
reviewer find them.

---

## Part 2. Twenty questions you are likely to be asked

### About the research

**1. Summarise your thesis in two minutes.**
> I studied whether fairness mitigation works in student-success prediction, using two public Portuguese datasets that differ
> in size. On the larger one, 3,630 university students, every mitigation method significantly reduced the gap in error rates
> between male and female students. On the smaller one, 395 school students, nothing could be shown to work, and the methods that
> looked perfectly fair were flagging everyone or no one. I broke that failure into four hypotheses (group size, threshold
> resolution, noise floor and estimate variance) and turned them into a pre-flight check practitioners can run before they
> mitigate. After the thesis I rebuilt it as an open-source system with a dashboard, and in doing so I improved the check so it
> can tell "too small to measure" apart from "genuinely fair".

**2. What is your main contribution? Isn't "small data is noisy" obvious?**
> That small data is noisy is obvious. What's not obvious is *how small is too small for a given fairness claim*, and what
> practitioners should do about it. Most fairness papers report improvements without asking whether they could have been
> measured on that data. My contribution is making that question operational: a cheap check with three concrete criteria, and
> evidence that its verdicts line up with when mitigation actually succeeded. I'd be the first to say that two datasets isn't
> enough to validate the thresholds, which is why I'd like to test it more widely.

**3. Why equalized odds rather than demographic parity?**
> In this setting the base rates genuinely differ: in the dropout test set, 56% of male students dropped out vs 30% of female
> students. Demographic parity would force the model to flag both groups at the same rate, which means missing at-risk men or
> over-flagging women. Equalized odds asks for equal error rates given the true outcome: struggling students in both groups
> should be equally likely to be reached. I still report demographic parity, because an institution might care about how
> support is distributed, but I don't optimise for it by default.

**4. The deployed model still has an equalized-odds gap of 0.150. Is that fair?**
> It's fairer, not fair. The equal-opportunity gap, the share of at-risk students reached, is down to 0.036 and its permutation
> p-value is 0.5, so it's not distinguishable from zero. The remaining gap is in false-positive rates: male students who would
> not drop out are flagged more often. Whether that is acceptable depends on what a flag leads to. If it's an advisor's email,
> an extra false positive costs little; if it led to something punitive it would matter a lot. I'd want the institution to make
> that call explicitly, and the dashboard shows the trade-off so they can.

**5. If you remove gender, how can the model still be unfair?**
> Other features carry information about gender: programme of study, fee status, age at enrolment. Removing gender reduced the
> cross-validated gap from 0.298 to 0.173, about 40%, so most of it remained. That's the classic finding that "fairness
> through unawareness" doesn't work. What I'd like to study next is which of those proxy pathways are legitimate, for example
> programme choice, and which reflect a structural problem, like debt.

**6. Why didn't you use post-processing, which gave smaller gaps?**
> Because it needs the student's gender at the moment of prediction, to apply a different threshold. That is explicit
> differential treatment, which many institutions can't legally or ethically adopt. It's still in the study and on the
> fairness page, so people can see what it would achieve, but the recommendation only considers methods that work without
> gender at prediction time.

**7. Tell me about the pre-flight check's limitations.**
> Three main ones, and one I already ran into. When I first ran v2, it called a gap of 0.49 "insufficient data" because the group
> was small, which was obviously wrong, so I added a "clear gap, coarse measurement" verdict. That taught me the check needs testing on
> many more datasets before anyone relies on it. Beyond that: the thresholds (0.05 half-width, 0.05 lower bound) are choices, not laws; they encode what I'd call a
> meaningful gap in this setting. It was derived from two datasets from one country. And "insufficient data" leaves small
> groups unaudited, which is honest but not a solution. A Bayesian hierarchical approach that pools across groups is the
> direction I'd take for that.

**8. What would change your conclusions?**
> If the check's verdicts didn't predict mitigation success on other datasets, the thresholds would be wrong. If a future
> cohort showed the deployed model's gap growing, the fairness result wouldn't be stable. And if someone showed that the debt
> and tuition features are recorded *after* trouble begins, the model would be partly leaking. I tested that last one: dropping
> them lowers AUC from 0.851 to 0.804, so they matter a lot, and I flag their timing as an open question in the model card.

### About methods

**9. Why the corrected t-test? What's wrong with a paired t-test on CV folds?**
> The folds share most of their training data, so their scores are correlated. A standard test treats them as independent and
> underestimates the variance, so it finds "significant" differences too easily. Nadeau and Bengio's correction inflates the
> variance by a factor based on the train/test ratio. In my benchmark it changes the conclusion: the naive test says XGBoost
> beats logistic regression, the corrected one says p = 0.347.

**10. How did you choose the decision threshold?**
> On out-of-fold predictions from the development set, I chose the threshold that maximised F1: 0.37 for dropout. I never
> touched the test set to choose it. In a real deployment I'd prefer the institution to choose it based on advisor capacity,
> for example "we can contact 200 students a term", and the dashboard's threshold sliders support that.

**11. Why do you need calibration if you have AUC?**
> AUC only measures ranking. If the dashboard says "45% risk", that number should mean that roughly 45 of 100 such students drop
> out; otherwise advisors misread it. Calibration also differs by group: in the test set, ECE is 0.060 for female and 0.064 for
> male students, which I report on the fairness page.

**12. How do you know SHAP explanations are right?**
> I know they're *faithful to the model*. The values add up exactly to the model's output, and I test that. Whether they reflect
> *causes* in the real world is a different question and the answer is no; SHAP explains the model, not the student. That's
> why the dashboard calls them "factors behind this estimate" and why counterfactuals are framed as support options, not causal
> claims.

**13. How did you handle intersectional groups?**
> The mitigation study covers gender × debtor status on dropout and sex × address on student-mat; the held-out audit adds
> gender × age band. Every group gets a reliability label (ok, caution, unreliable) based on its size. Some intersections are
> tiny: on student-mat, Fairlearn's ThresholdOptimizer failed on one fold for sex × address because a cell had only one outcome
> class. Failed folds are recorded rather than silently dropped, and paired tests use only valid folds. The pre-flight check
> calls the student-mat intersection insufficient data.

**14. Your student-mat model has an AUC of 0.665. Why include it at all?**
> Because it is what a lot of real educational data looks like: few students, weak signal. Its confidence interval runs from
> 0.546 to 0.780, which is honest about how little we know. The point of including it is to show what the pre-flight check does
> in that situation: it says don't claim mitigation success here.

### About the system and your process

**15. What happened to your original code, and how do I know your results are real?**
> The code that produced the thesis was lost; an older snapshot survived, and when I audited it I found it hadn't applied the
> thesis' own cleaning and had tuned thresholds on the test set. So I re-implemented the protocol from the thesis text and
> reproduced 31 of 32 headline numbers within 0.005. That tells me the printed results were real. I documented 12 corrections to
> the thesis text in the repository. Everything in FairPath now runs from one command and each report records the data hash
> and commit.

**16. What was the hardest bug?**
> A good one: the counterfactual search was returning poor suggestions. It turned out I computed whether any change reached the
> target, but Python's `list.sort` temporarily empties the list while sorting, and my sort key read that list, so the key
> saw it as empty. The ordering was silently wrong. I fixed it and wrote a regression test. Another was isotonic calibration
> making test calibration *worse*, which is why calibration is now chosen by cross-validation.

**17. How would you deploy this in a real university?**
> Carefully, and probably not as-is. I'd first validate on that university's own data and run the pre-flight check there. I'd
> agree with the institution what a flag leads to (an advisor conversation, never anything punitive), who sees scores
> (advisors only, not students or admissions), and how often the model is re-audited. I'd want a student representative
> involved. And I'd monitor fairness per cohort, since a model fair this year might not be next year.

**18. What would you do differently?**
> I'd plan for reproducibility from day one: version the data and code together, so losing a folder wouldn't matter. I'd also
> involve advisors earlier. The dashboard's design choices are reasonable guesses, but I haven't tested them with users.

### About fit

**19. Why do you want to do a Master's/PhD, and why with me?**
> Answer specifically. Name one or two of their papers and connect them to one research direction from
> `research_directions.md`. For example: "Your work on X raised a question I couldn't answer in my thesis: Y. I'd like to
> work on that." Don't flatter; show you've read the work.

**20. Do you have any questions for me?**
> Always have two or three ready. Good ones:
> * "What does a typical week look like for your students: how often do you meet, and how independent are they expected to be?"
> * "Are there datasets or partner institutions your group works with, where something like the pre-flight check could be tested?"
> * "What do your recent graduates go on to do?"
> * "Is there a project you'd like a new student to start with, or do students usually bring their own direction?"

---

## Part 3. Things to avoid

* **Don't oversell.** "Fairer, not fair." "Marginally best, not significantly." Precise modesty reads as competence.
* **Don't hide the lost code.** You found and fixed problems in your own work. That's a strength. Tell it plainly.
* **Don't guess numbers.** "I believe it was around 0.15; it's in the report" is fine. A confident wrong number is not.
* **Don't argue when challenged.** Say "That's a fair point", then think aloud: what evidence would settle it, and what would you
  change.
