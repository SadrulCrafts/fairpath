# Datasheet

Following the structure of *Datasheets for Datasets* (Gebru et al., 2021), for the two public datasets FairPath uses.
FairPath does not collect any data of its own.

## 1. Predict Students' Dropout and Academic Success

| | |
|---|---|
| **Source** | UCI Machine Learning Repository, dataset 697 — <https://archive.ics.uci.edu/dataset/697> |
| **Creators** | V. Realinho, J. Machado, L. Baptista, M. V. Martins (Polytechnic Institute of Portalegre) |
| **Citation** | Realinho et al., "Predicting Student Dropout and Academic Success", *Data* 7(11):146, 2022. doi:10.3390/data7110146 |
| **Licence** | CC BY 4.0 (use, share and adapt with attribution) |
| **Raw size** | 4,424 students, 36 features + outcome, one Portuguese higher-education institution, enrolments up to 2018/19 |
| **Identifiers** | none; anonymised at source |

**Motivation.** Created to support early identification of students at risk of not completing their degree.

**Composition.** Demographics (gender, age, marital status, nationality), application route and programme, prior qualifications and
admission grade, parents' qualifications and occupations, financial status (debtor, tuition up to date, scholarship), national
macro-economic indicators at enrolment, and per-semester academic results.

**Cleaning applied by FairPath** (identical to the thesis, defined in `configs/datasets/dropout.yaml`):

1. *794 rows with outcome `Enrolled` removed.* These students had not finished or left at the cut-off. Labelling them as
   "not graduated" would mix an unknown outcome into the negative class, and if enrolment duration differs by gender that
   mislabelling would itself create a gender signal.
2. *12 `Curricular units` columns removed.* They record end-of-semester results for the period being predicted. Keeping them
   gives an AUC around 0.92 that no enrolment-time early-warning system could achieve.

Result: **3,630 students, 24 features, 1,421 dropouts (39.1%)**. Gender: 2,381 female, 1,249 male (coded 0/1 in the source).

**Known issues and caveats.**

* Gender is recorded as binary. Non-binary students are invisible in this data and therefore in any audit built on it.
* `Debtor` and `Tuition fees up to date` may be recorded after enrolment. FairPath reports how much the model relies on them
  (`leakage_check` in `reports/dropout/final_model.json`).
* About 30% of the raw female/male gap in graduation rates is explained by which programmes students choose; the remaining 70% sits
  within programmes (thesis section 4.3.4, reproduced by `programme_audit`).
* Categorical fields are integer codes. Labels in the dashboard come from the UCI documentation; detailed occupation codes are
  shown with their major group.

## 2. Student Performance (Mathematics)

| | |
|---|---|
| **Source** | UCI Machine Learning Repository, dataset 320 — <https://archive.ics.uci.edu/dataset/320> |
| **Creators** | P. Cortez and A. M. G. Silva (University of Minho) |
| **Citation** | Cortez & Silva, "Using data mining to predict secondary school student performance", Proc. FUBUTEC 2008, pp. 5–12 |
| **Licence** | CC BY 4.0 |
| **Raw size** | 395 students, 30 features + 3 grades, two Portuguese secondary schools, 2005/06 |
| **Identifiers** | none; collected through school reports and questionnaires |

**Composition.** School, demographics, family background, study habits, support, social life and self-reported alcohol use and
health; grades G1, G2 (in-year) and G3 (final, 0–20).

**Cleaning applied by FairPath.** Target is fail = G3 < 10 (130 students, 32.9%). G1 and G2 are dropped because they are not
known at the start of the year. Sex: 208 female, 187 male; address: 88 rural, 307 urban.

**Known issues and caveats.**

* With 395 students, any subgroup estimate is noisy; intersectional cells (sex x address) have as few as 44 students. FairPath's
  pre-flight check flags this explicitly.
* Several features are sensitive personal information (alcohol use, relationships, family status). FairPath never suggests
  changing personal circumstances; only school-side support appears as an "actionable" lever.
* Collected in 2005/06; school practice and student life have changed since.

## Data handling in this repository

* **Raw data is never committed.** `data/raw/` is git-ignored and a pre-commit hook blocks it. `make data` downloads both files from
  UCI on demand.
* **Synthetic copies** with the same columns are committed in `data/synthetic/`, generated with a per-class Gaussian copula
  (`src/fairpath/data/synthetic.py`). No synthetic row equals a real row (checked when generated). They exist so that anyone can
  clone and run the code, and so CI never needs the real data. They are not a formal privacy mechanism, and **no result reported
  anywhere in the project comes from them** unless it is marked SYNTHETIC.
* Model artifacts contain no training rows. The SHAP background sample stored in each bundle and the example students shown in the
  dashboard are drawn from the synthetic data.
* The API is stateless: inputs are validated, scored in memory and discarded.
