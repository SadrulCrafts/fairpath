# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-01

First public release. Full notes: [docs/RELEASE_NOTES_v1.0.0.md](docs/RELEASE_NOTES_v1.0.0.md).

### Added
- Config-driven data layer with the thesis' cleaning rules, schema validation and Gaussian-copula synthetic data.
- Benchmark of five model families (LR, RF, HistGradientBoosting, LightGBM, XGBoost) with Optuna tuning and 5x5 repeated CV.
- Statistics: rank-biserial effect size, Holm correction, Nadeau-Bengio corrected t-test, stratified bootstrap CIs, permutation tests.
- Mitigation study: blind, Reweighing, blind + Reweighing, Exponentiated Gradient (DP/EO), thesis group thresholds, ThresholdOptimizer (DP/EO).
- Pre-flight check v1 and v2, plus a `fairpath preflight` CLI. v2 has four verdicts: mitigate, clear gap with coarse measurement,
  monitor, insufficient data.
- `scripts/leakage_check.py`: measures how much the end-of-semester columns inflate AUC (0.845 → 0.954).
- Deployed models: blind + Reweighing, CV-selected calibration, F1-optimal threshold, split-conformal confidence.
- SHAP explanations aggregated to original features; actionable counterfactual search.
- FastAPI service (prediction, what-if, batch, PDF, reports) and Next.js dashboard (7 pages, light/dark, responsive).
- Thesis reproduction harness and documented thesis errata.
- 48 tests, CI on GitHub Actions, Docker image, Render blueprint, model card, datasheet, notebooks.

### Changed (relative to the thesis code)
- Dropout data: `Enrolled` rows and curricular-unit columns are now actually removed (the surviving thesis script kept them).
- Positive class is "at risk" instead of "success".
- Categorical codes are one-hot encoded instead of treated as numbers.
- Post-processing thresholds are tuned only on validation splits (the old dashboard used test-tuned thresholds).

### Fixed
- Crash in group metrics when a group contains a single class.
- Threshold search silently skipping anything other than exactly two groups named F/M.
- Pre-flight v2 calling a large, clearly significant gap "insufficient data" when one group was small.
- Counterfactual ranking: `list.sort()` empties the list while sorting, so a sort key that read it saw an empty list.
- Effect size computed from the p-value instead of the rank-biserial definition.
