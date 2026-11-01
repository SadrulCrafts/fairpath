# Architecture

## Design principles

1. **One source of truth per decision.** Cleaning rules, sensitive attributes, feature labels and actionable levers live in
   `configs/datasets/*.yaml`; experiment settings in `configs/default.yaml`. Code reads them, never re-states them.
2. **Nothing learns from the test set.** An 80/20 split is made once (`pipeline/common.py::dev_test_split`). Tuning, model selection,
   calibration, threshold choice and the conformal quantile all use the development split. The test split is scored once in `train.py`.
3. **No hand-typed numbers.** Every figure in the README, docs and dashboard is read from `reports/*.json`, which the pipeline writes.
4. **Stateless serving.** The API keeps nothing a user sends.

## Pipeline stages

```
fairpath reproduce-thesis   -> reports/thesis_reproduction.json
fairpath benchmark          -> reports/<ds>/tuned_params.json, benchmark.json
fairpath fairness           -> reports/<ds>/fairness_study.json, preflight.json
fairpath train              -> artifacts/<ds>_model.joblib, artifacts/registry.json,
                               reports/<ds>/final_model.json, schema.json, examples.json
fairpath report             -> reports/overview.json, docs/figures/*.png, docs/RESULTS.md
scripts/export_static_data.py -> web/public/data/*.json  (read-only data bundled with the dashboard)
```

| Stage | Data | Design |
|---|---|---|
| thesis_repro | all rows | original protocol: success = positive, aware, original hyper-parameters, 5x2 CV |
| benchmark | development | Optuna (TPE, median pruning, 30 trials, 5-fold) per family, then 5x5 repeated stratified CV; corrected t-test + Wilcoxon, Holm |
| fairness_study | development | 5x2 CV; inner 80/20 split for post-processing; 9 methods x 2 models x 3 axes; Wilcoxon vs baseline (Holm); Pareto frontier; degenerate-solution detection |
| preflight | development | v1 and v2 on out-of-fold predictions of the aware baseline, rate-matched threshold |
| train | development, then test once | deployed (blind + Reweighing) and baseline (aware) configs; OOF predictions -> calibrator choice (CV Brier), F1-optimal threshold, conformal quantile; test metrics with bootstrap CIs; group audit with bootstrap + permutation; SHAP; permutation importance; leakage and programme checks |

## Package map

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic models for the YAML configs |
| `data/loaders.py` | read, clean, derive audit axes (age bands, parental education) |
| `data/schema.py` | JSON schema of inputs (built from spec + data), validation for API and batch uploads |
| `data/synthetic.py` | per-class Gaussian copula look-alike data |
| `features.py` | ColumnTransformer (impute, scale, one-hot with rare-level pooling) and one-hot -> feature map |
| `models.py` | estimator registry, default and thesis hyper-parameters, Optuna search spaces |
| `evaluation/` | metrics + bootstrap CIs, calibration (Platt / isotonic / none), statistical tests |
| `fairness/metrics.py` | vectorised group metrics, gap metrics, bootstrap CIs, permutation tests, threshold sweeps |
| `fairness/preflight.py` | pre-flight v1 (thesis) and v2 |
| `mitigation.py` | Reweighing, Exponentiated Gradient, thesis threshold search, ThresholdOptimizer wrappers |
| `explain/` | SHAP aggregation, actionable counterfactual search, split-conformal sets |
| `inference.py` | `ModelBundle` (everything needed to reproduce a prediction) and `Predictor` |
| `api/` | FastAPI app, Pydantic schemas, PDF builder |
| `pipeline/` | the stages above, plus figures and reporting |

## Model bundle

A bundle is one joblib file: fitted pipeline, calibrator, threshold, risk bands, conformal quantile and alpha, input schema, a SHAP
background sample (drawn from the **synthetic** data, so no real rows ship with the model) and metadata (data fingerprint, git commit,
timestamp, test AUC, fairness note). `artifacts/registry.json` records each bundle's SHA-256.

## Serving

```
browser ──> Next.js static export (web/out)
   │           ├── read-only data: /data/*.json (bundled), falls back to /api/reports/*
   │           └── live calls:     POST /api/datasets/{ds}/predict | whatif | batch | batch/report.pdf
   └──> FastAPI (uvicorn) ── Predictor(bundle) ── SHAP / counterfactuals / conformal
```

In the Docker image FastAPI serves both the API and the static dashboard from one origin. The dashboard can also be hosted separately
(set `NEXT_PUBLIC_API_BASE` at build time and `FAIRPATH_CORS_ORIGINS` on the API).

## Determinism

Seed 42 everywhere (splits, Optuna sampler, estimators, bootstrap, permutations); pinned dependencies; provenance stamped into every report.
