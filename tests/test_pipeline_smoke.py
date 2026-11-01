"""End-to-end smoke test of every pipeline stage on synthetic data with a tiny budget."""

import pytest

from fairpath.config import load_settings
from fairpath.data import load_dataset
from fairpath.pipeline import benchmark, fairness_study, reporting, train


@pytest.mark.slow
def test_full_pipeline_on_synthetic(tmp_path):
    s = load_settings().model_copy(deep=True)
    s.paths = {**s.paths, "reports": str(tmp_path / "reports"), "artifacts": str(tmp_path / "artifacts")}
    s.datasets = ["student_mat"]
    s.models = ["logistic_regression", "lightgbm"]
    s.tuning = {**s.tuning, "n_trials": 2}
    s.cv = {**s.cv, "benchmark": {"n_splits": 3, "n_repeats": 1}, "fairness": {"n_splits": 3, "n_repeats": 1}}
    s.bootstrap = {**s.bootstrap, "n_resamples": 50}
    s.permutation_test = {"n_permutations": 50}
    ds = load_dataset("student_mat", s, source="synthetic")
    b = benchmark.run(ds, s)
    assert b["best_model"] in s.models
    f = fairness_study.run(ds, s)
    assert "sex" in f["summary"] and f["preflight"]["axes"]["sex"]["v2"]["verdict_code"]
    t = train.run(ds, s)
    assert 0.5 < t["test_metrics"]["deployed"]["roc_auc"]["value"] <= 1
    assert (tmp_path / "artifacts" / "synthetic" / "student_mat_model.joblib").exists()
    ov = reporting.run(s, synthetic=True)
    assert "student_mat" in ov["datasets"]
