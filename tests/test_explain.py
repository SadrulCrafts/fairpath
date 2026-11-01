import numpy as np

from fairpath.explain.conformal import coverage, fit_conformal, prediction_set
from fairpath.inference import Predictor


def test_conformal_coverage_on_fresh_data(rng):
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int)
    q = fit_conformal(y[:2000], p[:2000], alpha=0.1)
    cov = coverage(y[2000:], p[2000:], q)
    assert cov["coverage"] >= 0.87
    assert prediction_set(0.99, q) == [1]


def test_shap_values_add_up_to_model_output(bundles, student):
    pred = Predictor(bundles["student_mat"])
    x = student.X.iloc[[3]]
    loc = pred.explainer.local(x)
    raw = pred.b.pipeline.predict_proba(x[pred.b.features])[:, 1][0]
    logit = np.log(raw / (1 - raw))
    assert abs(loc["output_value"] - logit) < 1e-4


def test_counterfactuals_respect_protected_features(bundles, student):
    pred = Predictor(bundles["student_mat"])
    risks = pred.predict_risk(pred.prepare(student.X, fill_defaults=False))
    i = int(np.argmax(risks))
    x = pred.prepare(student.X.iloc[[i]].reset_index(drop=True))
    cf = pred.counterfactuals(x, target=risks[i] - 0.05)
    protected = {f.name for f in pred.spec.features if f.protected}
    actionable = {f.name for f in pred.spec.features if f.is_actionable}
    assert cf["options"], "expected at least one option for the riskiest student"
    for opt in cf["options"]:
        changed = {c["feature"] for c in opt["changes"]}
        assert not (changed & protected)
        assert changed <= actionable
        assert opt["new_risk"] < cf["current_risk"]


def test_predict_one_full_payload(bundles, dropout):
    pred = Predictor(bundles["dropout"])
    out = pred.predict_one(dropout.X.iloc[0].to_dict())
    assert 0 <= out["risk"] <= 1
    assert out["band"] in {"low", "moderate", "high"}
    assert out["confidence"]["level"] in {"confident", "uncertain", "atypical"}
    assert out["explanation"]["top"][0]["label"]
    assert "options" in out["counterfactuals"]


def test_tree_shap_is_additive(bundles, dropout):
    pred = Predictor(bundles["dropout"])  # LightGBM bundle
    x = dropout.X.iloc[[5]]
    loc = pred.explainer.local(x)
    raw = pred.b.pipeline.predict_proba(x[pred.b.features])[:, 1][0]
    assert abs(loc["output_value"] - np.log(raw / (1 - raw))) < 1e-3


def test_counterfactuals_prefer_fewest_changes():
    """Regression test: options must be ordered by number of changes when the target is reachable."""
    from pathlib import Path

    from fairpath.config import PROJECT_ROOT
    from fairpath.inference import Predictor

    path = PROJECT_ROOT / "artifacts" / "dropout_model.joblib"
    if not path.exists():
        import pytest

        pytest.skip("trained bundle not present")
    p = Predictor.from_path(Path(path))
    x = p.prepare({"course": "9500", "tuition_up_to_date": 0, "debtor": 1, "scholarship_holder": 0})
    cf = p.counterfactuals(x)
    sizes = [o["n_changes"] for o in cf["options"]]
    assert sizes == sorted(sizes)
    assert cf["reachable"] and cf["options"][0]["new_risk"] < cf["target_risk"]
