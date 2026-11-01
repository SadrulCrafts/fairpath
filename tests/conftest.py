"""Shared fixtures. Everything runs on the committed SYNTHETIC data, so CI needs no downloads."""

from __future__ import annotations

import numpy as np
import pytest

from fairpath.config import load_settings
from fairpath.data import load_dataset
from fairpath.data.schema import build_schema
from fairpath.evaluation.calibration import ProbabilityCalibrator
from fairpath.explain.conformal import fit_conformal
from fairpath.inference import ModelBundle
from fairpath.mitigation import reweighing_weights
from fairpath.models import fit_pipeline, make_pipeline


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def student(settings):
    return load_dataset("student_mat", settings, source="synthetic")


@pytest.fixture(scope="session")
def dropout(settings):
    return load_dataset("dropout", settings, source="synthetic")


def _bundle(ds, model="logistic_regression"):
    feats = ds.spec.model_features(blind=True)
    w = reweighing_weights(ds.y, ds.sensitive[ds.spec.primary_sensitive])
    pipe = fit_pipeline(make_pipeline(model, ds.spec, feats, {}), ds.X[feats], ds.y, w)
    raw = pipe.predict_proba(ds.X[feats])[:, 1]
    cal = ProbabilityCalibrator("sigmoid").fit(raw, ds.y)
    p = cal.transform(raw)
    return ModelBundle(
        dataset=ds.name,
        version="test",
        model_name=model,
        params={},
        features=feats,
        pipeline=pipe,
        calibrator=cal,
        threshold=0.4,
        bands={"low": 0.25, "high": 0.5},
        qhat=fit_conformal(ds.y, p, 0.1),
        alpha=0.1,
        schema=build_schema(ds.spec, ds.X, blind=True),
        background=ds.X.sample(80, random_state=0),
        metadata={"model_label": "Test model", "fairness_note": {"text": "test note", "groups": []}},
    )


@pytest.fixture(scope="session")
def bundles(student, dropout):
    return {"student_mat": _bundle(student, "logistic_regression"), "dropout": _bundle(dropout, "lightgbm")}


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory, bundles):
    d = tmp_path_factory.mktemp("artifacts")
    for name, b in bundles.items():
        b.save(d / f"{name}_model.joblib")
    return d


@pytest.fixture
def rng():
    return np.random.default_rng(0)
