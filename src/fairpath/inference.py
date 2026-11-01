"""Serving-side logic: load a versioned model bundle and turn one student into an answer.

A bundle is a single joblib file holding everything needed to reproduce a prediction:
the fitted pipeline, the calibrator, the decision threshold, the conformal quantile, the
input schema and the provenance metadata (data fingerprint, git commit, metrics).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from fairpath.config import load_dataset_spec
from fairpath.data.schema import option_label, validate_frame
from fairpath.evaluation.calibration import ProbabilityCalibrator
from fairpath.explain.conformal import confidence_info
from fairpath.explain.counterfactual import find_counterfactuals
from fairpath.explain.shap_explain import PipelineExplainer

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    dataset: str
    version: str
    model_name: str
    params: dict[str, Any]
    features: list[str]
    pipeline: Pipeline
    calibrator: ProbabilityCalibrator
    threshold: float
    bands: dict[str, float]
    qhat: float
    alpha: float
    schema: dict[str, Any]
    background: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest

    @staticmethod
    def load(path: Path) -> ModelBundle:
        return joblib.load(path)


def risk_band(p: float, bands: dict[str, float]) -> str:
    if p >= bands["high"]:
        return "high"
    if p >= bands["low"]:
        return "moderate"
    return "low"


BAND_TEXT = {
    "low": "Low risk: no extra action suggested beyond normal check-ins.",
    "moderate": "Moderate risk: worth a friendly check-in from an advisor.",
    "high": "High risk: prioritise a conversation and offer concrete support.",
}


class Predictor:
    """Thin, stateless wrapper used by the API. Nothing a user sends is stored."""

    def __init__(self, bundle: ModelBundle):
        self.b = bundle
        self.spec = load_dataset_spec(bundle.dataset)

    @classmethod
    def from_path(cls, path: Path) -> Predictor:
        return cls(ModelBundle.load(path))

    @cached_property
    def explainer(self) -> PipelineExplainer:
        return PipelineExplainer(self.b.pipeline, self.b.background[self.b.features])

    # -- core -------------------------------------------------------------------------
    def prepare(self, records: pd.DataFrame | list[dict] | dict, fill_defaults: bool = True) -> pd.DataFrame:
        if isinstance(records, dict):
            records = [records]
        df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        df = df.reset_index(drop=True)
        return validate_frame(df, self.b.schema, fill_defaults=fill_defaults)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.b.pipeline.predict_proba(X[self.b.features])[:, 1]
        return self.b.calibrator.transform(raw)

    def predict_table(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_risk(X)
        out = pd.DataFrame(
            {
                "risk": np.round(p, 4),
                "band": [risk_band(v, self.b.bands) for v in p],
                "flagged": (p >= self.b.threshold).astype(int),
                "confidence": [confidence_info(v, self.b.qhat, self.b.alpha)["level"] for v in p],
            }
        )
        return out

    def explain_one(self, x: pd.DataFrame, top_k: int = 8) -> dict[str, Any]:
        loc = self.explainer.local(x, top_k=top_k)
        for c in loc["top"] + loc["all"]:
            c["label"] = self.spec.feature(c["feature"]).label
            c["display_value"] = self.display_value(c["feature"], c["value"])
        return loc

    def display_value(self, feature: str, value: Any) -> str:
        f = self.spec.feature(feature)
        if f.type in ("categorical", "binary"):
            v = int(value) if f.type == "binary" and value is not None else value
            return option_label(self.spec, feature, v)
        if value is None:
            return "-"
        return f"{value:g}"

    def counterfactuals(self, x: pd.DataFrame, target: float | None = None) -> dict[str, Any]:
        target = target if target is not None else self.b.threshold
        current = float(self.predict_risk(x)[0])
        if current < target:
            # already below the flag threshold: aim for the next band down instead
            target = self.b.bands["low"] if current >= self.b.bands["low"] else current
        cf = find_counterfactuals(x.iloc[0], self.predict_risk, self.spec, self.b.schema, target)
        for opt in cf["options"]:
            for ch in opt["changes"]:
                ch["from_label"] = self.display_value(ch["feature"], ch["from"])
                ch["to_label"] = self.display_value(ch["feature"], ch["to"])
        return cf

    def fairness_note(self) -> dict[str, Any]:
        return self.b.metadata.get("fairness_note", {})

    def predict_one(self, record: dict, explain: bool = True, counterfactual: bool = True) -> dict[str, Any]:
        x = self.prepare(record)
        p = float(self.predict_risk(x)[0])
        band = risk_band(p, self.b.bands)
        out: dict[str, Any] = {
            "dataset": self.b.dataset,
            "model_version": self.b.version,
            "risk": p,
            "band": band,
            "band_text": BAND_TEXT[band],
            "flagged": bool(p >= self.b.threshold),
            "threshold": self.b.threshold,
            "confidence": confidence_info(p, self.b.qhat, self.b.alpha),
            "fairness_note": self.fairness_note(),
        }
        if explain:
            out["explanation"] = self.explain_one(x)
        if counterfactual:
            out["counterfactuals"] = self.counterfactuals(x)
        return out


def load_registry(artifacts_dir: Path) -> dict[str, Any]:
    p = artifacts_dir / "registry.json"
    return json.loads(p.read_text()) if p.exists() else {"models": {}}
