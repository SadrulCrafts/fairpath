"""SHAP explanations aggregated back to the original (human-readable) features.

The model sees one-hot columns ("course=9500", "course=9119", ...). Nobody wants to read
that, so SHAP values of every column derived from one input are summed. Because SHAP
values are additive, this is exact rather than an approximation.

Values are in the model's raw output space (log-odds for boosting and logistic models).
The dashboard shows them as "pushes risk up / down" with relative bar lengths.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from fairpath.features import output_to_input_map

log = logging.getLogger(__name__)


class PipelineExplainer:
    def __init__(
        self, pipeline: Pipeline, background: pd.DataFrame, max_background: int = 200, seed: int = 42
    ):
        self.pipeline = pipeline
        self.pre = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.features = list(background.columns)
        self.col_map = np.asarray(output_to_input_map(self.pre))
        bg = background.sample(min(max_background, len(background)), random_state=seed)
        bg_t = self.pre.transform(bg)
        if bg_t.shape[1] != len(self.col_map):
            raise RuntimeError("feature map does not match transformed width")
        kind = type(self.model).__name__
        if hasattr(self.model, "coef_"):
            self.explainer = shap.LinearExplainer(self.model, bg_t)
            self.kind = "linear"
        else:
            try:
                self.explainer = shap.TreeExplainer(
                    self.model, data=bg_t, model_output="raw", feature_perturbation="interventional"
                )
                self.kind = "tree"
            except Exception:  # pragma: no cover - fallback for unsupported models
                log.warning("TreeExplainer does not support %s; using the permutation explainer", kind)
                f = lambda z: self.model.predict_proba(z)[:, 1]
                self.explainer = shap.Explainer(f, bg_t)
                self.kind = "permutation"
        ev = self.explainer.expected_value
        ev = np.asarray(ev).ravel()
        self.expected_value = float(ev[-1])

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X[self.features])
        if self.kind == "permutation":
            vals = self.explainer(Xt).values
        else:
            vals = self.explainer.shap_values(Xt)
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, cols, classes) -> positive class
            vals = vals[..., -1]
        elif isinstance(vals, list):  # pragma: no cover
            vals = vals[-1]
        return vals

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values per original feature, one row per student."""
        raw = self._raw(X)
        out = pd.DataFrame(0.0, index=X.index, columns=self.features)
        for feat in self.features:
            mask = self.col_map == feat
            if mask.any():
                out[feat] = raw[:, mask].sum(axis=1)
        return out

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self.explain(X)
        imp = sv.abs().mean().sort_values(ascending=False)
        signed = sv.mean()
        return pd.DataFrame(
            {"feature": imp.index, "mean_abs_shap": imp.values, "mean_shap": signed[imp.index].values}
        )

    def local(self, x: pd.DataFrame, top_k: int = 8) -> dict[str, Any]:
        sv = self.explain(x).iloc[0]
        order = sv.abs().sort_values(ascending=False).index
        contributions = [
            {"feature": f, "shap": float(sv[f]), "value": _jsonable(x.iloc[0][f])} for f in order
        ]
        return {
            "base_value": self.expected_value,
            "output_value": float(self.expected_value + sv.sum()),
            "top": contributions[:top_k],
            "rest_shap": float(sum(c["shap"] for c in contributions[top_k:])),
            "all": contributions,
            "space": "log-odds",
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v
