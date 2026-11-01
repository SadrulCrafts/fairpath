"""Bias mitigation at all three stages of the pipeline.

* **Pre-processing** -- Reweighing (Kamiran & Calders, 2012): give each (group, label)
  cell a weight so that group and outcome are statistically independent in the weighted
  training data. Only needs the sensitive attribute at training time.
* **In-processing** -- Exponentiated Gradient reduction (Agarwal et al., 2018, via
  Fairlearn): trains a randomised ensemble that optimises accuracy subject to a fairness
  constraint (demographic parity or equalized odds).
* **Post-processing** -- (a) the thesis' per-group threshold grid search with the
  lexicographic objective (FPR gap, then TPR gap, then F1); (b) Fairlearn's
  ThresholdOptimizer (Hardt et al., 2016), which picks (possibly randomised) group
  thresholds on a validation split. Both need the sensitive attribute at prediction time.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def reweighing_weights(y, s) -> np.ndarray:
    """w(s, y) = P(S=s) P(Y=y) / P(S=s, Y=y), rescaled to mean 1."""
    y = np.asarray(y)
    s = pd.Series(np.asarray(s)).astype(str).to_numpy()
    n = len(y)
    w = np.ones(n, dtype=float)
    for sv in np.unique(s):
        ps = np.mean(s == sv)
        for yv in np.unique(y):
            cell = (s == sv) & (y == yv)
            pjoint = cell.mean()
            if pjoint > 0:
                w[cell] = ps * np.mean(y == yv) / pjoint
    return w * (n / w.sum())


def apply_group_thresholds(scores, groups, thr_map: dict[str, float], default: float = 0.5) -> np.ndarray:
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    t = np.array([thr_map.get(v, default) for v in g])
    return (np.asarray(scores) >= t).astype(int)


def thesis_threshold_search(
    y_val, scores_val, groups_val, grid: np.ndarray, max_exhaustive: int = 20_000
) -> dict[str, float]:
    """Per-group thresholds minimising (FPR gap, TPR gap, -F1) lexicographically.

    Works for any number of groups: exhaustive search when the grid is small enough,
    otherwise coordinate descent from the best common threshold.
    (The thesis code only handled exactly {F, M} and silently skipped everything else.)
    """
    y_val = np.asarray(y_val, int)
    scores_val = np.asarray(scores_val, float)
    g = pd.Series(np.asarray(groups_val)).astype(str).to_numpy()
    levels = sorted(set(g))

    codes = np.searchsorted(levels, g)
    pos = np.bincount(codes, weights=y_val, minlength=len(levels))
    neg = np.bincount(codes, minlength=len(levels)) - pos

    def score(tmap):
        # vectorised: this runs up to 10^4 times per fold for intersectional groups
        t = np.array([tmap[lv] for lv in levels])[codes]
        pred = (scores_val >= t).astype(float)
        tp = np.bincount(codes, weights=y_val * pred, minlength=len(levels))
        fp = np.bincount(codes, weights=(1 - y_val) * pred, minlength=len(levels))
        tpr = np.divide(tp, pos, out=np.full(len(levels), np.nan), where=pos > 0)
        fpr = np.divide(fp, neg, out=np.full(len(levels), np.nan), where=neg > 0)
        rng = lambda v: float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).sum() >= 2 else 0.0
        TP, FP = tp.sum(), fp.sum()
        FN = pos.sum() - TP
        f1 = 2 * TP / (2 * TP + FP + FN) if TP > 0 else 0.0
        return (round(rng(fpr), 10), round(rng(tpr), 10), -f1)

    if len(grid) ** len(levels) <= max_exhaustive:
        best, best_s = None, None
        for combo in itertools.product(grid, repeat=len(levels)):
            tmap = {lv: float(t) for lv, t in zip(levels, combo, strict=True)}
            s = score(tmap)
            if best_s is None or s < best_s:
                best, best_s = tmap, s
        return best  # type: ignore[return-value]

    common = min(grid, key=lambda t: score({lv: float(t) for lv in levels}))
    tmap = {lv: float(common) for lv in levels}
    for _ in range(3):
        for lv in levels:
            tmap[lv] = float(min(grid, key=lambda t, lv=lv: score({**tmap, lv: float(t)})))
    return tmap


class FittedProba(BaseEstimator, ClassifierMixin):
    """Expose a fitted pipeline to Fairlearn without it trying to refit or re-validate it."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, X, y, **kw):  # pragma: no cover - prefit only
        return self

    def predict_proba(self, X):
        # float64 on purpose: XGBoost returns float32, which Fairlearn's pandas code rejects
        return np.asarray(self.pipeline.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return self.pipeline.predict(X)


def fit_threshold_optimizer(
    fitted_pipeline, X_val: pd.DataFrame, y_val, s_val, constraint: str, seed: int = 42
) -> ThresholdOptimizer:
    to = ThresholdOptimizer(
        estimator=FittedProba(fitted_pipeline),
        constraints=constraint,  # "demographic_parity" | "equalized_odds"
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    to.fit(X_val, np.asarray(y_val), sensitive_features=pd.Series(np.asarray(s_val)).astype(str).to_numpy())
    return to


def threshold_optimizer_predict(
    to: ThresholdOptimizer, X, s, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (hard predictions, probability of a positive prediction)."""
    s = pd.Series(np.asarray(s)).astype(str).to_numpy()
    pmf = to._pmf_predict(X, sensitive_features=s)[:, 1]
    pred = to.predict(X, sensitive_features=s, random_state=seed)
    return np.asarray(pred, int), np.asarray(pmf, float)


def fit_exponentiated_gradient(
    estimator, Xt: np.ndarray, y, s, constraint: str, eps: float = 0.02, max_iter: int = 50
) -> ExponentiatedGradient:
    """In-processing on an already-transformed numeric matrix ``Xt``.

    The preprocessor is fitted on the training fold first; Fairlearn then only sees
    numbers, which avoids it silently dropping DataFrame column names.
    """
    cons = DemographicParity() if constraint == "demographic_parity" else EqualizedOdds()
    eg = ExponentiatedGradient(clone(estimator), constraints=cons, eps=eps, max_iter=max_iter)
    eg.fit(Xt, np.asarray(y), sensitive_features=pd.Series(np.asarray(s)).astype(str).to_numpy())
    return eg


def eg_scores(eg: ExponentiatedGradient, Xt: np.ndarray) -> np.ndarray:
    return np.asarray(eg._pmf_predict(Xt)[:, 1], float)


# needs_sensitive_at_prediction: the score for a student depends on their group, either because the
# attribute is a model input ("aware") or because a group-specific rule is applied afterwards.
MITIGATION_INFO: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "No mitigation (aware)",
        "stage": "none",
        "needs_sensitive_at_prediction": True,
        "uses_sensitive_input": True,
        "about": "The model sees every feature, including the sensitive one.",
    },
    "blind": {
        "label": "Fairness through unawareness",
        "stage": "pre",
        "needs_sensitive_at_prediction": False,
        "uses_sensitive_input": False,
        "about": "The sensitive column is simply removed. Proxies can still carry the signal.",
    },
    "reweighing": {
        "label": "Reweighing (aware)",
        "stage": "pre",
        "needs_sensitive_at_prediction": True,
        "uses_sensitive_input": True,
        "about": "Training rows are re-weighted so group and outcome look independent; the attribute stays an input.",
    },
    "blind_reweighing": {
        "label": "Blind + Reweighing (deployed)",
        "stage": "pre",
        "needs_sensitive_at_prediction": False,
        "uses_sensitive_input": False,
        "about": "What the dashboard ships: no sensitive input, reweighed training.",
    },
    "expgrad_dp": {
        "label": "Exponentiated Gradient (DP)",
        "stage": "in",
        "needs_sensitive_at_prediction": True,
        "uses_sensitive_input": True,
        "about": "Constrained training that targets equal flag rates (demographic parity). Randomised predictions.",
    },
    "expgrad_eo": {
        "label": "Exponentiated Gradient (EO)",
        "stage": "in",
        "needs_sensitive_at_prediction": True,
        "uses_sensitive_input": True,
        "about": "Constrained training that targets equal error rates (equalized odds). Randomised predictions.",
    },
    "thesis_thresholds": {
        "label": "Group thresholds (thesis)",
        "stage": "post",
        "needs_sensitive_at_prediction": True,
        "uses_sensitive_input": True,
        "about": "Per-group cut-offs chosen on a validation split: minimise the FPR gap, then the TPR gap, "
        "then maximise F1.",
    },
    "threshold_opt_eo": {
        "label": "ThresholdOptimizer (EO)",
        "stage": "post",
        "needs_sensitive_at_prediction": True,
        "uses_sensitive_input": True,
        "about": "Fairlearn's Hardt-style post-processing for equalized odds.",
    },
    "threshold_opt_dp": {
        "label": "ThresholdOptimizer (DP)",
        "stage": "post",
        "needs_sensitive_at_prediction": True,
        "uses_sensitive_input": True,
        "about": "Fairlearn post-processing for demographic parity.",
    },
}
