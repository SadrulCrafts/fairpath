"""Post-hoc probability calibration (Platt scaling or isotonic regression).

The calibrator is fitted on *out-of-fold* predictions of the development set, so it never
sees a probability the base model produced for its own training rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic", "none"):
            raise ValueError(method)
        self.method = method
        self._model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> ProbabilityCalibrator:
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if self.method == "none" or self._model is None:
            return p
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        out = self._model.predict(p)
        # isotonic gives flat steps; keep probabilities strictly inside (0, 1) for display
        return np.clip(out, 0.005, 0.995)


def choose_method(n: int, cfg: dict) -> str:
    rule = cfg["method_by_size"]
    return rule["large"] if n >= rule["cutoff"] else rule["small"]
