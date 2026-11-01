"""Performance and calibration metrics, with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_INFO = {
    "roc_auc": "ROC-AUC: chance that a random at-risk student is ranked above a random successful one.",
    "pr_auc": "PR-AUC (average precision): how precise the flags are across all recall levels.",
    "brier": "Brier score: mean squared error of the predicted probabilities (lower is better).",
    "ece": "Expected calibration error: average gap between predicted risk and observed rate.",
    "f1": "F1 for the at-risk class: balance of precision and recall at the chosen threshold.",
    "precision": "Precision: of the students flagged, the share who really were at risk.",
    "recall": "Recall: of the students really at risk, the share the model flagged.",
    "accuracy": "Accuracy: share of all students classified correctly.",
    "balanced_accuracy": "Balanced accuracy: mean of recall on each class (robust to imbalance).",
    "macro_f1": "Macro-F1: average F1 of both classes; exposes the 'always predict majority' trick.",
}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "n": int(m.sum()),
                "mean_pred": float(np.mean(p[m])),
                "frac_pos": float(np.mean(y[m])),
            }
        )
    return rows


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two_classes else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "flag_rate": float(pred.mean()),
    }


def bootstrap_ci(
    fn: Callable[[np.ndarray], float],
    n: int,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    strata: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI for any statistic of row indices.

    ``fn`` receives an index array and returns the statistic. Resampling is stratified on
    ``strata`` (e.g. outcome x group) so small groups never vanish from a resample.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    if strata is None:
        for b in range(n_resamples):
            stats[b] = fn(rng.integers(0, n, n))
    else:
        groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
        for b in range(n_resamples):
            idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
            stats[b] = fn(idx)
    stats = stats[np.isfinite(stats)]
    a = (1 - ci) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)), stats


def metrics_with_ci(y, p, threshold, n_resamples=1000, ci=0.95, seed=42) -> dict[str, dict[str, float]]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    point = classification_metrics(y, p, threshold)
    keys = ["roc_auc", "pr_auc", "brier", "ece", "f1", "precision", "recall", "balanced_accuracy"]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    draws = {k: [] for k in keys}
    for _ in range(n_resamples):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        m = classification_metrics(y[idx], p[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    a = (1 - ci) / 2
    out = {}
    for k in keys:
        arr = np.asarray(draws[k])
        out[k] = {"value": point[k], "lo": float(np.quantile(arr, a)), "hi": float(np.quantile(arr, 1 - a))}
    for k in ["accuracy", "macro_f1", "log_loss", "flag_rate"]:
        out[k] = {"value": point[k]}
    return out


def best_f1_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> float:
    grid = grid if grid is not None else np.round(np.arange(0.10, 0.81, 0.01), 2)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])
