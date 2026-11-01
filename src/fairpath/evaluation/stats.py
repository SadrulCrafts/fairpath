"""Statistical tests for comparing models and mitigation methods across CV folds.

Fixes relative to the thesis code:

* effect size is the matched-pairs **rank-biserial correlation**
  r = (W+ - W-) / (W+ + W-), as the thesis text describes (the old code derived r from
  the p-value, which is a different quantity and breaks when p == 0);
* p-values from families of tests are Holm-corrected;
* fold scores from repeated CV are not independent, so model comparisons also use the
  Nadeau & Bengio (2003) corrected resampled t-test, which widens the variance by
  (1/k + n_test/n_train) instead of pretending the 25 folds are independent.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def effect_label(r: float) -> str:
    r = abs(r)
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "small"
    if r < 0.5:
        return "medium"
    return "large"


def wilcoxon_paired(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    d = a - b
    if np.allclose(d, 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "effect": "negligible",
            "median_diff": 0.0,
            "n": len(a),
        }
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    r = rank_biserial(a, b)
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "rank_biserial": r,
        "effect": effect_label(r),
        "median_diff": float(np.median(d)),
        "n": len(a),
    }


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def corrected_resampled_ttest(a, b, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio corrected t-test for repeated k-fold CV scores."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return {"t": 0.0, "p_value": 1.0, "mean_diff": float(d.mean())}
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"t": float(t), "p_value": float(p), "mean_diff": float(d.mean())}


def corrected_ci(scores, n_train: int, n_test: int, ci: float = 0.95) -> tuple[float, float]:
    s = np.asarray(scores, float)
    k = len(s)
    se = np.sqrt((1 / k + n_test / n_train) * np.var(s, ddof=1))
    h = stats.t.ppf(0.5 + ci / 2, df=k - 1) * se
    return float(s.mean() - h), float(s.mean() + h)


def friedman(*samples) -> dict:
    stat, p = stats.friedmanchisquare(*samples)
    n, k = len(samples[0]), len(samples)
    return {"statistic": float(stat), "p_value": float(p), "kendalls_w": float(stat / (n * (k - 1)))}


def p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
