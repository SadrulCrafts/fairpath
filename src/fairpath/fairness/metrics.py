"""Group fairness metrics with uncertainty.

Convention: the positive class is **at risk** and a prediction of 1 means "flag this
student for outreach". Read the metrics with that in mind:

* flag rate (selection rate)  -- demographic parity compares these across groups;
* TPR (recall)  -- of students who really were at risk, how many we caught. A TPR gap is
  an *equal opportunity* gap: one group's struggling students get missed more often;
* FPR  -- of students who would have been fine, how many we flagged anyway. A gap means
  one group carries more unnecessary interventions and the stigma that can come with them;
* PPV (precision)  -- *predictive parity*: does a flag mean the same thing for every group?

All gaps use max-minus-min across groups (worst case), as in the thesis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METRIC_EXPLAINERS = {
    "spd": (
        "Demographic parity difference",
        "Difference in how often each group gets flagged. 0 means every group is flagged at the same rate.",
    ),
    "di_ratio": (
        "Disparate impact ratio",
        "Lowest flag rate divided by the highest. 1 is parity; below 0.8 fails the 'four-fifths' rule of thumb.",
    ),
    "tpr_gap": (
        "Equal opportunity gap",
        "Difference in how many truly at-risk students each group gets caught. A gap means some groups' "
        "struggling students are missed more often.",
    ),
    "fpr_gap": (
        "False positive rate gap",
        "Difference in how often students who would have been fine are flagged anyway.",
    ),
    "fnr_gap": (
        "False negative rate gap",
        "Difference in how often at-risk students are missed (the mirror image of the TPR gap).",
    ),
    "eodds": (
        "Equalized odds difference",
        "The larger of the TPR and FPR gaps. 0 means the model makes the same kinds of mistakes for everyone.",
    ),
    "ppv_gap": (
        "Predictive parity gap",
        "Difference in precision: whether a flag means the same thing for every group.",
    ),
    "calibration_gap": (
        "Calibration gap",
        "Difference between groups in (average predicted risk minus observed rate). "
        "0 means a 40% risk means 40% for everyone.",
    ),
    "accuracy_gap": ("Accuracy gap", "Difference in overall accuracy between groups."),
}

GAP_KEYS = [
    "spd",
    "di_ratio",
    "tpr_gap",
    "fpr_gap",
    "fnr_gap",
    "eodds",
    "ppv_gap",
    "calibration_gap",
    "accuracy_gap",
]


def _safe_div(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    out = np.full_like(a, np.nan, dtype=float)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _counts(y: np.ndarray, pred: np.ndarray, codes: np.ndarray, k: int, scores: np.ndarray | None):
    n = np.bincount(codes, minlength=k)
    pos = np.bincount(codes, weights=y, minlength=k)
    flag = np.bincount(codes, weights=pred, minlength=k)
    tp = np.bincount(codes, weights=y * pred, minlength=k)
    fp = flag - tp
    fn = pos - tp
    tn = n - pos - fp
    msc = np.bincount(codes, weights=scores, minlength=k) if scores is not None else None
    return n, pos, flag, tp, fp, fn, tn, msc


def _gaps_from_counts(n, pos, flag, tp, fp, fn, tn, msc) -> dict[str, float]:
    valid = n > 0
    flag_rate = _safe_div(flag, n)[valid]
    tpr = _safe_div(tp, pos)[valid]
    fpr = _safe_div(fp, n - pos)[valid]
    ppv = _safe_div(tp, flag)[valid]
    acc = _safe_div(tp + tn, n)[valid]

    def rng(v):
        v = v[np.isfinite(v)]
        return float(v.max() - v.min()) if len(v) >= 2 else float("nan")

    out = {
        "spd": rng(flag_rate),
        "di_ratio": float(flag_rate.min() / flag_rate.max()) if flag_rate.max() > 0 else float("nan"),
        "tpr_gap": rng(tpr),
        "fpr_gap": rng(fpr),
        "fnr_gap": rng(1 - tpr),
        "ppv_gap": rng(ppv),
        "accuracy_gap": rng(acc),
    }
    out["eodds"] = float(np.nanmax([out["tpr_gap"], out["fpr_gap"]]))
    if msc is not None:
        calib = (_safe_div(msc, n) - _safe_div(pos, n))[valid]
        out["calibration_gap"] = rng(calib)
    else:
        out["calibration_gap"] = float("nan")
    return out


def encode_groups(groups) -> tuple[np.ndarray, list]:
    s = pd.Series(np.asarray(groups)).astype(str)
    levels = sorted(s.unique().tolist())
    codes = s.map({g: i for i, g in enumerate(levels)}).to_numpy()
    return codes, levels


def gaps(y, pred, groups, scores=None) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    codes, levels = encode_groups(groups)
    sc = None if scores is None else np.asarray(scores, float)
    return _gaps_from_counts(*_counts(y, pred, codes, len(levels), sc))


def group_table(
    y, pred, groups, scores=None, labels: dict | None = None, small_n: int = 30, caution_n: int = 100
) -> list[dict[str, Any]]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    codes, levels = encode_groups(groups)
    sc = None if scores is None else np.asarray(scores, float)
    n, pos, flag, tp, fp, fn, tn, msc = _counts(y, pred, codes, len(levels), sc)
    rows = []
    for i, g in enumerate(levels):
        if n[i] == 0:
            continue
        neg = n[i] - pos[i]
        row = {
            "group": g,
            "label": (labels or {}).get(g, g),
            "n": int(n[i]),
            "n_at_risk": int(pos[i]),
            "base_rate": float(pos[i] / n[i]),
            "flag_rate": float(flag[i] / n[i]),
            "tpr": float(tp[i] / pos[i]) if pos[i] else None,
            "fpr": float(fp[i] / neg) if neg else None,
            "fnr": float(fn[i] / pos[i]) if pos[i] else None,
            "ppv": float(tp[i] / flag[i]) if flag[i] else None,
            "accuracy": float((tp[i] + tn[i]) / n[i]),
            "mean_score": float(msc[i] / n[i]) if msc is not None else None,
            "reliability": "unreliable" if n[i] < small_n else ("caution" if n[i] < caution_n else "ok"),
        }
        rows.append(row)
    return rows


def fairness_report(
    y,
    scores,
    groups,
    threshold: float | dict[str, float] = 0.5,
    labels: dict | None = None,
    n_boot: int = 1000,
    n_perm: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    small_n: int = 30,
    caution_n: int = 100,
) -> dict[str, Any]:
    """Per-group table, gap metrics, bootstrap CIs and permutation-test p-values.

    ``threshold`` may be a single number or a {group: threshold} map (post-processing).
    """
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    if isinstance(threshold, dict):
        thr = np.array([threshold.get(v, 0.5) for v in g])
    else:
        thr = np.full(len(g), float(threshold))
    pred = (scores >= thr).astype(int)

    codes, levels = encode_groups(g)
    k = len(levels)
    obs = _gaps_from_counts(*_counts(y.astype(float), pred.astype(float), codes, k, scores))

    rng = np.random.default_rng(seed)
    strata = codes * 2 + y
    strata_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    boot = {key: [] for key in GAP_KEYS}
    for _ in range(n_boot):
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata_idx])
        r = _gaps_from_counts(
            *_counts(y[idx].astype(float), pred[idx].astype(float), codes[idx], k, scores[idx])
        )
        for key in GAP_KEYS:
            boot[key].append(r[key])

    perm_counts = {key: 0 for key in ["spd", "tpr_gap", "fpr_gap", "eodds"]}
    for _ in range(n_perm):
        pc = rng.permutation(codes)
        r = _gaps_from_counts(*_counts(y.astype(float), pred.astype(float), pc, k, None))
        for key in perm_counts:
            if r[key] >= obs[key] - 1e-12:
                perm_counts[key] += 1

    a = (1 - ci) / 2
    gap_rows = {}
    for key in GAP_KEYS:
        arr = np.asarray(boot[key], float)
        arr = arr[np.isfinite(arr)]
        entry = {
            "value": obs[key],
            "lo": float(np.quantile(arr, a)) if len(arr) else None,
            "hi": float(np.quantile(arr, 1 - a)) if len(arr) else None,
            "boot_sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "label": METRIC_EXPLAINERS[key][0],
        }
        if key in perm_counts:
            entry["perm_p"] = float((perm_counts[key] + 1) / (n_perm + 1))
        gap_rows[key] = entry

    table = group_table(y, pred, g, scores, labels, small_n, caution_n)
    warnings = []
    for row in table:
        if row["reliability"] == "unreliable":
            warnings.append(f"Group '{row['label']}' has only {row['n']} students; its rates are very noisy.")
        elif row["reliability"] == "caution":
            warnings.append(f"Group '{row['label']}' has {row['n']} students; treat its rates with caution.")
    return {"groups": table, "gaps": gap_rows, "warnings": warnings, "n": len(y)}


def threshold_sweep(y, scores, groups, grid: np.ndarray | None = None) -> dict[str, Any]:
    """Per-group confusion counts at every threshold on a grid.

    The dashboard uses this to recompute any fairness metric for any combination of
    per-group thresholds in the browser, without another server round-trip.
    """
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.951, 0.01), 2)
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    out: dict[str, Any] = {"thresholds": grid.tolist(), "groups": {}}
    for level in sorted(set(g)):
        m = g == level
        yy, ss = y[m], scores[m]
        tp, fp, fn, tn = [], [], [], []
        for t in grid:
            pr = ss >= t
            tp.append(int((pr & (yy == 1)).sum()))
            fp.append(int((pr & (yy == 0)).sum()))
            fn.append(int((~pr & (yy == 1)).sum()))
            tn.append(int((~pr & (yy == 0)).sum()))
        out["groups"][level] = {"n": int(m.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out
