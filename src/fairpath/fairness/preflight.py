"""The practitioner pre-flight check: should we even try fairness mitigation here?

v1 is the three-condition rule from the thesis (section 6.4.3):

1. smallest group in a validation split has >= 100 students,
2. the model's |SPD| is >= 3x the |SPD| a random (stratified dummy) predictor gets,
3. the bootstrap coefficient of variation of |SPD| is <= 50%.

While rebuilding it I hit two problems with v1, and v2 is my attempt to fix them:

* Check 3 divides by the point estimate. If a dataset is genuinely close to fair, |SPD|
  is near zero and the CV explodes, so v1 tells you "do not mitigate" for a fair dataset
  and for a hopelessly small one, without telling you which.
* Checks 2 and 3 need a trained model, so the rule is not quite "classifier-free".

v2 asks three separate questions, each answered with a standard statistical tool:

1. **Precision** -- with these group sizes, how wide is the 95% confidence interval of a
   flag-rate gap? If its half-width is bigger than the reduction we hope to achieve
   (default 0.05), a successful mitigation could not be told apart from noise. This is the
   thesis' own standard-error argument made explicit, and it needs no model at all. (The
   stricter minimum detectable effect at 80% power is reported alongside.)
2. **Signal** -- is the observed gap distinguishable from chance? (permutation test)
3. **Room** -- is the gap's 95% CI lower bound at least the target reduction?

Verdicts: *mitigate and measure*; *clear gap, coarse measurement* (signal and room pass and the
gap's lower bound exceeds the minimum detectable change, but the interval is wider than the
target, so mitigation is worth trying but only large changes can be confirmed);
*no detectable disparity - monitor*; or *insufficient data - report intervals only*.

The "coarse" verdict was added after the first full run: debtor status on the dropout data had a
flag-rate gap of 0.49 (permutation p < 0.001) but failed the precision check by 0.006 because
debtors are a small group. Calling that "insufficient data" would be wrong; the gap is not in doubt,
only the resolution with which a *change* in it can be measured.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fairpath.fairness.metrics import encode_groups, gaps


def _fold_ids(n: int, k: int, rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    ids = np.empty(n, dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        ids[idx] = np.arange(len(idx)) % k
    return ids


def preflight_v1(
    y,
    scores,
    groups,
    threshold: float = 0.5,
    n_folds: int = 5,
    inner_val: float = 0.2,
    n_boot: int = 1000,
    min_n: int = 100,
    ratio: float = 3.0,
    max_cv: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Thesis rule, computed on out-of-fold predictions at test-fold scale."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    pred = (scores >= threshold).astype(int)
    _, levels = encode_groups(g)
    counts = pd.Series(g).value_counts()

    # Check 1: expected size of the smallest group in a validation split
    val_frac = (1 - 1 / n_folds) * inner_val
    min_val_n = int(np.floor(counts.min() * val_frac))

    # Check 2: model |SPD| vs stratified-dummy |SPD|, both averaged over test folds
    folds = _fold_ids(len(y), n_folds, rng, y)
    model_spd, dummy_spd = [], []
    flag_rate = pred.mean()
    for f in range(n_folds):
        m = folds == f
        model_spd.append(gaps(y[m], pred[m], g[m])["spd"])
        for _ in range(40):
            rand = (rng.random(m.sum()) < flag_rate).astype(int)
            dummy_spd.append(gaps(y[m], rand, g[m])["spd"])
    spd_model, spd_dummy = float(np.mean(model_spd)), float(np.mean(dummy_spd))
    noise_ratio = spd_model / spd_dummy if spd_dummy > 0 else float("inf")

    # Check 3: bootstrap CV of |SPD| on a test-fold-sized resample
    size = len(y) // n_folds
    boot = [
        gaps(y[idx], pred[idx], g[idx])["spd"]
        for idx in (rng.integers(0, len(y), size) for _ in range(n_boot))
    ]
    boot = np.asarray(boot)
    boot_cv = float(boot.std(ddof=1) / abs(spd_model)) if spd_model > 0 else float("inf")

    checks = [
        {
            "id": 1,
            "name": "Per-group validation size",
            "criterion": f">= {min_n}",
            "value": min_val_n,
            "passed": min_val_n >= min_n,
        },
        {
            "id": 2,
            "name": "Model |SPD| vs noise floor",
            "criterion": f">= {ratio:.0f}x",
            "value": round(noise_ratio, 2),
            "passed": noise_ratio >= ratio,
            "detail": {"spd_model": spd_model, "spd_dummy": spd_dummy},
        },
        {
            "id": 3,
            "name": "Bootstrap CV of |SPD|",
            "criterion": f"<= {max_cv:.0%}",
            "value": round(boot_cv, 3),
            "passed": boot_cv <= max_cv,
        },
    ]
    ok = all(c["passed"] for c in checks)
    return {
        "version": 1,
        "checks": checks,
        "passed": ok,
        "verdict": "Attempt mitigation" if ok else "Do not mitigate",
        "groups": {k: int(v) for k, v in counts.items()},
    }


def minimum_detectable_gap(n1: int, n2: int, p: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference in proportions a two-sided z-test detects with the given power."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))


def preflight_v2(
    y,
    scores,
    groups,
    threshold: float = 0.5,
    target: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.8,
    n_boot: int = 1000,
    n_perm: int = 2000,
    metric: str = "spd",
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    scores = np.asarray(scores, float)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    pred = (scores >= threshold).astype(int)
    codes, levels = encode_groups(g)
    counts = np.bincount(codes)

    # 1. precision -- uses only group sizes and the overall flag rate
    p = max(min(pred.mean(), 0.99), 0.01)
    n1, n2 = (int(c) for c in np.sort(counts)[:2])
    half_width = float(stats.norm.ppf(1 - alpha / 2) * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))
    mde = minimum_detectable_gap(n1, n2, p, alpha, power)

    # 2. signal -- permutation test on the chosen gap
    obs = gaps(y, pred, g, scores)[metric]
    hits = 0
    for _ in range(n_perm):
        if gaps(y, pred, rng.permutation(g))[metric] >= obs - 1e-12:
            hits += 1
    p_perm = (hits + 1) / (n_perm + 1)

    # 3. room -- stratified bootstrap CI of the gap
    strata = codes * 2 + y
    strata_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata_idx])
        boot.append(gaps(y[idx], pred[idx], g[idx])[metric])
    lo, hi = np.quantile(boot, [0.025, 0.975])

    checks = [
        {
            "id": 1,
            "name": "Precision: 95% CI half-width of a gap",
            "criterion": f"<= {target:.2f}",
            "value": round(half_width, 3),
            "passed": half_width <= target,
            "detail": {"mde_80_power": round(mde, 3), "smallest_groups": [n1, n2]},
        },
        {
            "id": 2,
            "name": f"Signal: permutation p-value ({metric.upper()})",
            "criterion": f"< {alpha}",
            "value": round(p_perm, 4),
            "passed": p_perm < alpha,
        },
        {
            "id": 3,
            "name": "Room: lower 95% CI bound of the gap",
            "criterion": f">= {target:.2f}",
            "value": round(float(lo), 3),
            "passed": lo >= target,
            "detail": {"gap": obs, "ci": [lo, hi]},
        },
    ]
    clear_gap = checks[1]["passed"] and checks[2]["passed"]
    if clear_gap and checks[0]["passed"]:
        verdict, code = "Mitigate and measure", "mitigate"
    elif clear_gap and lo >= mde:
        # even an imprecise study could confirm removing most of a gap this large
        verdict = (
            "Clear gap: mitigate, but only changes larger than about "
            f"{mde:.2f} can be confirmed on data this size"
        )
        code = "mitigate_coarse"
    elif not checks[0]["passed"]:
        verdict, code = "Insufficient data: report intervals, do not claim mitigation success", "insufficient"
    else:
        verdict, code = "No disparity worth mitigating detected: monitor", "monitor"
    return {
        "version": 2,
        "metric": metric,
        "checks": checks,
        "verdict": verdict,
        "verdict_code": code,
        "gap": obs,
        "ci": [float(lo), float(hi)],
        "groups": {lvl: int(c) for lvl, c in zip(levels, counts, strict=True)},
    }
