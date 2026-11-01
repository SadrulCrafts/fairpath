"""Split-conformal prediction sets for a binary risk model.

A point estimate like "38% risk" hides how sure the model is. Conformal prediction turns
the calibrated probability into a *set* of plausible outcomes with a coverage guarantee:
over many students, the true outcome falls in the set about (1 - alpha) of the time.

* set = {at risk} or {not at risk}  -> the model is confident at this coverage level;
* set = {at risk, not at risk}      -> genuinely uncertain: a human should look closely.

Non-conformity score: 1 - p(true class)  (the "LAC" score of Sadinle et al., 2019).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_conformal(y_cal, p_cal, alpha: float = 0.1) -> float:
    y = np.asarray(y_cal, int)
    p = np.asarray(p_cal, float)
    s = np.where(y == 1, 1 - p, p)
    n = len(s)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q, method="higher"))


def prediction_set(p: float, qhat: float) -> list[int]:
    out = []
    if p <= qhat:
        out.append(0)
    if 1 - p <= qhat:
        out.append(1)
    return out


def confidence_info(p: float, qhat: float, alpha: float) -> dict[str, Any]:
    ps = prediction_set(p, qhat)
    if len(ps) == 1:
        level, text = "confident", (
            f"At {1 - alpha:.0%} coverage the model commits to one outcome for this profile."
        )
    elif len(ps) == 2:
        level, text = "uncertain", (
            "The model cannot separate the two outcomes for this profile with the usual "
            f"{1 - alpha:.0%} coverage. Treat the score as a weak signal."
        )
    else:
        level, text = "atypical", "This profile looks unlike the training data; the score may be unreliable."
    return {"prediction_set": ps, "level": level, "explanation": text, "coverage": 1 - alpha}


def coverage(y, p, qhat: float) -> dict[str, float]:
    y = np.asarray(y, int)
    sets = [prediction_set(pi, qhat) for pi in np.asarray(p, float)]
    covered = np.mean([yi in s for yi, s in zip(y, sets, strict=True)])
    size = np.mean([len(s) for s in sets])
    uncertain = np.mean([len(s) == 2 for s in sets])
    return {"coverage": float(covered), "avg_set_size": float(size), "uncertain_share": float(uncertain)}
