"""Actionable counterfactuals: "what realistic support could move this student to lower risk?"

Design choices, all deliberate:

* Only features marked ``actionable`` in the dataset YAML can change. Protected or
  immutable features (gender, age, nationality, parents' education, past failures, ...)
  are never touched, so the tool cannot suggest "be a different person".
* Changes are framed as **support the institution can offer** ("loan a laptop", "offer
  a payment plan"), never as a judgement of the student.
* The search is exhaustive over small sets of changes (up to ``max_changes``), so the
  result is deterministic and easy to audit. With the handful of actionable features in
  these datasets that is a few thousand candidates, scored in one batched model call.
* We prefer fewer changes, then smaller changes, and we return a few *different* options
  rather than three near-copies of the same one.

DiCE (Mothilal et al., 2020) is the usual library for this; I wrote a small, transparent
search instead because DiCE's random/genetic methods are non-deterministic and pull in
heavy pinned dependencies, and the action spaces here are small enough to enumerate.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from fairpath.config import DatasetSpec

MIN_EXTRA_REDUCTION = 0.05


def candidate_values(spec: DatasetSpec, feature: str, current: Any, schema_field: dict) -> list[Any]:
    f = spec.feature(feature)
    act = f.actionable
    if act is None:
        return []
    if act.to is not None:
        target = act.to
        if f.type == "binary":
            target = float(target)
        elif f.type == "categorical":
            target = str(target)
        return [] if str(current) == str(target) else [target]
    step = float(f.step or 1)
    lo, hi = schema_field.get("data_min", f.min), schema_field.get("data_max", f.max)
    vals = []
    if act.direction == "increase":
        for k in (1, 2, 3):
            v = current + k * step
            if v <= hi:
                vals.append(float(v))
    else:
        if feature == "absences":  # long-tailed count: try meaningful cuts, not one-at-a-time
            vals = [float(v) for v in (current * 0.5, current * 0.25, 2.0, 0.0) if v < current]
            vals = sorted(set(round(v) for v in vals), reverse=True)
        else:
            for k in (1, 2, 3):
                v = current - k * step
                if v >= lo:
                    vals.append(float(v))
    return vals


def _distance(spec, feature, old, new, schema_field) -> float:
    f = spec.feature(feature)
    if f.type in ("binary", "categorical"):
        return 1.0
    span = max(float(schema_field.get("data_max", 1)) - float(schema_field.get("data_min", 0)), 1e-9)
    return abs(float(new) - float(old)) / span * 4  # a full-range move costs about 4 binary flips


def find_counterfactuals(
    x: pd.Series,
    predict_risk: Callable[[pd.DataFrame], np.ndarray],
    spec: DatasetSpec,
    schema: dict,
    target_risk: float,
    max_changes: int = 3,
    n_results: int = 3,
) -> dict[str, Any]:
    fields = {f["name"]: f for f in schema["fields"]}
    actionable = [f.name for f in spec.features if f.is_actionable and f.name in x.index]
    options = {f: candidate_values(spec, f, x[f], fields[f]) for f in actionable}
    options = {f: v for f, v in options.items() if v}
    current_risk = float(predict_risk(x.to_frame().T)[0])

    combos: list[dict[str, Any]] = []
    feats = list(options)
    for r in range(1, min(max_changes, len(feats)) + 1):
        for subset in itertools.combinations(feats, r):
            for values in itertools.product(*(options[f] for f in subset)):
                combos.append(dict(zip(subset, values, strict=True)))
    if not combos:
        return {
            "current_risk": current_risk,
            "target_risk": target_risk,
            "options": [],
            "reachable": False,
            "actionable_features": actionable,
            "message": "None of the recorded features are things an institution could change for this student.",
        }

    rows = pd.DataFrame([x.to_dict()] * len(combos))
    for i, c in enumerate(combos):
        for f, v in c.items():
            rows.at[i, f] = v
    for col in x.index:
        rows[col] = rows[col].astype(object if spec.feature(col).type == "categorical" else float)
    risks = predict_risk(rows)

    scored = []
    for c, r in zip(combos, risks, strict=True):
        cost = sum(_distance(spec, f, x[f], v, fields[f]) for f, v in c.items())
        scored.append((c, float(r), cost))

    reach = [s for s in scored if s[1] < target_risk]
    reachable = bool(reach)  # evaluated once: list.sort() empties the list while sorting
    pool = list(reach) if reachable else sorted(scored, key=lambda s: s[1])[:50]
    if reachable:
        pool.sort(key=lambda s: (len(s[0]), s[2], s[1]))  # fewest, then smallest, changes first
    else:
        pool.sort(key=lambda s: (s[1], len(s[0]), s[2]))  # otherwise: whatever helps most

    chosen: list[tuple[dict, float, float]] = []
    for c, r, cost in pool:
        keys = set(c)
        # a bigger package that contains an option we already have must earn its place:
        # it has to lower the risk by at least 5 more percentage points
        if any(set(prev).issubset(keys) and r > prev_r - MIN_EXTRA_REDUCTION for prev, prev_r, _ in chosen):
            continue
        chosen.append((c, r, cost))
        if len(chosen) >= n_results:
            break

    out = []
    for c, r, cost in chosen:
        changes = []
        for f, v in c.items():
            fs = spec.feature(f)
            changes.append(
                {
                    "feature": f,
                    "label": fs.label,
                    "from": _plain(x[f]),
                    "to": _plain(v),
                    "action": fs.actionable.action if fs.actionable else "",
                }
            )
        out.append(
            {
                "changes": changes,
                "new_risk": r,
                "risk_reduction": current_risk - r,
                "n_changes": len(c),
                "cost": round(cost, 3),
            }
        )
    return {
        "current_risk": current_risk,
        "target_risk": target_risk,
        "reachable": reachable,
        "options": out,
        "actionable_features": actionable,
        "message": (
            None
            if reachable
            else "No combination of up to three supported changes brings the predicted risk below the target; "
            "these are the changes that help most."
        ),
    }


def _plain(v: Any) -> Any:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    if isinstance(v, np.generic):
        return v.item()
    return v
