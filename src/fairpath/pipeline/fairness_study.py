"""Bias mitigation compared side by side, inside one leak-free cross-validation loop.

For each fold of 5x2 RepeatedStratifiedKFold on the development split:

    outer-train --> inner-train (80%) + validation (20%)    [post-processing tuning only]
    outer-test  --> every method is scored on exactly the same rows

Score-based methods use a *rate-matched* cut-off chosen on the training fold (flag the
training base rate), so every method flags roughly the same number of students and the
comparison is about *who* gets flagged, not how many.

Methods: baseline (aware), blind, Reweighing, blind + Reweighing, Exponentiated Gradient
(demographic parity / equalized odds), the thesis' group-threshold search, and Fairlearn's
ThresholdOptimizer (DP / EO). Each is compared with the baseline using Wilcoxon tests
(Holm-corrected) on fold-level gaps and F1, and summarised as a fairness-accuracy Pareto
frontier.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from fairpath.config import Settings
from fairpath.data.loaders import Dataset, intersection_series
from fairpath.evaluation.stats import holm, wilcoxon_paired
from fairpath.fairness.metrics import gaps
from fairpath.mitigation import (
    MITIGATION_INFO,
    apply_group_thresholds,
    eg_scores,
    fit_exponentiated_gradient,
    fit_threshold_optimizer,
    reweighing_weights,
    thesis_threshold_search,
    threshold_optimizer_predict,
)
from fairpath.models import MODEL_LABELS, fit_pipeline, make_pipeline
from fairpath.pipeline.common import dev_test_split, provenance, read_json, reports_dir, timed, write_json

log = logging.getLogger(__name__)

ALL_METHODS = [
    "baseline",
    "blind",
    "reweighing",
    "blind_reweighing",
    "expgrad_dp",
    "expgrad_eo",
    "thesis_thresholds",
    "threshold_opt_eo",
    "threshold_opt_dp",
]
METRICS = [
    "roc_auc",
    "f1",
    "balanced_accuracy",
    "flag_rate",
    "spd",
    "eodds",
    "tpr_gap",
    "fpr_gap",
    "ppv_gap",
    "di_ratio",
]


def _score(y, scores, pred, groups) -> dict[str, float]:
    g = gaps(y, pred, groups)
    auc = roc_auc_score(y, scores) if len(np.unique(scores)) > 1 else 0.5
    return {
        "roc_auc": auc,
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "flag_rate": float(np.mean(pred)),
        **{k: g[k] for k in ["spd", "eodds", "tpr_gap", "fpr_gap", "ppv_gap", "di_ratio"]},
    }


def axis_groups(ds: Dataset, axis: str) -> pd.Series:
    if "+" in axis:
        return intersection_series(ds, axis.split("+"))
    return ds.sensitive[axis].astype(str)


def run_axis(
    ds: Dataset,
    settings: Settings,
    axis: str,
    base_model: str,
    params: dict,
    methods: list[str],
    idx: np.ndarray,
) -> pd.DataFrame:
    spec = ds.spec
    aware = spec.model_features(blind=False)
    # "blind" removes the attribute(s) being audited when they are model inputs
    axis_cols = [a for a in axis.split("+") if a in aware]
    blind_feats = [f for f in aware if f not in axis_cols] if axis_cols else spec.model_features(blind=True)
    X = ds.X.iloc[idx].reset_index(drop=True)
    y = ds.y[idx]
    s = axis_groups(ds, axis).iloc[idx].reset_index(drop=True).to_numpy()
    cfg = settings.cv["fairness"]
    cv = RepeatedStratifiedKFold(
        n_splits=cfg["n_splits"], n_repeats=cfg["n_repeats"], random_state=settings.seed
    )
    grid = np.linspace(
        settings.fairness["thesis_threshold_grid"]["min"],
        settings.fairness["thesis_threshold_grid"]["max"],
        settings.fairness["thesis_threshold_grid"]["steps"],
    )
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, Xte, ytr, yte, s_tr, s_te = X.iloc[tr], X.iloc[te], y[tr], y[te], s[tr], s[te]
        mk = lambda feats: make_pipeline(base_model, spec, feats, params, settings.seed)
        w = reweighing_weights(ytr, s_tr)

        rate = ytr.mean()

        def rm(pipe, feats):
            """Rate-matched cut-off from the training fold: flag the training base rate."""
            return float(np.quantile(pipe.predict_proba(Xtr[feats])[:, 1], 1 - rate))

        base = fit_pipeline(mk(aware), Xtr[aware], ytr)
        p_base = base.predict_proba(Xte[aware])[:, 1]
        t_base = rm(base, aware)
        res: dict[str, tuple[np.ndarray, np.ndarray]] = {"baseline": (p_base, (p_base >= t_base).astype(int))}

        if "blind" in methods:
            m = fit_pipeline(mk(blind_feats), Xtr[blind_feats], ytr)
            p = m.predict_proba(Xte[blind_feats])[:, 1]
            res["blind"] = (p, (p >= rm(m, blind_feats)).astype(int))
        if "reweighing" in methods:
            m = fit_pipeline(mk(aware), Xtr[aware], ytr, w)
            p = m.predict_proba(Xte[aware])[:, 1]
            res["reweighing"] = (p, (p >= rm(m, aware)).astype(int))
        if "blind_reweighing" in methods:
            m = fit_pipeline(mk(blind_feats), Xtr[blind_feats], ytr, w)
            p = m.predict_proba(Xte[blind_feats])[:, 1]
            res["blind_reweighing"] = (p, (p >= rm(m, blind_feats)).astype(int))
        for meth, cons in (("expgrad_dp", "demographic_parity"), ("expgrad_eo", "equalized_odds")):
            if meth in methods:
                pre = clone(base.named_steps["preprocess"]).fit(Xtr[aware])
                eg = fit_exponentiated_gradient(
                    base.named_steps["model"], pre.transform(Xtr[aware]), ytr, s_tr, cons, max_iter=40
                )
                Xt_te = pre.transform(Xte[aware])
                res[meth] = (
                    eg_scores(eg, Xt_te),
                    np.asarray(eg.predict(Xt_te, random_state=settings.seed), int),
                )

        needs_inner = any(m in methods for m in ("thesis_thresholds", "threshold_opt_eo", "threshold_opt_dp"))
        if needs_inner:
            Xin, Xval, yin, yval, sin_, sval = train_test_split(
                Xtr,
                ytr,
                s_tr,
                test_size=settings.cv["inner_val_size"],
                random_state=settings.seed,
                stratify=ytr,
            )
            inner = fit_pipeline(mk(aware), Xin[aware], yin)
            if "thesis_thresholds" in methods:
                tmap = thesis_threshold_search(yval, inner.predict_proba(Xval[aware])[:, 1], sval, grid)
                res["thesis_thresholds"] = (p_base, apply_group_thresholds(p_base, s_te, tmap))
            for meth, cons in (
                ("threshold_opt_eo", "equalized_odds"),
                ("threshold_opt_dp", "demographic_parity"),
            ):
                if meth in methods:
                    to = fit_threshold_optimizer(inner, Xval[aware], yval, sval, cons, settings.seed)
                    pred, pmf = threshold_optimizer_predict(to, Xte[aware], s_te, settings.seed)
                    res[meth] = (pmf, pred)

        for meth, (sc, pr) in res.items():
            rows.append(
                {"axis": axis, "model": base_model, "method": meth, "fold": fold, **_score(yte, sc, pr, s_te)}
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (axis, model), block in df.groupby(["axis", "model"]):
        methods = []
        base = block[block.method == "baseline"].sort_values("fold")
        pvals = {k: [] for k in ("spd", "eodds", "f1")}
        for meth, b in block.groupby("method", sort=False):
            b = b.sort_values("fold")
            entry = {"method": meth, **MITIGATION_INFO[meth]}
            for k in METRICS:
                v = b[k].to_numpy(dtype=float)
                entry[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1))}
            if meth != "baseline":
                for k in pvals:
                    t = wilcoxon_paired(base[k].to_numpy(), b[k].to_numpy())
                    entry.setdefault("tests", {})[k] = t
                    pvals[k].append((meth, t["p_value"]))
            methods.append(entry)
        for k, plist in pvals.items():
            adj = holm([p for _, p in plist]) if plist else []
            for (meth, _), a in zip(plist, adj, strict=True):
                next(e for e in methods if e["method"] == meth)["tests"][k]["p_holm"] = a
        # Pareto frontier on (equalized-odds gap low, F1 high)
        pts = [(e["method"], e["eodds"]["mean"], e["f1"]["mean"]) for e in methods]
        frontier = [
            m
            for m, gx, fy in pts
            if not any((g2 <= gx and f2 >= fy) and (g2 < gx or f2 > fy) for _, g2, f2 in pts)
        ]
        for e in methods:
            e["pareto"] = e["method"] in frontier
        out.setdefault(axis, {})[model] = {
            "methods": methods,
            "frontier": frontier,
            "model_label": MODEL_LABELS[model],
        }
    return out


def recommend(axis_block: dict[str, Any]) -> dict[str, Any]:
    """Pick, per model, the method a careful practitioner would ship (no group-specific treatment
    at prediction time), and note what post-processing could add if that is legally allowed."""
    recs = {}
    for model, blk in axis_block.items():
        ms = {e["method"]: e for e in blk["methods"]}
        base = ms["baseline"]
        deployable = [e for e in blk["methods"] if not e["needs_sensitive_at_prediction"]]

        def utility(e):  # reward gap reduction, penalise F1 loss twice as hard
            return (base["eodds"]["mean"] - e["eodds"]["mean"]) - 2 * max(
                0.0, base["f1"]["mean"] - e["f1"]["mean"]
            )

        best = max(deployable, key=utility)
        post = [e for e in blk["methods"] if e["needs_sensitive_at_prediction"]]
        best_post = min(post, key=lambda e: e["eodds"]["mean"]) if post else None
        recs[model] = {
            "deployable_choice": best["method"],
            "deployable_label": best["label"],
            "eodds_change": best["eodds"]["mean"] - base["eodds"]["mean"],
            "f1_change": best["f1"]["mean"] - base["f1"]["mean"],
            "lowest_gap_post": best_post["method"] if best_post else None,
        }
    return recs


def run_preflight(
    ds: Dataset, settings: Settings, model: str, params: dict, idx: np.ndarray
) -> dict[str, Any]:
    """Thesis pre-flight (v1) and the revised version (v2) on out-of-fold predictions of the
    unmitigated, aware model -- the situation a practitioner is in before deciding anything."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from fairpath.fairness.preflight import preflight_v1, preflight_v2

    feats = ds.spec.model_features(blind=False)
    X = ds.X.iloc[idx].reset_index(drop=True)[feats]
    y = ds.y[idx]
    oof = cross_val_predict(
        make_pipeline(model, ds.spec, feats, params, settings.seed),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=settings.seed),
        method="predict_proba",
    )[:, 1]
    pf = settings.preflight
    out = {"dataset": ds.name, "model": model, "n": len(y), "threshold": 0.5, "axes": {}}
    for axis in list(ds.spec.sensitive) + ["+".join(i) for i in ds.spec.intersections]:
        g = axis_groups(ds, axis).iloc[idx].to_numpy()
        out["axes"][axis] = {
            "label": " x ".join(ds.spec.sensitive[a].label for a in axis.split("+")),
            "v1": preflight_v1(
                y,
                oof,
                g,
                min_n=pf["min_group_n"],
                ratio=pf["noise_floor_ratio"],
                max_cv=pf["max_bootstrap_cv"],
                seed=settings.seed,
                n_boot=settings.bootstrap["n_resamples"],
            ),
            "v2": preflight_v2(
                y,
                oof,
                g,
                target=pf["target_reduction"],
                alpha=pf["alpha"],
                power=pf["power"],
                n_boot=settings.bootstrap["n_resamples"],
                n_perm=settings.permutation_test["n_permutations"],
                seed=settings.seed,
            ),
        }
    return out


def run(ds: Dataset, settings: Settings) -> dict[str, Any]:
    split = dev_test_split(ds, settings)
    out_dir = reports_dir(settings, ds) / ds.name
    bench = read_json(out_dir / "benchmark.json")
    tuned = read_json(out_dir / "tuned_params.json")
    best = bench["best_model"]
    models = ["logistic_regression"] + ([best] if best != "logistic_regression" else [])
    axes = list(ds.spec.sensitive)[:2] + ["+".join(ds.spec.intersections[0])]
    frames = []
    for axis in axes:
        for m in models:
            heavy = axis == ds.spec.primary_sensitive
            methods = ALL_METHODS if heavy else [x for x in ALL_METHODS if not x.startswith("expgrad")]
            with timed(f"fairness study {ds.name} | {axis} | {m}"):
                frames.append(run_axis(ds, settings, axis, m, tuned[m]["params"], methods, split.dev_idx))
    df = pd.concat(frames, ignore_index=True)
    summary = summarise(df)
    with timed(f"pre-flight checks {ds.name}"):
        preflight = run_preflight(ds, settings, best, tuned[best]["params"], split.dev_idx)
    write_json(out_dir / "preflight.json", preflight)
    payload = {
        "dataset": ds.name,
        "provenance": provenance(ds),
        "design": f"{settings.cv['fairness']['n_splits']}x{settings.cv['fairness']['n_repeats']} "
        "RepeatedStratifiedKFold on the development split; inner 80/20 split for post-processing",
        "axes": axes,
        "models": models,
        "summary": summary,
        "recommendations": {axis: recommend(blk) for axis, blk in summary.items()},
        "methods_info": MITIGATION_INFO,
        "preflight": preflight,
        "folds": df.round(6).to_dict(orient="records"),
    }
    write_json(out_dir / "fairness_study.json", payload)
    return payload
