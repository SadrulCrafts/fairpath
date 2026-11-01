"""Model comparison on the development split.

1. Optuna tunes each model family with 5-fold CV on the development set.
2. Each tuned model (plus two dummy baselines) is scored with 5x5 RepeatedStratifiedKFold.
3. Differences from the best model are tested with the Nadeau-Bengio corrected t-test and
   the Wilcoxon signed-rank test (Holm-corrected).

The held-out test set is not touched here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from fairpath.config import Settings
from fairpath.data.loaders import Dataset
from fairpath.evaluation.metrics import classification_metrics, reliability_table
from fairpath.evaluation.stats import corrected_ci, corrected_resampled_ttest, holm, wilcoxon_paired
from fairpath.fairness.metrics import gaps
from fairpath.models import MODEL_LABELS, make_pipeline, tune
from fairpath.pipeline.common import dev_test_split, provenance, read_json, reports_dir, timed, write_json

log = logging.getLogger(__name__)

PERF_KEYS = [
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "log_loss",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "flag_rate",
]
FAIR_KEYS = ["spd", "eodds", "tpr_gap", "fpr_gap", "di_ratio"]
RM_KEYS = ["f1", "precision", "recall", "balanced_accuracy", "flag_rate"]


def tune_all(ds: Dataset, settings: Settings, Xdev: pd.DataFrame, ydev: np.ndarray, blind: bool) -> dict:
    feats = list(Xdev.columns)
    out = {}
    for m in settings.models:
        if not settings.tuning["enabled"]:
            out[m] = {"params": {}, "cv_auc": None}
            continue
        with timed(f"tune {m} on {ds.name}"):
            params, score = tune(
                m,
                lambda p, m=m: make_pipeline(m, ds.spec, feats, p, settings.seed),
                Xdev,
                ydev,
                n_trials=settings.tuning["n_trials"],
                n_folds=settings.cv["tuning_folds"],
                seed=settings.seed,
            )
        out[m] = {"params": params, "cv_auc": score}
    return out


def run(ds: Dataset, settings: Settings, retune: bool = True) -> dict[str, Any]:
    split = dev_test_split(ds, settings)
    blind = False  # model comparison uses every feature, like the thesis
    feats = ds.spec.model_features(blind)
    Xdev = ds.X.iloc[split.dev_idx][feats].reset_index(drop=True)
    ydev = ds.y[split.dev_idx]
    sdev = ds.sensitive.iloc[split.dev_idx][ds.spec.primary_sensitive].astype(str).to_numpy()
    out_dir = reports_dir(settings, ds) / ds.name
    params_path = out_dir / "tuned_params.json"

    if retune or not params_path.exists():
        tuned = tune_all(ds, settings, Xdev, ydev, blind)
        write_json(params_path, tuned)
    else:
        tuned = read_json(params_path)

    cvb = settings.cv["benchmark"]
    cv = RepeatedStratifiedKFold(
        n_splits=cvb["n_splits"], n_repeats=cvb["n_repeats"], random_state=settings.seed
    )
    splits = list(cv.split(Xdev, ydev))
    candidates = list(settings.models) + ["dummy_most_frequent", "dummy_stratified"]
    fold_rows: list[dict[str, Any]] = []
    oof = {m: np.full(len(ydev), np.nan) for m in candidates}

    for m in candidates:
        params = tuned.get(m, {}).get("params", {})
        with timed(f"cv {m} on {ds.name} ({len(splits)} folds)"):
            for k, (tr, te) in enumerate(splits):
                pipe = make_pipeline(m, ds.spec, feats, params, settings.seed)
                pipe.fit(Xdev.iloc[tr], ydev[tr])
                p = pipe.predict_proba(Xdev.iloc[te])[:, 1]
                if k < cvb["n_splits"]:
                    oof[m][te] = p
                perf = classification_metrics(ydev[te], p, 0.5)
                fair = gaps(ydev[te], (p >= 0.5).astype(int), sdev[te], p)
                # rate-matched threshold: flag the same share of students as the training base rate.
                # Chosen on the training fold only, so it is leak-free and comparable across models
                # (some well-regularised models never reach 0.5 on this data).
                thr_rm = float(np.quantile(pipe.predict_proba(Xdev.iloc[tr])[:, 1], 1 - ydev[tr].mean()))
                perf_rm = classification_metrics(ydev[te], p, thr_rm)
                fair_rm = gaps(ydev[te], (p >= thr_rm).astype(int), sdev[te], p)
                fold_rows.append(
                    {
                        "model": m,
                        "fold": k,
                        **perf,
                        **{f: fair[f] for f in FAIR_KEYS},
                        **{f"{key}_rm": perf_rm[key] for key in RM_KEYS},
                        **{f"{f}_rm": fair_rm[f] for f in FAIR_KEYS},
                    }
                )

    folds = pd.DataFrame(fold_rows)
    n_test = len(splits[0][1])
    n_train = len(splits[0][0])

    summary = []
    for m in candidates:
        f = folds[folds.model == m]
        row: dict[str, Any] = {
            "model": m,
            "label": MODEL_LABELS[m],
            "params": tuned.get(m, {}).get("params", {}),
            "tuning_cv_auc": tuned.get(m, {}).get("cv_auc"),
        }
        for key in PERF_KEYS + FAIR_KEYS + [f"{k}_rm" for k in RM_KEYS + FAIR_KEYS]:
            vals = f[key].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            lo, hi = corrected_ci(vals, n_train, n_test) if len(vals) > 2 else (np.nan, np.nan)
            row[key] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "lo": lo, "hi": hi}
        summary.append(row)

    real = [r for r in summary if not r["model"].startswith("dummy")]
    best = max(real, key=lambda r: r["roc_auc"]["mean"])["model"]
    comps = []
    best_auc = folds[folds.model == best].sort_values("fold")["roc_auc"].to_numpy()
    for m in settings.models:
        if m == best:
            continue
        other = folds[folds.model == m].sort_values("fold")["roc_auc"].to_numpy()
        comps.append(
            {
                "model": m,
                "vs": best,
                "corrected_t": corrected_resampled_ttest(best_auc, other, n_train, n_test),
                "wilcoxon": wilcoxon_paired(best_auc, other),
            }
        )
    if comps:
        adj_t = holm([c["corrected_t"]["p_value"] for c in comps])
        adj_w = holm([c["wilcoxon"]["p_value"] for c in comps])
        for c, a, b in zip(comps, adj_t, adj_w, strict=True):
            c["corrected_t"]["p_holm"] = a
            c["wilcoxon"]["p_holm"] = b

    calibration = {
        m: reliability_table(
            ydev[np.isfinite(oof[m])], oof[m][np.isfinite(oof[m])], settings.calibration["n_bins"]
        )
        for m in settings.models
    }
    payload = {
        "dataset": ds.name,
        "provenance": provenance(ds),
        "cv": f"{cvb['n_splits']}x{cvb['n_repeats']} " "RepeatedStratifiedKFold on the development split",
        "n_dev": len(ydev),
        "primary_axis": ds.spec.primary_sensitive,
        "threshold": 0.5,
        "threshold_note": "Metrics without a suffix use a 0.5 cut-off; '_rm' metrics use a rate-matched cut-off "
        "chosen on each training fold so that the flag rate equals the training base rate.",
        "summary": summary,
        "best_model": best,
        "comparisons": comps,
        "calibration": calibration,
        "folds": folds.round(6).to_dict(orient="records"),
    }
    write_json(out_dir / "benchmark.json", payload)
    return payload
