"""Re-run the thesis protocol and put the printed numbers next to re-computed ones.

This stage deliberately copies the *original* setup rather than the upgraded one:

* positive class = success (pass / graduate), default threshold 0.5;
* "aware" models (the sensitive attribute is an input);
* the hyper-parameters the thesis code actually used (``config.yaml`` of the original
  project -- not the ones printed in thesis section 3.3, see docs/REPRODUCIBILITY.md);
* the original preprocessing (integer-coded dropout columns treated as numeric);
* 5x2 RepeatedStratifiedKFold with seed 42 and the inner 80/20 split for thresholds.

Nothing from this stage feeds the dashboard model. It exists so that anyone can check
which thesis claims hold up and by how much.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairpath.config import Settings, load_dataset_spec, load_thesis_reported
from fairpath.data.loaders import read_raw, resolve_path
from fairpath.evaluation.stats import wilcoxon_paired
from fairpath.fairness.metrics import gaps
from fairpath.mitigation import apply_group_thresholds, reweighing_weights, thesis_threshold_search
from fairpath.pipeline.common import timed, write_json

log = logging.getLogger(__name__)
MODELS = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


def load_thesis_frame(name: str, settings: Settings) -> tuple[pd.DataFrame, np.ndarray, pd.Series, str]:
    spec = load_dataset_spec(name)
    path, source = resolve_path(spec, settings)
    df = read_raw(spec, path)
    if name == "student_mat":
        y = (df["G3"] >= 10).astype(int).to_numpy()
        X = df.drop(columns=["G1", "G2", "G3"])
        s = df["sex"].astype(str)
    else:
        df = df[df["Target"] != "Enrolled"].reset_index(drop=True)
        y = (df["Target"] == "Graduate").astype(int).to_numpy()
        X = df.drop(columns=[c for c in df.columns if c.startswith("Curricular units")] + ["Target"])
        s = df["Gender"].astype(str)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype(object)
    return X, y, s, source


def _pre(X: pd.DataFrame, dense: bool) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    num_steps = [("imp", SimpleImputer(strategy="median"))] + ([] if dense else [("sc", StandardScaler())])
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ]
    )


def thesis_model(name: str, X: pd.DataFrame) -> Pipeline:
    if name == "logistic_regression":
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
        return Pipeline([("pre", _pre(X, dense=False)), ("m", est)])
    if name == "random_forest":
        est = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=2)
        return Pipeline([("pre", _pre(X, dense=False)), ("m", est)])
    est = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=300, random_state=42)
    return Pipeline([("pre", _pre(X, dense=True)), ("m", est)])


def _fold_metrics(y, p, pred, s) -> dict[str, float]:
    g = gaps(y, pred, s)
    return {
        "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "F1": f1_score(y, pred, zero_division=0),
        "Macro_F1": f1_score(y, pred, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y, pred),
        "SPD": g["spd"],
        "TPR_gap": g["tpr_gap"],
        "FPR_gap": g["fpr_gap"],
    }


def reproduce_dataset(name: str, settings: Settings) -> dict[str, Any]:
    X, y, s, source = load_thesis_frame(name, settings)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    grid = np.linspace(0.1, 0.9, 10)
    rows: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        Xtr, Xte, ytr, yte, str_, ste = X.iloc[tr], X.iloc[te], y[tr], y[te], s.iloc[tr], s.iloc[te]
        for strat in ("most_frequent", "stratified"):
            d = DummyClassifier(strategy=strat, random_state=42).fit(Xtr, ytr)
            p = d.predict_proba(Xte)[:, 1]
            rows.append(
                {
                    "fold": fold,
                    "model": f"dummy_{strat}",
                    "condition": "baseline",
                    **_fold_metrics(yte, p, d.predict(Xte), ste),
                }
            )
        # inner split shared by every model in this fold (thesis protocol)
        Xin, Xval, yin, yval, sin, sval = train_test_split(
            Xtr, ytr, str_, test_size=0.2, random_state=42, stratify=ytr
        )
        w_outer = reweighing_weights(ytr, str_)
        w_inner = reweighing_weights(yin, sin)
        for m in MODELS:
            for cond_pre, (w_o, w_i) in {"none": (None, None), "reweighing": (w_outer, w_inner)}.items():
                kw_o = {} if w_o is None else {"m__sample_weight": w_o}
                kw_i = {} if w_i is None else {"m__sample_weight": w_i}
                outer = thesis_model(m, Xtr).fit(Xtr, ytr, **kw_o)
                p = outer.predict_proba(Xte)[:, 1]
                pred = (p >= 0.5).astype(int)
                inner = thesis_model(m, Xin).fit(Xin, yin, **kw_i)
                tmap = thesis_threshold_search(yval, inner.predict_proba(Xval)[:, 1], sval, grid)
                pred_post = apply_group_thresholds(p, ste, tmap)
                base_cond = "baseline" if cond_pre == "none" else "pre"
                post_cond = "post" if cond_pre == "none" else "combined"
                rows.append(
                    {"fold": fold, "model": m, "condition": base_cond, **_fold_metrics(yte, p, pred, ste)}
                )
                rows.append(
                    {
                        "fold": fold,
                        "model": m,
                        "condition": post_cond,
                        **_fold_metrics(yte, p, pred_post, ste),
                        "thresholds": {k: round(v, 3) for k, v in tmap.items()},
                    }
                )
    df = pd.DataFrame(rows)
    summary = (
        df.drop(columns=["thresholds"], errors="ignore").groupby(["model", "condition"]).agg(["mean", "std"])
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.drop(columns=["fold_mean", "fold_std"]).reset_index()

    tests = {}
    for m in MODELS:
        base = df[(df.model == m) & (df.condition == "baseline")].sort_values("fold")
        for cond in ("post", "pre", "combined"):
            other = df[(df.model == m) & (df.condition == cond)].sort_values("fold")
            tests[f"{m}:{cond}"] = {
                "SPD": wilcoxon_paired(base["SPD"].abs(), other["SPD"].abs()),
                "F1": wilcoxon_paired(base["F1"], other["F1"]),
            }
    noise_floor = float(df[df.model == "dummy_stratified"]["SPD"].mean())
    return {
        "dataset": name,
        "data_source": source,
        "n": len(y),
        "folds": int(df.fold.max()),
        "summary": summary,
        "tests": tests,
        "noise_floor_spd": noise_floor,
    }


def compare_with_thesis(results: dict[str, Any]) -> list[dict[str, Any]]:
    rep = load_thesis_reported()
    rows = []
    for ds, r in results.items():
        summ = pd.DataFrame(r["summary"]) if not isinstance(r["summary"], pd.DataFrame) else r["summary"]
        get = lambda m, c, k: float(summ[(summ.model == m) & (summ.condition == c)][f"{k}_mean"].iloc[0])
        for m, vals in rep["table_5_1"][ds].items():
            mm = m if m in MODELS else "dummy_most_frequent"
            for k, v in vals.items():
                if k not in ("AUC", "F1", "Macro_F1", "Accuracy"):
                    continue
                ours = get(mm, "baseline", k)
                rows.append(
                    {
                        "table": "5.1",
                        "dataset": ds,
                        "model": m,
                        "metric": k,
                        "thesis": v,
                        "reproduced": round(ours, 3),
                        "diff": round(ours - v, 3),
                    }
                )
        for m, conds in rep["table_6_1"][ds].items():
            for c, v in conds.items():
                ours = get(m, c, "SPD")
                rows.append(
                    {
                        "table": "6.1 (|SPD|)",
                        "dataset": ds,
                        "model": m,
                        "metric": c,
                        "thesis": v,
                        "reproduced": round(ours, 3),
                        "diff": round(ours - v, 3),
                    }
                )
        rows.append(
            {
                "table": "noise floor",
                "dataset": ds,
                "model": "dummy_stratified",
                "metric": "|SPD|",
                "thesis": rep["noise_floor"][ds],
                "reproduced": round(r["noise_floor_spd"], 3),
                "diff": round(r["noise_floor_spd"] - rep["noise_floor"][ds], 3),
            }
        )
    for row in rows:
        d = abs(row["diff"])
        row["status"] = "match" if d <= 0.005 else ("close" if d <= 0.03 else "differs")
    return rows


def run(settings: Settings) -> dict[str, Any]:
    results = {}
    for name in settings.datasets:
        with timed(f"thesis reproduction: {name}"):
            results[name] = reproduce_dataset(name, settings)
    comparison = compare_with_thesis(results)
    payload = {
        "results": results,
        "comparison": comparison,
        "note": (
            "Thesis protocol: success = positive class, aware models, thesis hyper-parameters, "
            "5x2 RepeatedStratifiedKFold (seed 42). Thesis Table 6.1 used 5x1 folds, so small "
            "differences there are expected."
        ),
    }
    synthetic = any(r["data_source"] == "synthetic" for r in results.values())
    base = settings.path("reports") / ("synthetic" if synthetic else "")
    write_json(base / "thesis_reproduction.json", payload)
    return payload
