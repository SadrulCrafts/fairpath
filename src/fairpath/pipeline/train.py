"""Train the model that the dashboard serves, and evaluate it once on the held-out test set.

Deployed configuration (decided up front, see docs/MODEL_CARD.md):

* the best model family from the benchmark, with its tuned hyper-parameters;
* **blind**: the primary sensitive attribute (gender / sex) is not an input;
* **Reweighing** on that attribute during training -- the institution never applies a
  different rule to different groups when it scores a student;
* probabilities calibrated on out-of-fold predictions (isotonic or Platt by data size);
* an "at-risk" flag threshold chosen on out-of-fold predictions (max F1);
* split-conformal prediction sets for an honest confidence indicator.

For comparison, an "aware, unmitigated" model goes through exactly the same steps.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fairpath import __version__
from fairpath.config import Settings
from fairpath.data.loaders import Dataset, group_label, intersection_series, load_dataset
from fairpath.data.schema import build_schema
from fairpath.evaluation.calibration import ProbabilityCalibrator, choose_method
from fairpath.evaluation.metrics import (
    best_f1_threshold,
    classification_metrics,
    metrics_with_ci,
    reliability_table,
)
from fairpath.explain.conformal import coverage, fit_conformal
from fairpath.explain.shap_explain import PipelineExplainer
from fairpath.fairness.metrics import fairness_report, threshold_sweep
from fairpath.inference import ModelBundle, Predictor
from fairpath.mitigation import reweighing_weights
from fairpath.models import MODEL_LABELS, fit_pipeline, make_pipeline
from fairpath.pipeline.common import (
    dev_test_split,
    provenance,
    read_json,
    reports_dir,
    timed,
    write_json,
)

log = logging.getLogger(__name__)

LEAKAGE_SUSPECTS = {"dropout": ["debtor", "tuition_up_to_date"]}


def _oof_raw(
    ds: Dataset,
    model: str,
    params: dict,
    feats: list[str],
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    pipe = make_pipeline(model, ds.spec, feats, params, seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    kw = {} if w is None else {"params": {"model__sample_weight": w}}
    return cross_val_predict(pipe, X[feats], y, cv=cv, method="predict_proba", **kw)[:, 1]


def fit_config(
    ds: Dataset,
    settings: Settings,
    model: str,
    params: dict,
    blind: bool,
    reweigh: bool,
    Xdev: pd.DataFrame,
    ydev: np.ndarray,
    sdev: np.ndarray,
) -> dict[str, Any]:
    feats = ds.spec.model_features(blind)
    w = reweighing_weights(ydev, sdev) if reweigh else None
    raw_oof = _oof_raw(ds, model, params, feats, Xdev, ydev, w, settings.seed)
    method = choose_method(len(ydev), settings.calibration)
    cal = ProbabilityCalibrator(method).fit(raw_oof, ydev)
    cal_oof = cal.transform(raw_oof)
    thr = best_f1_threshold(ydev, cal_oof)
    qhat = fit_conformal(ydev, cal_oof, settings.conformal["alpha"])
    pipe = fit_pipeline(make_pipeline(model, ds.spec, feats, params, settings.seed), Xdev[feats], ydev, w)
    return {
        "features": feats,
        "pipeline": pipe,
        "calibrator": cal,
        "threshold": thr,
        "qhat": qhat,
        "calibration_method": method,
        "raw_oof": raw_oof,
        "cal_oof": cal_oof,
        "weights": w,
    }


def predict(cfg: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = cfg["pipeline"].predict_proba(X[cfg["features"]])[:, 1]
    return raw, cfg["calibrator"].transform(raw)


def axis_list(ds: Dataset) -> list[str]:
    return list(ds.spec.sensitive) + ["+".join(i) for i in ds.spec.intersections]


def groups_for(ds: Dataset, axis: str, idx: np.ndarray) -> np.ndarray:
    s = intersection_series(ds, axis.split("+")) if "+" in axis else ds.sensitive[axis].astype(str)
    return s.iloc[idx].astype(str).to_numpy()


def labels_for(ds: Dataset, axis: str) -> dict[str, str]:
    if "+" in axis:
        return {}
    return {str(k): v for k, v in ds.spec.sensitive[axis].groups.items()}


def run(ds: Dataset, settings: Settings) -> dict[str, Any]:
    out_dir = reports_dir(settings, ds) / ds.name
    bench = read_json(out_dir / "benchmark.json")
    tuned = read_json(out_dir / "tuned_params.json")
    model = bench["best_model"]
    params = tuned[model]["params"]
    split = dev_test_split(ds, settings)
    Xdev, Xte = ds.X.iloc[split.dev_idx].reset_index(drop=True), ds.X.iloc[split.test_idx].reset_index(
        drop=True
    )
    ydev, yte = ds.y[split.dev_idx], ds.y[split.test_idx]
    prim = ds.spec.primary_sensitive
    sdev = groups_for(ds, prim, split.dev_idx)

    with timed(f"fit deployed + baseline configs ({MODEL_LABELS[model]}) on {ds.name}"):
        dep = fit_config(
            ds,
            settings,
            model,
            params,
            blind=settings.deployment["blind"],
            reweigh=settings.deployment["mitigation"] == "reweighing",
            Xdev=Xdev,
            ydev=ydev,
            sdev=sdev,
        )
        base = fit_config(
            ds, settings, model, params, blind=False, reweigh=False, Xdev=Xdev, ydev=ydev, sdev=sdev
        )

    raw_dep, p_dep = predict(dep, Xte)
    raw_base, p_base = predict(base, Xte)
    B, ci = settings.bootstrap["n_resamples"], settings.bootstrap["ci"]
    with timed("test-set metrics with bootstrap CIs"):
        test_metrics = {
            "deployed": metrics_with_ci(yte, p_dep, dep["threshold"], B, ci, settings.seed),
            "baseline": metrics_with_ci(yte, p_base, base["threshold"], B, ci, settings.seed),
            "deployed_uncalibrated": classification_metrics(yte, raw_dep, dep["threshold"]),
        }
    calibration = {
        "method": dep["calibration_method"],
        "oof_ece_before": classification_metrics(ydev, dep["raw_oof"])["ece"],
        "oof_ece_after": classification_metrics(ydev, dep["cal_oof"])["ece"],
        "test_ece_before": classification_metrics(yte, raw_dep)["ece"],
        "test_ece_after": classification_metrics(yte, p_dep)["ece"],
        "test_brier_before": classification_metrics(yte, raw_dep)["brier"],
        "test_brier_after": classification_metrics(yte, p_dep)["brier"],
        "reliability_before": reliability_table(yte, raw_dep),
        "reliability_after": reliability_table(yte, p_dep),
    }

    fairness: dict[str, Any] = {}
    sweeps: dict[str, Any] = {}
    with timed("test-set fairness audit (bootstrap + permutation)"):
        for axis in axis_list(ds):
            g = groups_for(ds, axis, split.test_idx)
            labs = labels_for(ds, axis)
            kw = dict(
                labels=labs,
                n_boot=B,
                n_perm=settings.permutation_test["n_permutations"],
                ci=ci,
                seed=settings.seed,
                small_n=settings.fairness["small_group_warning"],
                caution_n=settings.fairness["caution_group_size"],
            )
            dep_rep = fairness_report(yte, p_dep, g, dep["threshold"], **kw)
            base_rep = fairness_report(yte, p_base, g, base["threshold"], **kw)
            for row in dep_rep["groups"]:  # per-group calibration of the deployed model
                m = g == row["group"]
                row["ece"] = classification_metrics(yte[m], p_dep[m])["ece"] if m.sum() > 5 else None
                row["coverage"] = coverage(yte[m], p_dep[m], dep["qhat"])["coverage"]
            fairness[axis] = {
                "label": " x ".join(ds.spec.sensitive[a].label for a in axis.split("+")),
                "deployed": dep_rep,
                "baseline": base_rep,
            }
            sweeps[axis] = {
                "deployed": threshold_sweep(yte, p_dep, g),
                "baseline": threshold_sweep(yte, p_base, g),
                "labels": labs,
            }

    conf = {
        "alpha": settings.conformal["alpha"],
        "qhat": dep["qhat"],
        "test": coverage(yte, p_dep, dep["qhat"]),
    }

    # explanations ------------------------------------------------------------------
    synth = load_dataset(ds.name, settings, source="synthetic")
    background = synth.X.sample(min(300, len(synth.X)), random_state=settings.seed).reset_index(drop=True)
    with timed("SHAP global importance"):
        expl = PipelineExplainer(dep["pipeline"], background[dep["features"]])
        gi = expl.global_importance(Xte)
        dep_samples = synth.X.sample(min(400, len(synth.X)), random_state=1).reset_index(drop=True)
        sv = expl.explain(dep_samples)
        top = gi["feature"].head(6).tolist()
        dependence = {
            f: [{"x": _j(dep_samples.at[i, f]), "shap": float(sv.at[i, f])} for i in range(len(sv))]
            for f in top
        }
        expl_aware = PipelineExplainer(base["pipeline"], background[base["features"]])
        gi_aware = expl_aware.global_importance(Xte).reset_index(drop=True)
    with timed("permutation importance (aware model, held-out test)"):
        pi = permutation_importance(
            base["pipeline"],
            Xte[base["features"]],
            yte,
            scoring="roc_auc",
            n_repeats=20,
            random_state=settings.seed,
            n_jobs=1,
        )
        perm = (
            pd.DataFrame({"feature": base["features"], "mean": pi.importances_mean, "sd": pi.importances_std})
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )
    sens_rank = {
        "feature": prim,
        "shap_rank": int(gi_aware.index[gi_aware.feature == prim][0]) + 1,
        "perm_rank": int(perm.index[perm.feature == prim][0]) + 1,
        "perm_importance": float(perm.loc[perm.feature == prim, "mean"].iloc[0]),
        "perm_sd": float(perm.loc[perm.feature == prim, "sd"].iloc[0]),
        "n_features": len(base["features"]),
    }

    # robustness checks -------------------------------------------------------------
    leakage = None
    suspects = [f for f in LEAKAGE_SUSPECTS.get(ds.name, []) if f in dep["features"]]
    if suspects:
        from sklearn.metrics import roc_auc_score

        feats_wo = [f for f in dep["features"] if f not in suspects]
        oof_wo = _oof_raw(ds, model, params, feats_wo, Xdev, ydev, dep["weights"], settings.seed)
        leakage = {
            "dropped": suspects,
            "auc_with": float(roc_auc_score(ydev, dep["raw_oof"])),
            "auc_without": float(roc_auc_score(ydev, oof_wo)),
        }

    programme = programme_audit(ds, dep, Xdev, ydev, split.dev_idx) if ds.name == "dropout" else None

    # bundle --------------------------------------------------------------------------
    schema = build_schema(ds.spec, ds.X, blind=settings.deployment["blind"])
    prim_rep = fairness[prim]["deployed"]
    note = fairness_note(ds, prim, prim_rep, fairness[prim]["baseline"])
    bundle = ModelBundle(
        dataset=ds.name,
        version=__version__,
        model_name=model,
        params=params,
        features=dep["features"],
        pipeline=dep["pipeline"],
        calibrator=dep["calibrator"],
        threshold=dep["threshold"],
        bands=settings.decision["risk_bands"],
        qhat=dep["qhat"],
        alpha=settings.conformal["alpha"],
        schema=schema,
        background=background,
        metadata={
            **provenance(ds),
            "model_label": MODEL_LABELS[model],
            "mitigation": "reweighing",
            "blind": settings.deployment["blind"],
            "fairness_note": note,
            "test_auc": test_metrics["deployed"]["roc_auc"]["value"],
        },
    )
    art_dir = settings.path("artifacts") / ("synthetic" if ds.is_synthetic else "")
    path = art_dir / f"{ds.name}_model.joblib"
    digest = bundle.save(path)
    examples = pick_examples(Predictor(bundle), synth.X)

    payload = {
        "dataset": ds.name,
        "provenance": provenance(ds),
        "model": model,
        "model_label": MODEL_LABELS[model],
        "params": params,
        "n_dev": len(ydev),
        "n_test": len(yte),
        "deployed": {
            "blind": settings.deployment["blind"],
            "mitigation": "reweighing",
            "features": dep["features"],
            "threshold": dep["threshold"],
            "bands": settings.decision["risk_bands"],
        },
        "baseline": {"features": base["features"], "threshold": base["threshold"]},
        "test_metrics": test_metrics,
        "calibration": calibration,
        "conformal": conf,
        "fairness": fairness,
        "threshold_sweeps": sweeps,
        "global_importance": gi.to_dict(orient="records"),
        "dependence": dependence,
        "aware_importance": {
            "shap": gi_aware.to_dict(orient="records"),
            "permutation": perm.to_dict(orient="records"),
        },
        "sensitive_rank": sens_rank,
        "leakage_check": leakage,
        "programme_audit": programme,
        "examples": examples,
        "artifact": {"path": str(path.name), "sha256": digest, "version": __version__},
    }
    write_json(out_dir / "final_model.json", payload)
    write_json(out_dir / "schema.json", schema)
    write_json(out_dir / "examples.json", examples)
    update_registry(settings, ds, path, digest, payload)
    return payload


def fairness_note(ds: Dataset, axis: str, dep: dict, base: dict) -> dict[str, Any]:
    g = {r["label"]: r for r in dep["groups"]}
    names = list(g)
    spd, eo = dep["gaps"]["spd"], dep["gaps"]["eodds"]
    text = (
        f"This model does not use {ds.spec.sensitive[axis].label.lower()} as an input and was trained with "
        f"Reweighing. On {dep['n']} held-out students it flagged "
        + " and ".join(f"{g[n]['flag_rate']:.0%} of {n.lower()} students" for n in names)
        + f" (gap {spd['value']:.2f}, 95% CI {spd['lo']:.2f} to {spd['hi']:.2f}). "
        + "Of the students who really went on to struggle, it caught "
        + " and ".join(f"{(g[n]['tpr'] or 0):.0%} ({n.lower()})" for n in names)
        + ". "
        + "Part of the flag-rate gap reflects real differences in outcomes between groups in this data, "
        "so a perfectly equal flag rate is not the goal; equal chances of being helped are."
    )
    return {
        "axis": axis,
        "axis_label": ds.spec.sensitive[axis].label,
        "text": text,
        "groups": dep["groups"],
        "spd": spd,
        "eodds": eo,
        "baseline_spd": base["gaps"]["spd"],
        "baseline_eodds": base["gaps"]["eodds"],
    }


def programme_audit(ds: Dataset, dep: dict, Xdev: pd.DataFrame, ydev: np.ndarray, dev_idx: np.ndarray):
    """Gender gaps *within* each programme (thesis section 4.3.4 recommends this monitoring)."""
    g = ds.sensitive["gender"].iloc[dev_idx].to_numpy()
    flags = (dep["cal_oof"] >= dep["threshold"]).astype(int)
    rows = []
    for code, block in Xdev.groupby("course"):
        idx = block.index.to_numpy()
        f_idx, m_idx = idx[g[idx] == 0], idx[g[idx] == 1]
        if len(f_idx) < 20 or len(m_idx) < 20:
            continue
        from fairpath.data.schema import option_label

        rows.append(
            {
                "course": code,
                "label": option_label(ds.spec, "course", code),
                "n_female": len(f_idx),
                "n_male": len(m_idx),
                "risk_rate_female": float(ydev[f_idx].mean()),
                "risk_rate_male": float(ydev[m_idx].mean()),
                "flag_rate_female": float(flags[f_idx].mean()),
                "flag_rate_male": float(flags[m_idx].mean()),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df["outcome_gap"] = df.risk_rate_male - df.risk_rate_female
    df["flag_gap"] = df.flag_rate_male - df.flag_rate_female
    w = df.n_female + df.n_male
    return {
        "rows": df.sort_values("flag_gap", ascending=False).to_dict(orient="records"),
        "weighted_flag_gap": float((df.flag_gap * w).sum() / w.sum()),
        "weighted_outcome_gap": float((df.outcome_gap * w).sum() / w.sum()),
        "note": "Out-of-fold predictions on the development split; programmes with at least 20 students "
        "of each gender.",
    }


def pick_examples(pred: Predictor, synth_X: pd.DataFrame) -> list[dict[str, Any]]:
    """Three illustrative students drawn from the SYNTHETIC data (never real records)."""
    X = pred.prepare(synth_X, fill_defaults=False)
    p = pred.predict_risk(X)
    targets = [("Lower-risk example", 0.08), ("Borderline example", 0.42), ("Higher-risk example", 0.80)]
    out = []
    used: set[int] = set()
    for name, t in targets:
        order = np.argsort(np.abs(p - t))
        i = int(next(j for j in order if j not in used))
        used.add(i)
        rec = {k: _j(v) for k, v in synth_X.iloc[i].to_dict().items()}
        out.append({"name": name, "risk": float(p[i]), "record": rec, "synthetic": True})
    return out


def update_registry(settings: Settings, ds: Dataset, path, digest: str, payload: dict) -> None:
    reg_path = path.parent / "registry.json"
    reg = read_json(reg_path) if reg_path.exists() else {"models": {}}
    reg["models"][ds.name] = {
        "file": path.name,
        "sha256": digest,
        "version": __version__,
        "model": payload["model"],
        "trained_at": payload["provenance"]["generated_at"],
        "git_commit": payload["provenance"]["git_commit"],
        "data_source": ds.source,
        "data_fingerprint": ds.fingerprint,
        "test_auc": payload["test_metrics"]["deployed"]["roc_auc"]["value"],
        "threshold": payload["deployed"]["threshold"],
    }
    write_json(reg_path, reg)


def _j(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


__all__ = ["group_label", "run"]
