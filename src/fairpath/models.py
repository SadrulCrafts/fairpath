"""Model registry, default hyper-parameters and Optuna search spaces."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fairpath.config import DatasetSpec
from fairpath.features import build_preprocessor

log = logging.getLogger(__name__)

N_JOBS = 2

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "dummy_most_frequent": "Dummy (most frequent)",
    "dummy_stratified": "Dummy (stratified)",
}

# Hyper-parameters used in the thesis (configs/config.yaml of the original project).
THESIS_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 300, "max_depth": 8},
    "hist_gradient_boosting": {"learning_rate": 0.05, "max_depth": 6, "max_iter": 300},
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0},
    "random_forest": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2},
    "hist_gradient_boosting": {"learning_rate": 0.05, "max_depth": 6, "max_iter": 300},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15},
    "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
}


def make_estimator(name: str, params: dict[str, Any] | None = None, seed: int = 42) -> BaseEstimator:
    p = dict(DEFAULT_PARAMS.get(name, {}))
    p.update(params or {})
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, **p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **p)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=seed, n_jobs=N_JOBS, verbose=-1, deterministic=True, force_row_wise=True, **p
        )
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss", **p)
    if name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")
    if name == "dummy_stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    raise KeyError(f"Unknown model '{name}'")


def make_pipeline(
    name: str, spec: DatasetSpec, features: list[str], params: dict | None = None, seed: int = 42
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(spec, features)),
            ("model", make_estimator(name, params, seed)),
        ]
    )


def fit_pipeline(
    pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None
) -> Pipeline:
    if sample_weight is None:
        return pipe.fit(X, y)
    return pipe.fit(X, y, model__sample_weight=sample_weight)


def search_space(name: str, trial: optuna.Trial) -> dict[str, Any]:
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 48),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise KeyError(name)


def tune(
    name: str,
    make: Callable[[dict[str, Any]], Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    n_trials: int,
    n_folds: int,
    seed: int,
) -> tuple[dict[str, Any], float]:
    """Optuna TPE search maximising mean ROC-AUC over stratified K folds.

    Only ever called on the development split; the held-out test set is untouched.
    """
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = search_space(name, trial)
        scores = []
        for k, (tr, va) in enumerate(folds):
            pipe = make(params)
            pipe.fit(X.iloc[tr], y[tr])
            scores.append(roc_auc_score(y[va], pipe.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )
    # always evaluate the hand-picked defaults too, so tuning can only help
    if name in DEFAULT_PARAMS:
        study.enqueue_trial({k: v for k, v in DEFAULT_PARAMS[name].items()}, skip_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    if name == "lightgbm":
        best["subsample_freq"] = 1
    log.info("Tuned %s: AUC=%.4f params=%s", name, study.best_value, best)
    return best, float(study.best_value)


def clone_pipe(p: Pipeline) -> Pipeline:
    return clone(p)
