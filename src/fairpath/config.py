"""Typed access to the YAML configuration files.

Two kinds of config live in ``configs/``:

* ``default.yaml`` -- experiment settings (seeds, CV design, tuning budget, ...)
* ``datasets/<name>.yaml`` -- everything we know about a dataset: where it comes from,
  how the target is built, which columns are sensitive, and how each feature should be
  shown to a human in the dashboard.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# FAIRPATH_ROOT lets an installed (non-editable) package find configs/, e.g. in the Docker image.
PROJECT_ROOT = Path(os.environ.get("FAIRPATH_ROOT") or Path(__file__).resolve().parents[2])
CONFIG_DIR = PROJECT_ROOT / "configs"

FeatureType = Literal["numeric", "categorical", "binary"]


class Actionable(BaseModel):
    """How (and whether) an institution could realistically change a feature."""

    direction: Literal["increase", "decrease"] | None = None
    to: Any | None = None
    action: str


class FeatureSpec(BaseModel):
    name: str
    source: str
    type: FeatureType
    group: str = "Other"
    label: str
    help: str | None = None
    labels: dict[Any, str] | None = None
    labels_ref: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    protected: bool = False
    actionable: Actionable | None = None

    @property
    def is_actionable(self) -> bool:
        return self.actionable is not None and not self.protected


class SensitiveSpec(BaseModel):
    label: str
    groups: dict[Any, str]
    primary: bool = False
    derived_from: str | list[str] | None = None
    bins: list[float] | None = None
    rule: str | None = None
    cutoff: float | None = None


class TargetSpec(BaseModel):
    source: str
    threshold: float | None = None
    drop_values: list[str] = Field(default_factory=list)
    at_risk_values: list[str] = Field(default_factory=list)
    success_values: list[str] = Field(default_factory=list)
    at_risk_label: str = "At risk"
    success_label: str = "Success"


class DatasetSpec(BaseModel):
    name: str
    title: str
    short_title: str
    description: str
    file: str
    sep: str = ","
    source_url: str
    download_url: str
    zip_member: str | None = None
    licence: str
    citation: str
    target: TargetSpec
    drop_prefixes: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    sensitive: dict[str, SensitiveSpec]
    intersections: list[list[str]] = Field(default_factory=list)
    blind_drop: list[str] = Field(default_factory=list)
    features: list[FeatureSpec]
    label_sets: dict[str, dict[Any, str]] = Field(default_factory=dict)

    @property
    def primary_sensitive(self) -> str:
        for name, spec in self.sensitive.items():
            if spec.primary:
                return name
        return next(iter(self.sensitive))

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def feature(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)

    def labels_for(self, feature: FeatureSpec) -> dict[Any, str]:
        if feature.labels is not None:
            return feature.labels
        if feature.labels_ref:
            return self.label_sets.get(feature.labels_ref, {})
        return {}

    def model_features(self, blind: bool) -> list[str]:
        drop = set(self.blind_drop) if blind else set()
        return [f for f in self.feature_names if f not in drop]


class Settings(BaseModel):
    seed: int = 42
    paths: dict[str, str]
    datasets: list[str]
    allow_synthetic_fallback: bool = True
    split: dict[str, Any]
    cv: dict[str, Any]
    tuning: dict[str, Any]
    models: list[str]
    bootstrap: dict[str, Any]
    permutation_test: dict[str, Any]
    calibration: dict[str, Any]
    conformal: dict[str, Any]
    decision: dict[str, Any]
    fairness: dict[str, Any]
    deployment: dict[str, Any]
    preflight: dict[str, Any]

    def path(self, key: str) -> Path:
        p = Path(self.paths[key])
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=8)
def load_settings(path: str | Path | None = None) -> Settings:
    p = Path(path) if path else CONFIG_DIR / "default.yaml"
    return Settings(**yaml.safe_load(p.read_text(encoding="utf-8")))


@lru_cache(maxsize=16)
def load_dataset_spec(name: str) -> DatasetSpec:
    p = CONFIG_DIR / "datasets" / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No dataset spec called '{name}' in {p.parent}")
    return DatasetSpec(**yaml.safe_load(p.read_text(encoding="utf-8")))


def load_thesis_reported() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "thesis_reported.yaml").read_text(encoding="utf-8"))
