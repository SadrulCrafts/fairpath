"""Load a dataset, apply the documented cleaning steps and return model-ready pieces.

The cleaning rules come straight from the thesis (chapter 4) and live in the dataset
YAML, so there is exactly one place where they are defined:

* dropout -- drop ``Enrolled`` rows (outcome unknown) and the 12 end-of-semester
  ``Curricular units`` columns (they leak the outcome);
* student-mat -- drop the in-year grades ``G1`` and ``G2``; the target is ``G3 >= 10``.

Upgraded pipeline convention: the positive class is **at risk** (dropout / fail),
because that is the student an advisor needs to find.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fairpath.config import DatasetSpec, Settings, load_dataset_spec, load_settings

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    """A cleaned dataset in the feature space described by its YAML spec."""

    spec: DatasetSpec
    X: pd.DataFrame  # all spec features (including sensitive ones)
    y: np.ndarray  # 1 = at risk, 0 = success
    sensitive: pd.DataFrame  # one column per audit axis (derived ones included)
    source: str  # "real" or "synthetic"
    fingerprint: str
    n_raw: int
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    def features_for_model(self, blind: bool) -> pd.DataFrame:
        return self.X[self.spec.model_features(blind)]


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def read_raw(spec: DatasetSpec, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=spec.sep, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_path(spec: DatasetSpec, settings: Settings, source: str = "auto") -> tuple[Path, str]:
    real = settings.path("data_raw") / spec.file
    synth = settings.path("data_synthetic") / spec.file
    if source in ("auto", "real") and real.exists():
        return real, "real"
    if source == "real":
        raise FileNotFoundError(
            f"Real data for '{spec.name}' not found at {real}. Run `make data` "
            "(or `python scripts/download_data.py`) first."
        )
    if source == "synthetic" or settings.allow_synthetic_fallback:
        if not synth.exists():
            raise FileNotFoundError(f"Synthetic data not found at {synth}. Run `make synthetic`.")
        if source == "auto":
            log.warning("Real %s data not found -- using the SYNTHETIC copy at %s", spec.name, synth)
        return synth, "synthetic"
    raise FileNotFoundError(f"No data for '{spec.name}' (looked for {real}).")


def clean(spec: DatasetSpec, raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Apply the dataset's cleaning rules. Returns (features, at_risk_label, notes)."""
    notes: list[str] = []
    df = raw.copy()
    t = spec.target

    if t.drop_values:
        keep = ~df[t.source].isin(t.drop_values)
        notes.append(f"Removed {int((~keep).sum())} rows with {t.source} in {t.drop_values}.")
        df = df[keep]

    if t.threshold is not None:
        y = (pd.to_numeric(df[t.source]) < t.threshold).astype(int).to_numpy()
    else:
        unknown = set(df[t.source].unique()) - set(t.at_risk_values) - set(t.success_values)
        if unknown:
            raise ValueError(f"Unexpected target values in {spec.name}: {sorted(unknown)}")
        y = df[t.source].isin(t.at_risk_values).astype(int).to_numpy()

    dropped = [c for c in df.columns if any(c.startswith(p) for p in spec.drop_prefixes)]
    dropped += [c for c in spec.drop_columns if c in df.columns]
    if dropped:
        notes.append(f"Dropped {len(dropped)} leakage columns: {', '.join(dropped)}.")

    missing = [f.source for f in spec.features if f.source not in df.columns]
    if missing:
        raise ValueError(f"{spec.name}: missing expected columns {missing}")

    X = pd.DataFrame(index=df.index)
    for f in spec.features:
        col = df[f.source]
        if f.type == "categorical":
            X[f.name] = col.astype(str).str.strip().astype(object)
        else:
            X[f.name] = pd.to_numeric(col, errors="coerce").astype(float)
    return X.reset_index(drop=True), y, notes


def derive_sensitive(spec: DatasetSpec, X: pd.DataFrame) -> pd.DataFrame:
    """Build one column per audit axis, including derived axes (age bands, SES)."""
    out = pd.DataFrame(index=X.index)
    for name, s in spec.sensitive.items():
        if s.derived_from is None:
            col = X[name]
            out[name] = col.astype(int) if col.dtype != object else col
        elif s.bins is not None:
            src = X[s.derived_from]  # type: ignore[index]
            out[name] = pd.cut(src, bins=s.bins, labels=False, right=True).astype(int)
        elif s.rule == "max_le":
            cols = s.derived_from if isinstance(s.derived_from, list) else [s.derived_from]
            out[name] = (X[cols].max(axis=1) > s.cutoff).astype(int)
        else:
            raise ValueError(f"Cannot derive sensitive attribute {name}")
    return out


def load_dataset(name: str, settings: Settings | None = None, source: str = "auto") -> Dataset:
    settings = settings or load_settings()
    spec = load_dataset_spec(name)
    path, used = resolve_path(spec, settings, source)
    raw = read_raw(spec, path)
    X, y, notes = clean(spec, raw)
    sens = derive_sensitive(spec, X)
    ds = Dataset(
        spec=spec,
        X=X,
        y=y,
        sensitive=sens,
        source=used,
        fingerprint=_fingerprint(path),
        n_raw=len(raw),
        notes=notes,
    )
    log.info("Loaded %s (%s): n=%d, at-risk rate=%.3f", name, used, len(y), y.mean())
    return ds


def group_label(spec: DatasetSpec, axis: str, value) -> str:
    groups = spec.sensitive[axis].groups
    for k, v in groups.items():
        if str(k) == str(value):
            return v
    return str(value)


def intersection_series(ds: Dataset, axes: list[str]) -> pd.Series:
    parts = [ds.sensitive[a].map(lambda v, a=a: group_label(ds.spec, a, v)) for a in axes]
    s = parts[0].astype(str)
    for p in parts[1:]:
        s = s + " x " + p.astype(str)
    return s
