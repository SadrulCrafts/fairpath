"""Small helpers shared by the pipeline stages."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fairpath.config import PROJECT_ROOT, Settings
from fairpath.data.loaders import Dataset

log = logging.getLogger(__name__)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    log.info("> %s", label)
    yield
    log.info("< %s (%.1fs)", label, time.perf_counter() - t0)


@dataclass
class DevTestSplit:
    dev_idx: np.ndarray
    test_idx: np.ndarray


def dev_test_split(ds: Dataset, settings: Settings) -> DevTestSplit:
    """Stratified on outcome x primary sensitive group so both appear in the test set in proportion."""
    strata = pd.Series(ds.y).astype(str) + "_" + ds.sensitive[ds.spec.primary_sensitive].astype(str)
    idx = np.arange(len(ds.y))
    dev, test = train_test_split(
        idx, test_size=settings.split["test_size"], random_state=settings.seed, stratify=strata
    )
    return DevTestSplit(np.sort(dev), np.sort(test))


def provenance(ds: Dataset) -> dict[str, Any]:
    return {
        "data_source": ds.source,
        "data_fingerprint": ds.fingerprint,
        "n": len(ds.y),
        "n_raw": ds.n_raw,
        "cleaning": ds.notes,
        "git_commit": git_sha(),
        "generated_at": now_iso(),
        "synthetic": ds.is_synthetic,
    }


def reports_dir(settings: Settings, ds: Dataset | None = None) -> Path:
    base = settings.path("reports")
    if ds is not None and ds.is_synthetic:
        base = base / "synthetic"
    return base
