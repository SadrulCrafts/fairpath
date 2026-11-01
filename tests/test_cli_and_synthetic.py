import json

import numpy as np
import pandas as pd
import pytest

from fairpath.cli import main
from fairpath.config import load_dataset_spec, load_settings
from fairpath.data.loaders import read_raw
from fairpath.data.synthetic import make_synthetic


def test_synthetic_keeps_schema_and_class_balance():
    s = load_settings()
    spec = load_dataset_spec("student_mat")
    base = read_raw(spec, s.path("data_synthetic") / spec.file)  # works without the real data
    out = make_synthetic(base, spec, n=300, seed=1)
    assert list(out.columns) == list(base.columns)
    assert len(out) == 300
    rate = lambda d: (d["G3"] < 10).mean()
    assert abs(rate(out) - rate(base)) < 0.08
    assert set(out["sex"]) <= set(base["sex"])
    merged = out.merge(base.drop_duplicates(), how="inner")
    assert len(merged) == 0  # no row copied verbatim


def test_preflight_cli(tmp_path, capsys):
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({"y": rng.integers(0, 2, n), "g": rng.choice(["a", "b"], n), "p": rng.random(n)})
    path = tmp_path / "pf.csv"
    df.to_csv(path, index=False)
    main(["preflight", str(path), "--label", "y", "--group", "g"])
    model_free = json.loads(capsys.readouterr().out)
    assert "minimum_detectable_gap" in model_free
    main(["preflight", str(path), "--label", "y", "--group", "g", "--score", "p"])
    full = json.loads(capsys.readouterr().out)
    assert full["v2"]["verdict_code"] in {"mitigate", "mitigate_coarse", "monitor", "insufficient"}
    assert len(full["v1"]["checks"]) == 3


@pytest.mark.slow
def test_thesis_reproduction_harness_runs(tmp_path):
    from fairpath.pipeline import thesis_repro

    s = load_settings().model_copy(deep=True)
    s.paths = {**s.paths, "data_raw": str(tmp_path / "none")}  # force the synthetic copy
    res = thesis_repro.reproduce_dataset("student_mat", s)
    assert res["data_source"] == "synthetic"
    summ = pd.DataFrame(res["summary"])
    assert {"baseline", "post", "pre", "combined"} <= set(summ.condition)
    rows = thesis_repro.compare_with_thesis({"student_mat": res})
    assert all(r["status"] in {"match", "close", "differs"} for r in rows)


def test_fairpath_root_env_overrides_install_location(tmp_path):
    """An installed (non-editable) package must find configs/ through FAIRPATH_ROOT, as in the Docker image."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "FAIRPATH_ROOT": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "-c", "from fairpath.config import PROJECT_ROOT; print(PROJECT_ROOT)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == str(tmp_path)
