import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from fairpath.fairness.metrics import fairness_report, gaps, group_table, threshold_sweep


def test_gaps_match_fairlearn(rng):
    y = rng.integers(0, 2, 500)
    pred = rng.integers(0, 2, 500)
    g = rng.choice(["a", "b", "c"], 500)
    ours = gaps(y, pred, g)
    assert np.isclose(ours["spd"], demographic_parity_difference(y, pred, sensitive_features=g))
    assert np.isclose(ours["eodds"], equalized_odds_difference(y, pred, sensitive_features=g))


def test_single_class_group_does_not_crash():
    # the thesis code crashed here (confusion_matrix().ravel() on a 1x1 matrix)
    y = np.array([1, 1, 0, 1])
    pred = np.array([1, 1, 0, 1])
    g = np.array(["F", "F", "M", "M"])
    out = gaps(y, pred, g)
    assert np.isfinite(out["spd"])
    rows = group_table(y, pred, g)
    assert rows[0]["fpr"] is None  # no negatives in group F


def test_disparate_impact_and_perfect_parity():
    y = np.array([0, 1, 0, 1])
    pred = np.array([1, 0, 1, 0])
    g = np.array(["a", "a", "b", "b"])
    out = gaps(y, pred, g)
    assert out["spd"] == 0 and out["di_ratio"] == 1


def test_report_ci_brackets_point_and_flags_small_groups(rng):
    n = 400
    g = np.where(rng.random(n) < 0.06, "small", "big")
    y = rng.integers(0, 2, n)
    s = np.clip(0.5 * y + rng.normal(0.25, 0.2, n), 0, 1)
    rep = fairness_report(y, s, g, 0.5, n_boot=200, n_perm=200)
    for k, v in rep["gaps"].items():
        if v["lo"] is not None and k != "di_ratio":
            assert v["lo"] - 1e-9 <= v["value"] <= v["hi"] + 1e-9 or k == "calibration_gap"
    assert any("small" in w for w in rep["warnings"])
    assert 0 < rep["gaps"]["spd"]["perm_p"] <= 1


def test_permutation_detects_real_gap(rng):
    n = 2000
    g = rng.choice(["a", "b"], n)
    y = rng.integers(0, 2, n)
    s = np.where(g == "a", 0.8, 0.2) * 0.7 + 0.3 * rng.random(n)
    rep = fairness_report(y, s, g, 0.5, n_boot=100, n_perm=300)
    assert rep["gaps"]["spd"]["perm_p"] < 0.01


def test_threshold_sweep_counts_add_up(rng):
    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    g = rng.choice(["x", "y"], 300)
    sw = threshold_sweep(y, s, g)
    for grp in sw["groups"].values():
        tot = np.array(grp["tp"]) + grp["fp"] + np.array(grp["fn"]) + grp["tn"]
        assert (tot == grp["n"]).all()
