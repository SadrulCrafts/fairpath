import numpy as np
import pandas as pd

from fairpath.mitigation import apply_group_thresholds, reweighing_weights, thesis_threshold_search


def test_reweighing_makes_group_and_label_independent(rng):
    s = rng.choice(["F", "M"], 1000, p=[0.7, 0.3])
    y = (rng.random(1000) < np.where(s == "F", 0.3, 0.6)).astype(int)
    w = reweighing_weights(y, s)
    assert np.isclose(w.mean(), 1.0)
    df = pd.DataFrame({"s": s, "y": y, "w": w})
    rates = df.groupby("s").apply(lambda d: np.average(d.y, weights=d.w), include_groups=False)
    assert np.isclose(rates["F"], rates["M"], atol=1e-9)


def test_group_thresholds_applied_per_group():
    pred = apply_group_thresholds(np.array([0.3, 0.3, 0.6]), np.array(["a", "b", "b"]), {"a": 0.2, "b": 0.5})
    assert pred.tolist() == [1, 0, 1]


def test_threshold_search_handles_any_number_of_groups(rng):
    grid = np.linspace(0.1, 0.9, 5)
    for k in (2, 3, 4):
        g = rng.choice([f"g{i}" for i in range(k)], 400)
        y = rng.integers(0, 2, 400)
        s = rng.random(400)
        tmap = thesis_threshold_search(y, s, g, grid)
        assert set(tmap) == set(np.unique(g))
        assert all(np.isclose(t, grid).any() for t in tmap.values())


def test_threshold_search_reduces_fpr_gap(rng):
    n = 2000
    g = rng.choice(["a", "b"], n)
    y = rng.integers(0, 2, n)
    s = np.clip(0.4 * y + np.where(g == "a", 0.35, 0.1) + rng.normal(0, 0.1, n), 0, 1)
    tmap = thesis_threshold_search(y, s, g, np.linspace(0.1, 0.9, 17))
    from fairpath.fairness.metrics import gaps

    before = gaps(y, (s >= 0.5).astype(int), g)["fpr_gap"]
    after = gaps(y, apply_group_thresholds(s, g, tmap), g)["fpr_gap"]
    assert after < before
