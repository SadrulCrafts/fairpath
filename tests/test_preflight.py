import numpy as np

from fairpath.fairness.preflight import minimum_detectable_gap, preflight_v1, preflight_v2


def _data(rng, n, gap):
    g = rng.choice(["a", "b"], n)
    y = rng.integers(0, 2, n)
    s = np.clip(0.35 * y + np.where(g == "a", 0.30 + gap, 0.30) + rng.normal(0, 0.12, n), 0, 1)
    return y, s, g


def test_mde_matches_hand_calculation():
    # z(0.975)+z(0.8) = 2.8016; sqrt(.25*(1/100+1/100)) = 0.0707
    assert abs(minimum_detectable_gap(100, 100, 0.5) - 0.1981) < 1e-3


def test_large_biased_dataset_passes_both(rng):
    y, s, g = _data(rng, 4000, 0.25)
    assert preflight_v1(y, s, g, n_boot=200)["passed"]
    assert preflight_v2(y, s, g, n_boot=200, n_perm=200)["verdict_code"] == "mitigate"


def test_small_dataset_is_insufficient(rng):
    """A modest gap on 300 students: neither the gap nor a change in it can be pinned down."""
    y, s, g = _data(rng, 300, 0.1)
    v2 = preflight_v2(y, s, g, n_boot=200, n_perm=200)
    assert v2["verdict_code"] == "insufficient"
    assert not preflight_v1(y, s, g, n_boot=200)["passed"]


def test_v2_distinguishes_fair_from_small(rng):
    """The case v1 can't handle: plenty of data, no gap -> 'monitor', not 'insufficient'."""
    y, s, g = _data(rng, 6000, 0.0)
    v2 = preflight_v2(y, s, g, n_boot=200, n_perm=300)
    assert v2["verdict_code"] == "monitor"
    assert not preflight_v1(y, s, g, n_boot=200)["passed"]  # v1 just says "do not mitigate"


def test_clear_gap_on_a_small_group_is_coarse_not_insufficient(rng):
    """A large, unmistakable gap should not be called 'insufficient data' just because one group is small."""
    n_small, n_big = 120, 2400
    g = np.array(["a"] * n_small + ["b"] * n_big)
    y = rng.integers(0, 2, n_small + n_big)
    s = np.where(g == "a", rng.uniform(0.5, 1.0, g.size), rng.uniform(0.0, 0.6, g.size))
    v2 = preflight_v2(y, s, g, n_boot=200, n_perm=200)
    assert not v2["checks"][0]["passed"]
    assert v2["verdict_code"] == "mitigate_coarse"
