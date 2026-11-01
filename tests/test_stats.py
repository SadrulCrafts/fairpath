import numpy as np

from fairpath.evaluation.stats import (
    corrected_ci,
    corrected_resampled_ttest,
    holm,
    rank_biserial,
    wilcoxon_paired,
)


def test_rank_biserial_extremes_and_known_value():
    assert rank_biserial([2, 3, 4], [1, 1, 1]) == 1.0
    assert rank_biserial([1, 1, 1], [2, 3, 4]) == -1.0
    # diffs 1, -2, 3 -> ranks 1, 2, 3 -> W+ = 4, W- = 2 -> r = 2/6
    assert np.isclose(rank_biserial([2, 0, 4], [1, 2, 1]), 1 / 3)


def test_wilcoxon_zero_difference():
    out = wilcoxon_paired([1, 2, 3], [1, 2, 3])
    assert out["p_value"] == 1.0 and out["effect"] == "negligible"


def test_holm_matches_definition():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_corrected_test_is_more_conservative(rng):
    a = 0.80 + rng.normal(0, 0.01, 25)
    b = a - 0.004 + rng.normal(0, 0.003, 25)
    naive = wilcoxon_paired(a, b)["p_value"]
    corr = corrected_resampled_ttest(a, b, n_train=2300, n_test=580)["p_value"]
    assert corr > naive
    lo, hi = corrected_ci(a, 2300, 580)
    assert lo < a.mean() < hi
