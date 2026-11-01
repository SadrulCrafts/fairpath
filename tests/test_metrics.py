import numpy as np

from fairpath.evaluation.calibration import ProbabilityCalibrator
from fairpath.evaluation.metrics import (
    best_f1_threshold,
    classification_metrics,
    expected_calibration_error,
    metrics_with_ci,
)


def test_ece_near_zero_for_calibrated_scores(rng):
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p * 0.5, 0, 1)) > 0.2


def test_calibrator_reduces_ece(rng):
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    distorted = p**3
    for method in ("sigmoid", "isotonic"):
        cal = ProbabilityCalibrator(method).fit(distorted, y)
        assert expected_calibration_error(y, cal.transform(distorted)) < expected_calibration_error(
            y, distorted
        )


def test_metrics_and_ci(rng):
    y = rng.integers(0, 2, 400)
    p = np.clip(0.3 * y + rng.random(400) * 0.7, 0, 1)
    m = classification_metrics(y, p, 0.5)
    assert 0.5 < m["roc_auc"] <= 1
    ci = metrics_with_ci(y, p, 0.5, n_resamples=200)
    assert ci["roc_auc"]["lo"] <= m["roc_auc"] <= ci["roc_auc"]["hi"]
    t = best_f1_threshold(y, p)
    assert 0.1 <= t <= 0.8
