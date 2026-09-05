import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features.feature_builder import URLFeatureBuilder
from model.finalize import expected_calibration_error, select_threshold


# ------------------------------------------------------------ threshold rule

def test_select_threshold_perfect_separation():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.01, 0.02, 0.03, 0.90, 0.95, 0.99]
    t = select_threshold(y, p, target_recall=0.97, min_precision=0.90)
    assert t["fallback"] is False
    assert t["threshold"] == pytest.approx(0.90)
    assert t["recall"] == pytest.approx(1.0)
    assert t["precision"] == pytest.approx(1.0)


def test_select_threshold_fallback_when_policy_unreachable():
    y = [1, 1, 0, 0]
    p = [0.40, 0.60, 0.45, 0.55]
    t = select_threshold(y, p, target_recall=0.97, min_precision=0.90)
    assert t["fallback"] is True
    assert t["fallback_reason"]
    # F2-optimal: catching both positives beats precision here
    assert t["threshold"] == pytest.approx(0.40)
    assert t["recall"] == pytest.approx(1.0)
    assert t["precision"] == pytest.approx(0.5)


def test_select_threshold_constant_probabilities_do_not_crash():
    t = select_threshold([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5],
                         target_recall=0.97, min_precision=0.90)
    assert 0.0 <= t["threshold"] <= 1.0
    assert t["fallback"] is True


# ------------------------------------------------------------ calibration error

def test_ece_zero_when_confident_and_correct():
    assert expected_calibration_error([1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_ece_one_when_confident_and_wrong():
    assert expected_calibration_error([0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_ece_zero_for_uninformative_but_fair():
    assert expected_calibration_error([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_ece_partial_miscalibration_value():
    # 3 of 4 confident-correct, 1 confident-wrong: bin confidence 1.0,
    # accuracy 0.75 -> ECE 0.25
    assert expected_calibration_error([1, 1, 1, 0], [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.25)


# ------------------------------------------------------------ bundle round-trip

def _tiny_corpus():
    mal = [f"http://192.0.2.{10 + i}:8080/login{i}.php" for i in range(12)]
    ben = [f"https://docs{i}.example.org/guide{i}" for i in range(12)]
    return mal + ben, [1] * 12 + [0] * 12


def _cal_corpus():
    # 12 per class: the installed sklearn routes CalibratedClassifierCV
    # (even with FrozenEstimator) through cross_val_predict with
    # StratifiedKFold(n_splits=5), which needs >= 5 samples per class.
    # Production calibration used 1993 validation rows; the earlier
    # 4-row fixture failed for exactly this reason.
    mal = [f"http://192.0.2.{30 + i}:8080/login{i}.php" for i in range(6)]
    mal += [f"http://198.51.100.{20 + i}:9000/auth{i}.php" for i in range(6)]
    ben = [f"https://docs{i}.example.org/guide{i}" for i in range(6)]
    ben += [f"https://manual{i}.example.net/reference{i}" for i in range(6)]
    return mal + ben, [1] * 12 + [0] * 12


def test_final_bundle_roundtrip(tmp_path):
    X, y = _tiny_corpus()
    pipe = Pipeline([
        ("features", URLFeatureBuilder(scale_dense=True)),
        ("model", LogisticRegression(C=1.0, solver="liblinear",
                                     max_iter=500, random_state=0)),
    ])
    pipe.fit(X, y)

    X_cal, y_cal = _cal_corpus()
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
    calibrated.fit(X_cal, y_cal)

    bundle = {"format_version": 1, "model_name": "test-logreg",
              "calibrated_model": calibrated, "raw_pipeline": pipe,
              "threshold": 0.42}
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    loaded = joblib.load(path)

    assert loaded["threshold"] == 0.42
    assert loaded["model_name"] == "test-logreg"
    restored = loaded["calibrated_model"].predict_proba(X)
    assert np.allclose(calibrated.predict_proba(X), restored)
    # elementwise: each malicious URL must outrank each benign URL
    assert (restored[:12, 1] > restored[12:, 1]).all()
