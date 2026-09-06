import os

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from features.feature_builder import URLFeatureBuilder
from features.feature_extraction import FEATURE_NAMES
from services.shap_service import ShapExplainer


def _corpus():
    # RFC 5737 documentation IPs + RFC 2606 reserved domains only.
    mal = [f"http://192.0.2.{10 + i}:8080/login{i}.php" for i in range(10)]
    mal += [f"http://198.51.100.{30 + i}:9000/auth{i}.php" for i in range(10)]
    ben = [f"https://docs{i}.example.org/guide{i}" for i in range(10)]
    ben += [f"https://manual{i}.example.net/reference{i}" for i in range(10)]
    return mal + ben, [1] * 20 + [0] * 20


@pytest.fixture(scope="module")
def tiny_model():
    from lightgbm import LGBMClassifier

    X, y = _corpus()
    pipe = Pipeline([
        ("features", URLFeatureBuilder()),
        ("model", LGBMClassifier(n_estimators=30, num_leaves=15,
                                 min_child_samples=1, random_state=0,
                                 n_jobs=1, verbose=-1)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


def test_reconstruction_identity(tiny_model):
    pipe, X, _ = tiny_model
    ex = ShapExplainer(pipe, backend="native")
    for e in ex.explain([X[0], X[-1]]):
        assert e.reconstruction_error < 1e-6
        assert abs((e.base_value + e.lexical_total + e.ngram_total)
                   - e.raw_margin) < 1e-6


def test_direction_partition_and_topk(tiny_model):
    pipe, X, _ = tiny_model
    ex = ShapExplainer(pipe, backend="native", top_k_default=5)
    mal_exp, ben_exp = ex.explain([X[0], X[-1]])
    assert mal_exp.increasing and all(c.shap > 0 for c in mal_exp.increasing)
    assert ben_exp.decreasing and all(c.shap < 0 for c in ben_exp.decreasing)
    assert len(mal_exp.increasing) <= 5 and len(mal_exp.decreasing) <= 5
    mags = [abs(c.shap) for c in mal_exp.increasing]
    assert mags == sorted(mags, reverse=True)
    # evidence direction agrees with the model's own margin
    assert (mal_exp.lexical_total + mal_exp.ngram_total) > 0
    assert (ben_exp.lexical_total + ben_exp.ngram_total) < 0


def test_totals_partition(tiny_model):
    pipe, X, _ = tiny_model
    ex = ShapExplainer(pipe, backend="native")
    e = ex.explain([X[3]])[0]
    assert abs((e.lexical_total + e.ngram_total)
               - (e.raw_margin - e.base_value)) < 1e-9


def test_feature_names_alignment(tiny_model):
    pipe, _, _ = tiny_model
    ex = ShapExplainer(pipe, backend="native")
    assert ex.feature_names[: len(FEATURE_NAMES)] == list(FEATURE_NAMES)
    assert all(n.startswith("ngram:") for n in ex.feature_names[len(FEATURE_NAMES):])


def test_auto_backend_falls_back_without_shap(tiny_model, monkeypatch):
    import services.shap_service as mod

    monkeypatch.setattr(mod, "shap", None)
    pipe, _, _ = tiny_model
    ex = ShapExplainer(pipe, backend="auto")
    assert ex.backend == "native"


@pytest.mark.skipif(
    os.environ.get("RUN_SHAP_TESTS") != "1",
    reason="slow (numba JIT) - set RUN_SHAP_TESTS=1 to enable",
)
def test_shap_backend_reconstruction(tiny_model):
    pytest.importorskip("shap")
    pipe, X, _ = tiny_model
    ex = ShapExplainer(pipe, backend="shap")
    e = ex.explain([X[0]])[0]
    assert e.backend == "shap"
    assert e.reconstruction_error < 1e-4


def test_explanation_url_is_sanitized(tiny_model):
    pipe, _, _ = tiny_model
    ex = ShapExplainer(pipe, backend="native")
    e = ex.explain(["https://user:secretpw@example.com/login"])[0]
    assert "secretpw" not in e.url
    assert "[REDACTED]" in e.url
    assert "secretpw" not in str(e.to_dict())


def test_sklearn_rf_shap_backend_probability_space():
    """Regression (Phase 3R): sklearn RandomForest TreeExplainer returns
    per-class shap_values shaped (n, m, 2) and explains PROBABILITIES (no
    log-odds margin). Must reduce to the positive class, reconstruct against
    predict_proba, and report space='probability'."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = _corpus()
    pipe = Pipeline([
        ("features", URLFeatureBuilder()),
        ("model", RandomForestClassifier(n_estimators=25, random_state=0, n_jobs=1)),
    ])
    pipe.fit(X, y)
    ex = ShapExplainer(pipe, backend="shap")
    assert ex.backend == "shap"
    assert ex.space == "probability"
    mal, ben = ex.explain([X[0], X[-1]])  # previously raised: 3D shape error

    p_mal = pipe.predict_proba([X[0]])[0, 1]
    assert mal.reconstruction_error is not None
    assert abs((mal.base_value + mal.lexical_total + mal.ngram_total) - p_mal) < 0.1
    assert (mal.lexical_total + mal.ngram_total) > 0   # evidence direction
    assert (ben.lexical_total + ben.ngram_total) < 0
    assert mal.to_dict()["space"] == "probability"
