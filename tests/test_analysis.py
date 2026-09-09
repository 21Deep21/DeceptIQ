import pandas as pd

from model.analysis import ablation_rows, analyze_errors


def _tiny():
    mal = [f"http://192.0.2.{10 + i}:8080/login{i}.php" for i in range(12)]
    ben = [f"https://docs{i}.example.org/guide{i}" for i in range(12)]
    return mal + ben, [1] * 12 + [0] * 12


def _both_class_split():
    """Train/val split with BOTH classes on each side.

    The earlier X[:16]/X[16:] slicing put only benign URLs into
    validation (single-class y_true) -> compute_metrics returns NaN for
    roc_auc/pr_auc, and NaN != NaN makes even byte-identical results
    compare unequal. A test-design flaw, not model non-determinism.
    """
    X, y = _tiny()
    mal = [u for u, lab in zip(X, y) if lab == 1]
    ben = [u for u, lab in zip(X, y) if lab == 0]
    return (mal[:8] + ben[:8], [1] * 8 + [0] * 8,
            mal[8:] + ben[8:], [1] * 4 + [0] * 4)


def test_ablation_rows_three_modes():
    X_tr, y_tr, X_va, y_va = _both_class_split()
    rows = ablation_rows(X_tr, y_tr, X_va, y_va, seed=0)
    assert [r["feature_mode"] for r in rows] == ["combined", "lexical", "ngram"]
    for r in rows:
        for key in ("n_features", "accuracy", "precision", "recall", "f1",
                    "tn", "fp", "fn", "tp"):
            assert key in r
        assert 0.0 <= r["f1"] <= 1.0
        assert 0.0 <= r["roc_auc"] <= 1.0   # NaN fails <=, so this is a NaN guard
        assert 0.0 <= r["pr_auc"] <= 1.0
    assert rows[0]["n_features"] > rows[1]["n_features"]   # combined > lexical
    assert rows[0]["n_features"] > rows[2]["n_features"]   # combined > ngram


def test_ablation_rows_deterministic():
    X_tr, y_tr, X_va, y_va = _both_class_split()
    a = ablation_rows(X_tr, y_tr, X_va, y_va, seed=0)
    b = ablation_rows(X_tr, y_tr, X_va, y_va, seed=0)
    # fit_seconds is wall-clock timing, not a result - exclude it
    strip = lambda rows: [{k: v for k, v in r.items() if k != "fit_seconds"} for r in rows]
    assert strip(a) == strip(b)


def _err_df():
    return pd.DataFrame([
        {"url": "https://docs.example.org/a?x=1", "label": 0, "source": "sitemap",
         "threat_type": "benign", "registrable_domain": "example.org",
         "calibrated_probability": 0.31, "predicted_label": 1},
        {"url": "https://home.example.com/", "label": 0, "source": "majestic",
         "threat_type": "benign", "registrable_domain": "example.com",
         "calibrated_probability": 0.12, "predicted_label": 1},
        {"url": "http://clean.example.net/login", "label": 1, "source": "openphish",
         "threat_type": "phishing", "registrable_domain": "example.net",
         "calibrated_probability": 0.03, "predicted_label": 0},
    ])


def test_analyze_errors_buckets():
    out = analyze_errors(_err_df())
    assert out["total_errors"] == 3
    assert out["false_positives"]["count"] == 2
    assert out["false_positives"]["by_source"] == {"sitemap": 1, "majestic": 1}
    assert out["false_negatives"]["count"] == 1
    assert out["false_negatives"]["by_source"] == {"openphish": 1}


def test_analyze_errors_shape_and_confidence():
    out = analyze_errors(_err_df())
    shape = out["false_positives"]["url_shape"]
    assert shape["https_share"] == 1.0
    assert shape["has_query_share"] == 0.5
    assert shape["path_gt1_share"] == 0.5
    top = out["false_positives"]["most_confident_5"]
    assert top[0]["url"].endswith("a?x=1")
    assert abs(top[0]["calibrated_probability"] - 0.31) < 1e-9
    fn_top = out["false_negatives"]["most_confident_5"]
    assert fn_top[0]["calibrated_probability"] == 0.03
