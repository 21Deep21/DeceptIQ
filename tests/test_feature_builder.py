import numpy as np
from sklearn.base import clone

from features.feature_builder import URLFeatureBuilder
from features.feature_extraction import FEATURE_NAMES

CORPUS = [
    "https://secure-login.example.com/verify/account",
    "https://paypa1-verify.example.net/session?id=12",
    "http://192.0.2.10:8080/cgi-bin/login.php",
    "https://docs.example.org/documentation/index",
    "https://bit.ly/abc123",
    "http://xjkqp91kd.example.xyz/login",
]


def test_fit_transform_shapes_and_names():
    b = URLFeatureBuilder().fit(CORPUS)
    X = b.transform(CORPUS)
    n_dense = len(FEATURE_NAMES)
    assert X.shape[0] == len(CORPUS)
    assert X.shape[1] == n_dense + len(b.tfidf_.vocabulary_)
    assert b.feature_names_[:n_dense] == list(FEATURE_NAMES)
    assert all(name.startswith("ngram:") for name in b.feature_names_[n_dense:])


def test_unseen_data_introduces_no_new_features():
    b = URLFeatureBuilder().fit(CORPUS)
    X1 = b.transform(CORPUS)
    X2 = b.transform(["https://unseen.example.com/never-seen-before"])
    assert X2.shape[1] == X1.shape[1]  # leakage guard: vocabulary fixed at fit time


def test_dense_scaling_standardizes_varying_columns():
    b = URLFeatureBuilder(scale_dense=True).fit(CORPUS)
    X = b.transform(CORPUS)
    idx = list(FEATURE_NAMES).index("url_length")
    col = np.asarray(X[:, idx].todense()).ravel()
    assert abs(col.mean()) < 1e-9
    assert abs(col.std() - 1.0) < 1e-6


def test_builder_is_cloneable():
    b = URLFeatureBuilder(scale_dense=True)
    c = clone(b)
    assert c.scale_dense is True
    assert not hasattr(c, "tfidf_")  # clone must be unfitted
