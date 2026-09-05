"""URL feature-matrix builder: dense lexical features + char-ngram TF-IDF.

fit() learns ONLY from the URLs it is given (the training fold / training
split): the TF-IDF vectorizer and the optional StandardScaler (dense
block, used for logistic regression) are fitted here and NOWHERE else.
transform() applies the learned state, so training and inference always
compute features identically (no training-serving skew).

Leakage rule enforced structurally: this builder is only ever fitted
inside the training pipeline (per CV fold, or on the full training
split) - never on the full dataset before splitting.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from features.feature_extraction import FEATURE_NAMES, extract_lexical_features
from features.vectorizer import build_char_tfidf


@lru_cache(maxsize=131072)
def _dense_row(url: str) -> Tuple[float, ...]:
    """Cached dense feature row for a normalized URL.

    Safe to cache: lexical features are a pure function of the URL
    string. This makes repeated CV folds over the same URLs cheap.
    """
    feats = extract_lexical_features(url)
    return tuple(float(feats[name]) for name in FEATURE_NAMES)


class URLFeatureBuilder(BaseEstimator, TransformerMixin):
    """Transformer: raw URL strings -> sparse feature matrix.

    scale_dense=True standardizes the 26 dense columns (for the linear
    model); tree models consume raw values and skip scaling.
    """

    def __init__(self, scale_dense: bool = False):
        self.scale_dense = scale_dense

    def _dense_matrix(self, X) -> np.ndarray:
        return np.asarray([_dense_row(url) for url in X], dtype=np.float64)

    def fit(self, X, y=None):
        urls = list(X)
        dense = self._dense_matrix(urls)
        self.tfidf_ = build_char_tfidf().fit(urls)
        if self.scale_dense:
            self.scaler_ = StandardScaler().fit(dense)
        else:
            self.scaler_ = None
        ngram_names = [f"ngram:{name}" for name in self.tfidf_.get_feature_names_out()]
        self.feature_names_ = list(FEATURE_NAMES) + ngram_names
        return self

    def transform(self, X):
        dense = self._dense_matrix(X)
        if self.scaler_ is not None:
            dense = self.scaler_.transform(dense)
        Xt = self.tfidf_.transform(list(X))
        dense_csr = csr_matrix(np.asarray(dense, dtype=np.float64))
        return hstack([dense_csr, Xt], format="csr")
