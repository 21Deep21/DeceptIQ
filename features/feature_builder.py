"""URL feature-matrix builder: dense lexical features + char-ngram TF-IDF.

fit() learns ONLY from the URLs it is given (the training fold / training
split): the TF-IDF vectorizer and the optional StandardScaler are fitted
here and nowhere else. transform() applies the learned state, so training
and inference always compute features identically (no training-serving skew).

feature_mode (v1.1, ablation study):
  * "combined" (default, DEPLOYED): 26 dense lexical features + TF-IDF
  * "lexical"  : dense features only
  * "ngram"    : TF-IDF only

PICKLE COMPATIBILITY (v1.1 regression fix): bundles persisted with the
v1.0 class definition contain NO 'feature_mode' attribute, because pickle
restores __dict__ without calling __init__. A missing attribute means the
object was saved as a v1.0 'combined' builder - so transform() reads the
mode through _effective_mode() with that default. This is what keeps the
DEPLOYED v1.0 bundle loadable and servable after the v1.1 class change.
(Nothing in the serving path calls clone()/get_params() on a loaded
builder; only newly constructed builders are cloned.)
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

FEATURE_MODES = ("combined", "lexical", "ngram")
_DEFAULT_MODE = "combined"


def _effective_mode(builder) -> str:
    """feature_mode with the v1.0-pickle default ('combined')."""
    return getattr(builder, "feature_mode", _DEFAULT_MODE)


@lru_cache(maxsize=131072)
def _dense_row(url: str) -> Tuple[float, ...]:
    """Cached (deterministic) dense feature row for a normalized URL.

    Safe to cache: lexical features are a pure function of the URL string.
    """
    feats = extract_lexical_features(url)
    return tuple(float(feats[name]) for name in FEATURE_NAMES)


class URLFeatureBuilder(BaseEstimator, TransformerMixin):
    """Transformer: raw URL strings -> sparse feature matrix."""

    def __init__(self, scale_dense: bool = False, feature_mode: str = "combined"):
        self.scale_dense = scale_dense
        self.feature_mode = feature_mode

    def _dense_matrix(self, X) -> np.ndarray:
        return np.asarray([_dense_row(url) for url in X], dtype=np.float64)

    def fit(self, X, y=None):
        mode = _effective_mode(self)
        if mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {mode!r}")
        urls = list(X)
        dense = self._dense_matrix(urls)

        if self.scale_dense and mode != "ngram":
            self.scaler_ = StandardScaler().fit(dense)
        else:
            self.scaler_ = None

        if mode != "lexical":
            self.tfidf_ = build_char_tfidf().fit(urls)
        else:
            self.tfidf_ = None

        if mode == "lexical":
            self.feature_names_ = list(FEATURE_NAMES)
        elif mode == "ngram":
            self.feature_names_ = [f"ngram:{n}" for n in self.tfidf_.get_feature_names_out()]
        else:
            self.feature_names_ = (
                list(FEATURE_NAMES)
                + [f"ngram:{n}" for n in self.tfidf_.get_feature_names_out()]
            )
        return self

    def transform(self, X):
        mode = _effective_mode(self)
        if mode == "lexical":
            dense = self._dense_matrix(X)
            if self.scaler_ is not None:
                dense = self.scaler_.transform(dense)
            return csr_matrix(np.asarray(dense, dtype=np.float64))
        if mode == "ngram":
            return self.tfidf_.transform(list(X))
        dense = self._dense_matrix(X)
        if self.scaler_ is not None:
            dense = self.scaler_.transform(dense)
        Xt = self.tfidf_.transform(list(X))
        dense_csr = csr_matrix(np.asarray(dense, dtype=np.float64))
        return hstack([dense_csr, Xt], format="csr")
