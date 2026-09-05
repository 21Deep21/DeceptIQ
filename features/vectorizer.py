"""Character n-gram TF-IDF vectorizer for URL strings.

Char n-grams capture brand impersonation / typosquatting ('paypa1'),
embedded lure keywords, and obfuscation patterns without word
tokenisation (URLs have no spaces).

LEAKAGE RULE (critical): this vectorizer is CONSTRUCTED here but ONLY
FITTED inside the Phase 2 training pipeline, strictly on the TRAINING
split, after the domain-aware (grouped) split. It is never fitted on
the full dataset. The fitted vectorizer is persisted with the model.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from appconfig import get_config


def build_char_tfidf() -> TfidfVectorizer:
    """Unfitted char-ngram TF-IDF vectorizer from config.yaml."""
    tf = get_config().get("features", {}).get("tfidf", {})
    lo, hi = tf.get("ngram_range", [3, 5])
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(int(lo), int(hi)),
        min_df=int(tf.get("min_df", 3)),
        max_features=int(tf.get("max_features", 5000)),
        sublinear_tf=True,
        # host is already lowercased by normalization; case in
        # path/query is signal, so do not lowercase again
        lowercase=False,
    )
