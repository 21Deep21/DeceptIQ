from features.vectorizer import build_char_tfidf


def test_vectorizer_is_returned_unfitted():
    v = build_char_tfidf()
    assert not hasattr(v, "vocabulary_")  # leakage guard: construction never fits


def test_vectorizer_fits_only_on_provided_corpus():
    v = build_char_tfidf()
    corpus = [
        "https://secure-login-verify.example.com/",
        "https://paypa1-verify.example.net/",
        "http://198.51.100.7/cgi-bin/login.php",
        "https://example.org/documentation",
        "https://bit.ly/abc123",
    ]
    X = v.fit_transform(corpus)
    assert X.shape[0] == len(corpus)
    assert X.shape[1] > 0
