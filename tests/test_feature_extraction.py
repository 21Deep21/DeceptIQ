from features.feature_extraction import (
    FEATURE_NAMES,
    extract_lexical_features,
    safe_extract_features,
)

# RFC 2606 reserved domains only - no real targets in test code.
URL = "https://user:pass@sub.shop.example.co.uk:8443/cgi-bin/login?session=abc&retry=1#top"


def test_comprehensive_url():
    f = extract_lexical_features(URL)
    assert f["url_length"] == len(URL)  # URL is already in normalized form
    assert f["hostname_length"] == len("sub.shop.example.co.uk")
    assert f["path_length"] == len("/cgi-bin/login")
    assert f["query_length"] == len("session=abc&retry=1")
    assert f["fragment_length"] == len("top")
    assert f["num_dots"] == 4
    assert f["num_slashes"] == 4
    assert f["num_digits"] == 5  # 8443 + the '1' in retry=1
    assert f["subdomain_count"] == 2  # sub, shop
    assert f["num_query_params"] == 2
    assert f["has_ip_hostname"] == 0
    assert f["has_at_symbol"] == 1
    assert f["is_https"] == 1
    assert f["has_explicit_port"] == 1
    assert f["is_unusual_port"] == 1  # 8443 not in {80, 443}
    assert f["suspicious_keyword_hits"] == 2  # 'login', 'session'
    assert f["is_url_shortener"] == 0
    assert f["num_path_segments"] == 2
    assert 0.0 < f["digit_ratio"] < 1.0
    assert 0.0 < f["special_char_ratio"] < 1.0
    assert 0.0 < f["url_entropy"] < 5.0
    assert 0.0 < f["hostname_entropy"] < 5.0


def test_ip_host_features():
    f = extract_lexical_features("http://10.0.0.1/admin")
    assert f["has_ip_hostname"] == 1
    assert f["subdomain_count"] == 0
    assert f["is_url_shortener"] == 0


def test_shortener_detection():
    assert extract_lexical_features("https://bit.ly/3xYz9")["is_url_shortener"] == 1
    assert extract_lexical_features("https://notbit.ly.example.com/")["is_url_shortener"] == 0


def test_safe_extract_returns_none_on_malformed():
    assert safe_extract_features("not a url at all") is None
    assert safe_extract_features("http://") is None


def test_feature_names_match_feature_dict():
    f = extract_lexical_features("https://example.com/")
    assert set(f.keys()) == set(FEATURE_NAMES)
