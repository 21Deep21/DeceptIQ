"""Lexical URL feature extraction - fully OFFLINE.

Every feature is computed from the URL string alone; the URL is never
visited or requested. Features are computed on the NORMALIZED URL
(features.url_utils.normalize_url), and the same normalization is used
at dataset-construction and inference time (no training-serving skew).

Each feature encodes a pattern documented in phishing-detection
literature: long/complex URLs, many subdomains, IP hosts, userinfo '@'
tricks, lure keywords, shorteners, heavy percent-encoding. NONE of them
is individually proof of phishing - they are weak signals the model
weighs jointly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlsplit

from features.entropy import shannon_entropy
from features.url_utils import (
    MalformedURLError,
    is_ip_hostname,
    normalize_url,
    registrable_domain_of_host,
    subdomain_count,
)

# Heuristic lure vocabulary observed in phishing URLs. Weak signal only:
# these words also appear on legitimate sites; the model learns the weight.
SUSPICIOUS_KEYWORDS = (
    "login", "signin", "logon", "verify", "verification", "validate",
    "secure", "security", "account", "update", "confirm", "banking",
    "password", "passwd", "webscr", "invoice", "payment", "billing",
    "suspend", "unlock", "recovery", "authentication", "identity",
    "wallet", "alert", "urgent", "refund", "session", "token",
    "credential", "authorize",
)

# Well-known URL shortener registrable domains (static, documented list).
SHORTENER_DOMAINS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
    "buff.ly", "cutt.ly", "rb.gy", "rebrand.ly", "shorturl.at",
    "tiny.cc", "bit.do", "s.id", "shorte.st", "alturl.com", "v.gd",
})

FEATURE_NAMES = (
    "url_length", "hostname_length", "path_length", "query_length",
    "fragment_length", "num_dots", "num_hyphens", "num_underscores",
    "num_digits", "num_special_chars", "num_slashes", "subdomain_count",
    "num_query_params", "has_ip_hostname", "has_at_symbol", "is_https",
    "has_explicit_port", "is_unusual_port", "num_percent",
    "suspicious_keyword_hits", "is_url_shortener", "num_path_segments",
    "digit_ratio", "special_char_ratio", "url_entropy", "hostname_entropy",
)


def extract_lexical_features(url: str) -> Dict[str, Any]:
    """26 lexical features from a single URL string.

    Raises MalformedURLError for invalid input (malformed URLs must
    crash nothing - callers decide whether to drop or report)."""
    norm = normalize_url(url)
    parts = urlsplit(norm)
    hostport = parts.netloc.rpartition("@")[2]
    is_ipv6 = hostport.startswith("[")
    if is_ipv6:
        host = hostport[1 : hostport.rfind("]")]
    else:
        host = hostport.partition(":")[0]
    path, query = parts.path, parts.query
    lowered = norm.lower()

    n = len(norm)
    num_digits = sum(c.isdigit() for c in norm)
    num_special = sum(not c.isalnum() for c in norm)  # non-alphanumeric chars
    port = parts.port  # already validated by normalize_url
    host_is_ip = is_ipv6 or is_ip_hostname(host)
    registered = None if host_is_ip else registrable_domain_of_host(host)

    features = {
        "url_length": n,
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(parts.fragment),
        "num_dots": norm.count("."),
        "num_hyphens": norm.count("-"),
        "num_underscores": norm.count("_"),
        "num_digits": num_digits,
        "num_special_chars": num_special,
        "num_slashes": norm.count("/"),
        "subdomain_count": subdomain_count(norm),
        "num_query_params": len(parse_qsl(query, keep_blank_values=True)),
        "has_ip_hostname": int(host_is_ip),
        "has_at_symbol": int("@" in norm),
        "is_https": int(parts.scheme == "https"),
        "has_explicit_port": int(port is not None),
        "is_unusual_port": int(port is not None and port not in (80, 443)),
        "num_percent": norm.count("%"),
        # DISTINCT keywords present (avoids double-counting overlaps
        # like 'bank' inside 'banking')
        "suspicious_keyword_hits": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lowered),
        "is_url_shortener": int(registered is not None and registered in SHORTENER_DOMAINS),
        "num_path_segments": path.count("/") if path else 0,
        "digit_ratio": (num_digits / n) if n else 0.0,
        "special_char_ratio": (num_special / n) if n else 0.0,
        "url_entropy": shannon_entropy(norm),
        "hostname_entropy": shannon_entropy(host),
    }
    assert set(features) == set(FEATURE_NAMES)  # guards against drift
    return features


def safe_extract_features(url: str) -> Optional[Dict[str, Any]]:
    """None on malformed input instead of raising."""
    try:
        return extract_lexical_features(url)
    except MalformedURLError:
        return None
