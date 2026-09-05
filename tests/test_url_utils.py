import pytest

from features.url_utils import (
    MalformedURLError,
    extract_registrable_domain,
    is_ip_hostname,
    normalize_url,
)


def test_normalize_lowercases_scheme_and_host_only():
    assert normalize_url("HTTP://WWW.Example.COM/Path") == "http://www.example.com/Path"


def test_normalize_strips_outer_whitespace():
    assert normalize_url("   https://example.com/x\t\n") == "https://example.com/x"


def test_normalize_preserves_userinfo_port_query_fragment():
    url = "https://user:pass@example.com:8443/login?x=1#top"
    assert normalize_url(url) == url


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "notaurl",
        "ftp://example.com/",
        "javascript:alert(1)",
        "http://",
        "://example.com/",
        "http://exa mple.com/",
        "http://exa\tmple.com/",
        "http://" + "a" * 3000,
        "http://bad..host/",
        "http://.leading.dot/",
    ],
)
def test_malformed_urls_rejected(bad):
    with pytest.raises(MalformedURLError):
        normalize_url(bad)


@pytest.mark.parametrize(
    "host,expected",
    [
        ("1.2.3.4", True),
        ("255.255.255.255", True),
        ("999.1.1.1", False),
        ("example.com", False),
        ("::1", False),  # unbracketed IPv6 is not a valid URL host
        ("[::1]", True),
    ],
)
def test_is_ip_hostname(host, expected):
    assert is_ip_hostname(host) is expected


def test_registrable_domain_multilevel_public_suffix():
    assert extract_registrable_domain("https://a.b.example.co.uk/x") == "example.co.uk"


def test_registrable_domain_basic():
    assert extract_registrable_domain("http://login.example.com/") == "example.com"


def test_registrable_domain_ip_literal():
    assert extract_registrable_domain("http://192.168.10.5:8080/") == "192.168.10.5"


@pytest.mark.parametrize("url", ["http://localhost/", "http://co.uk/"])
def test_no_registrable_domain_raises(url):
    with pytest.raises(MalformedURLError):
        extract_registrable_domain(url)
