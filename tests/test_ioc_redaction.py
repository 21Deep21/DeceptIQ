import json

from features.ioc_extraction import extract_iocs, sanitize_url_for_logging


def test_query_parameter_credentials_redacted():
    r = extract_iocs("https://example.com/login?password=hunter2&session=abc&user=bob")
    params = {p["name"]: p["value"] for p in r["parameters"]}
    assert params["password"] == "[REDACTED]"
    assert params["session"] == "abc"   # session/token values are IOC evidence, kept
    assert "hunter2" not in json.dumps(r)
    assert r["components"]["query"] == "password=[REDACTED]&session=abc&user=bob"
    assert "hunter2" not in r["components"]["url"]


def test_partial_credential_name_not_redacted():
    r = extract_iocs("https://example.com/?sypass=xyz")
    assert r["parameters"][0]["value"] == "xyz"


def test_sanitize_url_for_logging_redacts_query_credentials():
    sanitized = sanitize_url_for_logging("https://example.com/l?password=hunter2&x=1")
    assert sanitized == "https://example.com/l?password=[REDACTED]&x=1"
    assert "hunter2" not in sanitized


from features.ioc_extraction import strip_url_credentials


def test_strip_url_credentials_removes_userinfo_keeps_host_port():
    assert (strip_url_credentials("https://user:pw@example.com:8443/l?x=1")
            == "https://example.com:8443/l?x=1")


def test_strip_url_credentials_redacts_query_credentials():
    assert (strip_url_credentials("https://example.com/l?password=hunter2&x=1")
            == "https://example.com/l?password=[REDACTED]&x=1")


def test_strip_url_credentials_never_raises():
    assert isinstance(strip_url_credentials("http://[::1"), str)
    assert isinstance(strip_url_credentials("not a url"), str)
