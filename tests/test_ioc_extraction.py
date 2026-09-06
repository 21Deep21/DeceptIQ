import pytest

from features.ioc_extraction import extract_iocs, sanitize_url_for_logging
from features.url_utils import MalformedURLError

SPEC_URL = "https://user:s3cr3t@example.com:443/login/index.html?session=123&x=1#top"


def test_full_decomposition_spec_example():
    r = extract_iocs(SPEC_URL)
    c = r["components"]
    assert c["scheme"] == "https"
    assert c["host"] == "example.com"
    assert c["registrable_domain"] == "example.com"
    assert c["subdomain"] is None
    assert c["port"] == 443
    assert c["path"] == "/login/index.html"
    assert c["query"] == "session=123&x=1"
    assert c["fragment"] == "top"
    assert c["is_ip"] is False and c["ip"] is None
    assert r["parameters"] == [
        {"name": "session", "value": "123"}, {"name": "x", "value": "1"}
    ]


def test_password_is_redacted_and_absent_from_output():
    r = extract_iocs(SPEC_URL)
    assert r["components"]["userinfo"] == "user:[REDACTED]"
    blob = repr(r)
    assert "s3cr3t" not in blob  # raw password never appears in structured output


def test_ip_host_components_and_flags():
    r = extract_iocs("http://192.0.2.10:8080/pay%2Fload?redirect=abc&session=1")
    c = r["components"]
    assert c["is_ip"] is True
    assert c["ip"] == "192.0.2.10"
    assert c["registrable_domain"] is None
    assert c["host"] == "192.0.2.10"
    assert c["port"] == 8080
    names = {f["flag"] for f in r["flags"]}
    assert {"ip_host", "unusual_port", "percent_encoding"} <= names


def test_unusual_port_flag():
    r = extract_iocs("https://example.com:8443/x")
    assert any(f["flag"] == "unusual_port" for f in r["flags"])


def test_percent_encoding_decoded_samples():
    r = extract_iocs("https://example.com/a%62c")
    assert any(f["flag"] == "percent_encoding" for f in r["flags"])
    # %62 decodes to 'b'
    row = next(i for i in r["iocs"] if i["type"] == "Encoded")
    assert "%62" in row["value"] and "'b'" in row["value"]


def test_double_encoding_flag():
    r = extract_iocs("http://example.com/%252e%252e")
    assert any(f["flag"] == "double_encoding" for f in r["flags"])


def test_many_query_params_flag():
    r = extract_iocs("https://example.com/?a=1&b=2&c=3&d=4&e=5&f=6")
    assert any(f["flag"] == "many_query_params" for f in r["flags"])


def test_suspicious_parameter_names_flag():
    r = extract_iocs(SPEC_URL)
    flag = next(f for f in r["flags"] if f["flag"] == "suspicious_parameter_names")
    assert "session" in flag["detail"]
    row = next(i for i in r["iocs"] if i["type"] == "Parameter" and "session" in i["value"])
    assert row["status"] == "Suspicious name"


def test_long_path_flag():
    r = extract_iocs("https://example.com/" + "a" * 150)
    assert any(f["flag"] == "long_path" for f in r["flags"])


def test_shortener_domain_flag():
    r = extract_iocs("https://bit.ly/abc123")
    assert any(f["flag"] == "url_shortener" for f in r["flags"])
    row = next(i for i in r["iocs"] if i["type"] == "Domain")
    assert row["status"] == "URL shortener"


def test_punycode_host_flag():
    r = extract_iocs("http://xn--80ak6aa92e.com/")
    assert any(f["flag"] == "punycode_host" for f in r["flags"])
    assert r["components"]["registrable_domain"] == "xn--80ak6aa92e.com"


def test_path_extension_flags():
    r1 = extract_iocs("http://192.0.2.5:8080/download/update.exe")
    assert any(f["flag"] == "executable_extension" for f in r1["flags"])
    r2 = extract_iocs("https://example.com/login.php")
    assert any(f["flag"] == "script_extension" for f in r2["flags"])


def test_malformed_url_raises():
    with pytest.raises(MalformedURLError):
        extract_iocs("not a url")


def test_ioc_table_covers_core_types():
    r = extract_iocs(SPEC_URL)
    types = {row["type"] for row in r["iocs"]}
    assert {"Scheme", "Userinfo", "Domain", "Subdomain", "TLD", "IP", "Port",
            "Path", "Parameter", "Fragment"} <= types


def test_sanitize_url_for_logging_redacts_password():
    assert (sanitize_url_for_logging("https://user:s3cr3t@example.com:8443/x")
            == "https://user:[REDACTED]@example.com:8443/x")


def test_sanitize_url_for_logging_never_raises():
    assert sanitize_url_for_logging("https://plain.example.com/x") == "https://plain.example.com/x"
    assert sanitize_url_for_logging("https://user@example.com/") == "https://user@example.com/"
    assert sanitize_url_for_logging("http://[::1") == "http://[::1"  # invalid input: passthrough
