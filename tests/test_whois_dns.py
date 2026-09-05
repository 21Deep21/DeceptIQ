import datetime
import os

import pytest

from features.dns_features import DnsInfo, query_dns
from features.whois_features import _to_date, query_whois

RUN_NETWORK = os.environ.get("RUN_NETWORK_TESTS") == "1"
requires_network = pytest.mark.skipif(
    not RUN_NETWORK, reason="network test - set RUN_NETWORK_TESTS=1 to enable"
)


def test_to_date_handles_python_whois_shapes():
    assert _to_date(datetime.datetime(2020, 1, 2, 12, 30)) == datetime.date(2020, 1, 2)
    assert _to_date(datetime.date(2021, 3, 4)) == datetime.date(2021, 3, 4)
    assert _to_date([datetime.datetime(2020, 1, 2), None]) == datetime.date(2020, 1, 2)
    assert _to_date("2021-05-06") == datetime.date(2021, 5, 6)
    assert _to_date("not a date") is None
    assert _to_date(None) is None
    assert _to_date([]) is None


@requires_network
def test_whois_returns_gracefully():
    info = query_whois("example.com", timeout=10)
    assert isinstance(info.domain, str)
    assert info.error is None or isinstance(info.error, str)


@requires_network
def test_whois_invalid_tld_reports_unavailable():
    info = query_whois("nonexistent-domain-xyz.invalid", timeout=10)
    assert info.domain_age_days is None  # unavailable, never a phishing signal
    assert info.error


@requires_network
def test_dns_returns_gracefully():
    info = query_dns("example.com", timeout=10)
    assert isinstance(info, DnsInfo)
    assert info.a_record_count is None or info.a_record_count >= 0
    assert info.has_mx is None or isinstance(info.has_mx, bool)


@requires_network
def test_dns_invalid_tld_reports_unavailable():
    info = query_dns("nonexistent-domain-xyz.invalid", timeout=10)
    assert info.a_record_count is None
    assert info.error
