import base64
import os

import pytest
import requests

from services.virustotal_service import ENV_KEY, VirusTotalClient, url_identifier

BASE_CFG = {
    "virustotal": {
        "enabled": True,
        "base_url": "https://unit.invalid/api/v3",
        "timeout_seconds": 2,
        "min_request_interval_seconds": 0,
    }
}


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {}), timeout))
        if self.exc is not None:
            raise self.exc
        return self.response


def _mk(monkeypatch, session, cfg=None):
    monkeypatch.setenv(ENV_KEY, "test-key")
    return VirusTotalClient(cfg or BASE_CFG, session=session)


def test_url_identifier_is_unpadded_urlsafe_base64():
    url = "https://example.com/some/path?x=1"
    ident = url_identifier(url)
    assert "=" not in ident
    assert all(c.isalnum() or c in "-_" for c in ident)
    padded = ident + "=" * (-len(ident) % 4)
    assert base64.urlsafe_b64decode(padded).decode("utf-8") == url


def test_status_disabled_without_key(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    vt = VirusTotalClient(BASE_CFG, session=FakeSession())
    assert vt.available is False
    assert vt.status()["status"] == "disabled"
    assert ENV_KEY in vt.status()["reason"]


def test_status_disabled_by_config(monkeypatch):
    monkeypatch.setenv(ENV_KEY, "k")
    cfg = dict(BASE_CFG, virustotal=dict(BASE_CFG["virustotal"], enabled=False))
    vt = VirusTotalClient(cfg, session=FakeSession())
    assert vt.status()["status"] == "disabled"
    assert "config" in vt.status()["reason"]


def test_lookup_disabled_makes_no_request(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    session = FakeSession()
    vt = VirusTotalClient(BASE_CFG, session=session)
    result = vt.lookup("https://example.com/")
    assert result["status"] == "disabled"
    assert session.calls == []


def test_lookup_not_found(monkeypatch):
    vt = _mk(monkeypatch, FakeSession(FakeResponse(404)))
    result = vt.lookup("https://example.com/never-scanned")
    assert result["status"] == "not_found"
    assert "never submitted" in result["reason"]
    assert "Not ground truth" in result["note"]


def test_lookup_ok_parses_engine_stats(monkeypatch):
    payload = {"data": {"attributes": {
        "last_analysis_stats": {"harmless": 70, "malicious": 12, "suspicious": 3,
                                "undetected": 15, "timeout": 0},
        "reputation": -12,
        "last_analysis_date": 1700000000,
    }}}
    vt = _mk(monkeypatch, FakeSession(FakeResponse(200, payload)))
    result = vt.lookup("https://example.com/")
    assert result["status"] == "ok"
    assert result["engines_detected"] == 15        # malicious + suspicious
    assert result["engines_total"] == 100
    assert result["reputation"] == -12
    assert result["last_analysis_stats"]["malicious"] == 12
    assert result["last_analysis_date"].startswith("20")
    assert "separately from the local ML verdict" in result["note"]


def test_lookup_rate_limited_429(monkeypatch):
    vt = _mk(monkeypatch, FakeSession(FakeResponse(429)))
    result = vt.lookup("https://example.com/")
    assert result["status"] == "rate_limited"
    assert "429" in result["reason"]


def test_lookup_invalid_key_401(monkeypatch):
    vt = _mk(monkeypatch, FakeSession(FakeResponse(401)))
    result = vt.lookup("https://example.com/")
    assert result["status"] == "error"
    assert "401" in result["reason"]
    assert "test-key" not in str(result)  # the key is never echoed anywhere


def test_lookup_network_failure(monkeypatch):
    vt = _mk(monkeypatch, FakeSession(exc=requests.ConnectionError("no route")))
    result = vt.lookup("https://example.com/")
    assert result["status"] == "error"
    assert "ConnectionError" in result["reason"]


def test_credentials_never_transmitted(monkeypatch):
    session = FakeSession(FakeResponse(404))
    vt = _mk(monkeypatch, session)
    vt.lookup("https://user:pw123@example.com/l?password=hunter2&x=1")
    called_url = session.calls[0][0]
    ident = called_url.rsplit("/urls/", 1)[1]
    padded = ident + "=" * (-len(ident) % 4)
    decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
    assert "pw123" not in decoded and "user" not in decoded
    assert "hunter2" not in decoded
    assert decoded == "https://example.com/l?password=[REDACTED]&x=1"
    # the API key travels in the header, never in the URL
    assert session.calls[0][1]["x-apikey"] == "test-key"


def test_local_throttle_refuses_rapid_second_request(monkeypatch):
    cfg = dict(BASE_CFG, virustotal=dict(BASE_CFG["virustotal"],
                                         min_request_interval_seconds=3600))
    vt = _mk(monkeypatch, FakeSession(FakeResponse(200)), cfg=cfg)
    first = vt.lookup("https://example.com/")
    second = vt.lookup("https://example.com/")
    assert first["status"] == "ok"
    assert second["status"] == "rate_limited"
    assert "local throttle" in second["reason"]


@pytest.mark.skipif(
    os.environ.get("RUN_VT_TESTS") != "1" or not os.environ.get(ENV_KEY),
    reason="live VirusTotal test - set RUN_VT_TESTS=1 and VT_API_KEY to enable",
)
def test_live_lookup_example_com():
    from appconfig import get_config
    vt = VirusTotalClient(get_config())
    result = vt.lookup("https://example.com/")
    assert result["status"] in ("ok", "rate_limited", "not_found")
    if result["status"] == "ok":
        assert result["engines_total"] > 0
