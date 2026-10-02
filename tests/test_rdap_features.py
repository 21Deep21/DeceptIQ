import json

import pytest
import requests

from features.rdap_features import parse_rdap_payload, query_rdap, rdap_response_sha256

BASE = "https://unit.invalid/domain/"

PAYLOAD = {
    "events": [
        {"eventAction": "registration", "eventDate": "1995-08-14T04:00:00Z"},
        {"eventAction": "expiration", "eventDate": "2027-08-13T04:00:00Z"},
    ],
    "entities": [{"roles": ["registrar"],
                  "vcardArray": ["vcard",
                                 [["fn", {}, "text", "Example Registrar, Inc."]]]}],
    "status": ["active"],
    "nameservers": [{"ldhName": "B.IANA-SERVERS.NET"}, {"ldhName": "A.IANA-SERVERS.NET"}],
}


class FakeResp:
    def __init__(self, code, payload=None):
        self.status_code = code
        self._payload = payload
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, mapping=None, exc=None):
        self.mapping = mapping or {}
        self.exc = exc

    def get(self, url, timeout=None, headers=None, **kw):
        if self.exc is not None:
            raise self.exc
        return self.mapping.get(url, FakeResp(404))


def test_parse_rdap_payload_full():
    p = parse_rdap_payload(PAYLOAD)
    assert p["registrar"] == "Example Registrar, Inc."
    assert p["creation_date"] == "1995-08-14T04:00:00Z"
    assert p["expiration_date"] == "2027-08-13T04:00:00Z"
    assert p["nameservers"] == ["a.iana-servers.net", "b.iana-servers.net"]
    assert p["status"] == ["active"]


def test_parse_rdap_payload_redacted_yields_none():
    p = parse_rdap_payload({"events": []})
    assert p["registrar"] is None and p["creation_date"] is None
    assert p["nameservers"] == [] and p["status"] == []


def test_query_rdap_ok():
    info = query_rdap("example.com", base_url=BASE,
                      session=FakeSession({BASE + "example.com": FakeResp(200, PAYLOAD)}))
    assert info.status_code == "ok"
    assert info.registrar == "Example Registrar, Inc."
    assert info.raw_json and rdap_response_sha256(info)
    assert info.error is None


def test_query_rdap_status_mapping():
    s = lambda r: query_rdap("x.test", base_url=BASE, session=r).status_code
    assert s(FakeSession()) == "not_found"                       # default 404
    assert s(FakeSession({BASE + "x.test": FakeResp(429)})) == "rate_limited"
    assert s(FakeSession({BASE + "x.test": FakeResp(503)})) == "rate_limited"
    assert s(FakeSession({BASE + "x.test": FakeResp(500)})) == "error"
    assert s(FakeSession(exc=requests.Timeout("t"))) == "timeout"
    assert s(FakeSession(exc=requests.ConnectionError("c"))) == "error"


def test_query_rdap_invalid_json_is_error():
    r = FakeSession({BASE + "x.test": FakeResp(200)})            # 200, no payload
    assert query_rdap("x.test", base_url=BASE, session=r).status_code == "error"


def test_no_credentials_in_any_output():
    info = query_rdap("example.com", base_url=BASE,
                      session=FakeSession({BASE + "example.com": FakeResp(200, PAYLOAD)}))
    blob = str(info) + str(parse_rdap_payload(PAYLOAD))
    assert "password" not in blob.lower() and "secret" not in blob.lower()
