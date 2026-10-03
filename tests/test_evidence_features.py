import json

import numpy as np
import pytest

from features.evidence_features import (
    EVIDENCE_FEATURE_NAMES,
    GROUP_COLUMNS,
    N_EVIDENCE_FEATURES,
    evidence_log_fingerprint,
    evidence_matrix_for_domains,
    load_evidence_matrix,
    mask_groups,
)

DNS_OK = {"domain": "d.test", "group": "dns", "status": "ok",
          "requested_at": "2026-10-02T17:38:10+00:00", "duration_ms": 100.0,
          "fields": {"a_record_count": 3, "a_record_ttl": 300, "has_mx": True,
                     "ns_count": 2, "ns_diversity": 2}}
RDAP_OK = {"domain": "d.test", "group": "rdap", "status": "ok",
           "requested_at": "2026-10-02T17:39:20+00:00", "duration_ms": 200.0,
           "fields": {"registrar": "Example Registrar",
                      "creation_date": "2016-10-02T00:00:00+00:00",
                      "expiration_date": "2036-10-02T00:00:00+00:00",
                      "status": ["active"],
                      "nameservers": ["a.ns.test", "b.ns.test"]}}


def _write_logs(tmp_path, dns_recs, rdap_recs):
    (tmp_path / "dns.jsonl").write_text(
        "\n".join(json.dumps(r) for r in dns_recs) + "\n", encoding="utf-8")
    (tmp_path / "rdap.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rdap_recs) + "\n", encoding="utf-8")


def test_feature_names_and_groups():
    assert N_EVIDENCE_FEATURES == 14
    lo, hi = GROUP_COLUMNS["dns"]; assert EVIDENCE_FEATURE_NAMES[lo:hi][0] == "dns_ok"
    lo, hi = GROUP_COLUMNS["rdap"]; assert EVIDENCE_FEATURE_NAMES[lo] == "rdap_ok"


def test_ok_records_populate_numerics(tmp_path):
    _write_logs(tmp_path, [DNS_OK], [RDAP_OK])
    m = load_evidence_matrix(tmp_path)
    v = m["d.test"]
    dns, rdap = v[:8], v[8:]
    assert dns[0] == 1 and dns[1] == 0 and dns[2] == 0        # ok indicators
    assert dns[3] == 3 and dns[4] == 300 and dns[5] == 1       # numerics
    assert rdap[0] == 1
    # age: 2016-10-02 -> 2026-10-02 = 3653 days (incl. leap days)
    assert 3650 <= rdap[3] <= 3656
    assert 7300 <= rdap[4] <= 7310                              # expiry ~10y
    assert rdap[5] == 2                                          # nameservers


def test_status_indicator_mapping(tmp_path):
    cases = [("ok", (1, 0, 0)), ("not_found", (0, 1, 0)),
             ("timeout", (0, 0, 1)), ("rate_limited", (0, 0, 1)),
             ("error", (0, 0, 1)), (None, (0, 0, 0))]
    for i, (status, expected) in enumerate(cases):
        _write_logs(tmp_path, [{**DNS_OK, "domain": f"d{i}.test", "status": status,
                                "fields": {"a_record_count": 9}}],
                    [{**RDAP_OK, "domain": f"d{i}.test", "status": "not_found",
                      "fields": None}])
        v = load_evidence_matrix(tmp_path)[f"d{i}.test"]
        assert tuple(v[:3]) == expected, status
        if status != "ok":
            assert v[3] == 0.0                     # numerics zero when not ok
        assert v[8] == 0 and v[9] == 1             # rdap not_found indicator


def test_last_record_wins(tmp_path):
    _write_logs(tmp_path, [{**DNS_OK, "status": "rate_limited", "fields": None}, DNS_OK], [])
    v = load_evidence_matrix(tmp_path)["d.test"]
    assert v[0] == 1 and v[3] == 3                 # second (ok) record used


def test_missing_domain_is_zero_state(tmp_path):
    _write_logs(tmp_path, [DNS_OK], [RDAP_OK])
    m = load_evidence_matrix(tmp_path)
    E = evidence_matrix_for_domains(["d.test", "absent.test", "192.0.2.5"], m)
    assert E.shape == (3, 14)
    assert np.all(E[1] == 0) and np.all(E[2] == 0)  # not-acquired state
    assert E[0][0] == 1


def test_mask_groups_zeroes_only_named_group():
    E = np.ones((3, 14))
    M = mask_groups(E, ("rdap",))
    assert np.all(M[:, :8] == 1) and np.all(M[:, 8:] == 0)
    assert np.all(E == 1)                            # original untouched (copy)
    M2 = mask_groups(E, ("dns", "rdap"))
    assert np.all(M2 == 0)
    with pytest.raises(ValueError):
        mask_groups(E, ("tls",))


def test_log_fingerprint_binds_content(tmp_path):
    _write_logs(tmp_path, [DNS_OK], [RDAP_OK])
    fp1 = evidence_log_fingerprint(tmp_path)
    assert fp1 == evidence_log_fingerprint(tmp_path)     # stable
    _write_logs(tmp_path, [DNS_OK], [{**RDAP_OK, "status": "not_found", "fields": None}])
    assert evidence_log_fingerprint(tmp_path) != fp1     # content-bound
