import json

import numpy as np
import pytest

from model.acquisition import (
    GroupProfile,
    build_profiles,
    load_records,
    precompute_states,
    replay_policy,
    select_band,
    summarize,
)

BAND = (0.02, 0.90)
THR = 0.5


def _pf(E):
    # p depends on evidence columns: dns_ok (0), rdap_ok (8), rdap_not_found (9)
    return np.clip(0.30 + 0.65 * E[:, 0] + 0.35 * E[:, 8] + 0.10 * E[:, 9], 0, 1)


def _setup():
    E = np.zeros((4, 14))
    E[0, 0] = 1; E[0, 8] = 1                      # a.test: dns ok, rdap ok
    E[1, 0] = 1; E[1, 9] = 1                      # b.test: dns ok, rdap not_found
    E[2, :] = 0                                    # IP host: no evidence
    E[3, 8] = 1                                    # c.test: rdap ok, dns absent
    states = precompute_states(_pf, E)
    urls = ["u0", "u1", "u2", "u3"]
    y = [1, 1, 0, 0]
    domains = ["a.test", "b.test", None, "c.test"]
    records = {
        "dns": {"a.test": {"domain": "a.test", "status": "ok", "duration_ms": 100.0},
                "b.test": {"domain": "b.test", "status": "ok", "duration_ms": 110.0},
                "c.test": {"domain": "c.test", "status": "timeout", "duration_ms": 5000.0}},
        "rdap": {"a.test": {"domain": "a.test", "status": "not_found", "duration_ms": 200.0},
                 "b.test": {"domain": "b.test", "status": "ok", "duration_ms": 210.0},
                 "c.test": {"domain": "c.test", "status": "ok", "duration_ms": 220.0}},
    }
    profiles = {
        "dns": GroupProfile("dns", 0.95, 500.0, 600.0, 0.05, 0.02, 0.02, 0.90, 10.0),
        "rdap": GroupProfile("rdap", 0.62, 800.0, 900.0, 0.03, 0.01, 0.01, 0.80, 3.0),
    }
    return states, urls, y, domains, records, profiles


def test_precompute_states_respects_masking():
    states, *_ = _setup()
    assert np.allclose(states["none"], [0.30, 0.30, 0.30, 0.30])
    assert np.allclose(states["dns_only"], [0.95, 0.95, 0.30, 0.30])
    assert np.allclose(states["rdap_only"], [0.65, 0.40, 0.30, 0.65])
    assert states["full"][0] == 1.0 and states["full"][2] == 0.30


def test_b1_never_requests():
    st, urls, y, dom, rec, prof = _setup()
    out = replay_policy("B1", urls, y, dom, st, rec, prof, THR, BAND)
    assert all(len(r.requests) == 0 and not r.escalated for r in out)
    assert all(r.final_condition == "none" and r.verdict == "safe" for r in out)


def test_b2_acquires_all_for_domain_hosted_only():
    st, urls, y, dom, rec, prof = _setup()
    out = replay_policy("B2", urls, y, dom, st, rec, prof, THR, BAND)
    assert len(out[0].requests) == 2 and len(out[2].requests) == 0
    assert out[0].total_latency_ms == pytest.approx(300.0)   # 100 + 200 recorded
    assert out[0].final_condition == "full" and out[0].verdict == "phishing"


def test_b3_gates_on_uncertainty():
    st, urls, y, dom, rec, prof = _setup()
    out = replay_policy("B3", urls, y, dom, st, rec, prof, THR, BAND)
    assert out[0].escalated and out[2].escalated is False    # IP never acquires
    # tight band: p0=0.30 is below low=0.31 -> confident -> nobody escalates
    out2 = replay_policy("B3", urls, y, dom, st, rec, prof, THR, (0.31, 0.90))
    assert all(not r.escalated for r in out2)


def test_p1_ranked_order_and_early_stop():
    st, urls, y, dom, rec, prof = _setup()
    out = replay_policy("P1", urls, y, dom, st, rec, prof, THR, BAND)
    # dns (score 10) first; after dns p=0.95 > 0.90 -> confident -> stop at 1
    assert [r.requests[0]["group"] for r in out[:2]] == ["dns", "dns"]
    assert all(len(r.requests) == 1 for r in out[:2])
    assert out[0].final_condition == "dns_only"
    # row 3: dns does not move p (0.30, still uncertain) -> rdap next -> 0.65
    assert [q["group"] for q in out[3].requests] == ["dns", "rdap"]
    assert out[3].final_condition == "full"


def test_p1_respects_budgets():
    st, urls, y, dom, rec, prof = _setup()
    # latency budget 400 < dns p95 500 -> nothing acquired
    out = replay_policy("P1", urls, y, dom, st, rec, prof, THR, BAND,
                        latency_budget_ms=400.0)
    assert all(len(r.requests) == 0 for r in out)
    # budget 600 fits dns (500) but not rdap (800) after charging 100
    out = replay_policy("P1", urls, y, dom, st, rec, prof, THR, BAND,
                        latency_budget_ms=600.0)
    assert len(out[3].requests) == 1
    # request budget 1: only dns even when still uncertain (row 3)
    out = replay_policy("P1", urls, y, dom, st, rec, prof, THR, BAND,
                        max_requests=1)
    assert len(out[3].requests) == 1 and out[3].final_condition == "dns_only"


def test_p1_min_group_score_skips_weak_group():
    st, urls, y, dom, rec, prof = _setup()
    out = replay_policy("P1", urls, y, dom, st, rec, prof, THR, BAND,
                        min_group_score=5.0)
    assert [q["group"] for q in out[3].requests] == ["dns"]    # rdap (3.0) skipped


def test_missing_record_fallback():
    st, urls, y, dom, rec, prof = _setup()
    out = replay_policy("B2", ["u"], [1], ["zz.test"], st, rec, prof, THR, BAND)
    q = out[0].requests
    assert all(x["status"] == "error" for x in q)
    assert q[0]["duration_ms"] == pytest.approx(500.0)   # profile p95 fallback
    assert q[1]["duration_ms"] == pytest.approx(800.0)


def test_summarize_operational_math():
    from model.acquisition import DecisionRecord
    r1 = DecisionRecord("u1", 1, "a.test", True, "phishing", 0.9, "full",
                        [{"group": "dns", "status": "ok", "duration_ms": 100.0}], 100.0)
    r2 = DecisionRecord("u2", 0, None, False, "safe", 0.1, "none", [], 0.0)
    s = summarize([r1, r2], THR)
    assert s["f1"] == 1.0 and s["fpr"] == 0.0
    assert s["mean_requests"] == 0.5 and s["pct_escalated"] == 0.5
    assert s["acquisition_success_rate"] == 1.0


def test_load_records_last_wins(tmp_path):
    (tmp_path / "dns.jsonl").write_text(
        json.dumps({"domain": "a.test", "status": "rate_limited", "duration_ms": 1}) + "\n"
        + json.dumps({"domain": "a.test", "status": "ok", "duration_ms": 2}) + "\n",
        encoding="utf-8")
    recs = load_records(tmp_path)
    assert recs["dns"]["a.test"]["status"] == "ok"
    assert recs["rdap"] == {}                     # missing file tolerated


def test_build_profiles_measures_records():
    st, urls, y, dom, rec, prof_unused = _setup()
    hosted = [d is not None for d in dom]
    recs = {"dns": {"x.test": {"status": "ok", "duration_ms": 100.0},
                    "y.test": {"status": "timeout", "duration_ms": 400.0}},
            "rdap": {"x.test": {"status": "ok", "duration_ms": 300.0}}}
    p = build_profiles(st, y, hosted, st, y, hosted, recs, THR)
    assert p["dns"].availability == pytest.approx(0.5)
    assert p["dns"].p95_ms == pytest.approx(400.0)   # ceil index: n=2 -> max
    assert p["rdap"].availability == pytest.approx(1.0)
    assert 0.0 <= p["dns"].stability <= 1.0
    assert p["dns"].score > 0 and p["rdap"].score > 0


def test_select_band_runs_and_returns_tuple():
    st, urls, y, dom, rec, prof = _setup()
    band, s = select_band(st, y, dom, rec, prof, THR,
                          grid=[(0.02, 0.90), (0.31, 0.90)])
    assert band in ((0.02, 0.90), (0.31, 0.90))
    assert "f1" in s and "mean_requests" in s
