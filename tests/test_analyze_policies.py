import numpy as np
import pandas as pd
import pytest

from model.analyze_policies import (
    _disable,
    _inject_timeouts,
    _rescore,
    _summarize_states,
    budget_analysis,
    ablation_analysis,
    cv_stability,
    failure_analysis,
)


def _ctx():
    E = np.zeros((4, 14))
    E[0, 0] = 1; E[0, 8] = 1
    E[1, 0] = 1; E[1, 9] = 1
    E[2, :] = 0
    E[3, 8] = 1
    st = {"none": np.full(4, 0.30), "dns_only": np.array([0.95, 0.95, 0.30, 0.30]),
          "rdap_only": np.array([0.65, 0.40, 0.30, 0.65]),
          "full": np.array([1.0, 1.0, 0.30, 0.65])}
    y = np.array([1, 1, 0, 0])
    dom = ["a.test", "b.test", None, "c.test"]
    records = {"dns": {d: {"domain": d, "status": "ok", "duration_ms": 100.0}
                       for d in ("a.test", "b.test", "c.test")},
               "rdap": {d: {"domain": d, "status": "ok", "duration_ms": 200.0}
                        for d in ("a.test", "b.test", "c.test")}}
    from model.acquisition import GroupProfile
    prof = {"dns": GroupProfile("dns", 0.95, 500.0, 600.0, 0.05, 0.02, 0.02, 0.90, 10.0),
            "rdap": GroupProfile("rdap", 0.62, 800.0, 900.0, 0.03, 0.01, 0.01, 0.80, 3.0)}
    return st, y, dom, records, prof


def test_disable_removes_group_only():
    st, y, dom, rec, prof = _ctx()
    r = _disable(rec, "dns")
    assert "dns" not in r and "rdap" in r
    assert _disable(rec, "rdap")["dns"] == rec["dns"]


def test_inject_timeouts_deterministic_fraction():
    st, y, dom, rec, prof = _ctx()
    r = _inject_timeouts(rec, "dns", 1 / 3)          # 1 of 3 ok domains
    to = [d for d, v in r["dns"].items() if v["status"] == "timeout"]
    assert len(to) == 1
    r2 = _inject_timeouts(rec, "dns", 1 / 3)         # reproducible
    assert [d for d, v in r2["dns"].items() if v["status"] == "timeout"] == to
    # rate 0 changes nothing
    assert _inject_timeouts(rec, "dns", 0.0) == rec


def test_rescore_neutralizes_named_terms():
    st, y, dom, rec, prof = _ctx()
    r = _rescore(prof, ("latency",))
    assert r["dns"].score == pytest.approx(prof["dns"].mean_abs_dp
                                           * prof["dns"].availability
                                           * prof["dns"].stability, rel=1e-6)
    v_only = _rescore(prof, ("availability", "stability", "latency"))
    assert v_only["dns"].score == pytest.approx(prof["dns"].mean_abs_dp, rel=1e-6)
    # non-scored fields preserved
    assert r["dns"].p95_ms == prof["dns"].p95_ms


def test_summarize_states_reports_four_conditions():
    st, y, dom, rec, prof = _ctx()
    mask = np.array([True, True, True, True])
    s = _summarize_states(st, y, mask, 0.5)
    assert set(s) == {"f1_none", "f1_dns", "f1_rdap", "f1_full"}


def test_budget_analysis_sweeps_configs():
    st, y, dom, rec, prof = _ctx()
    df = budget_analysis(st, y, dom, rec, prof, 0.5, (0.02, 0.90))
    assert len(df) == 12                                   # 2 x (5 budgets + None)
    assert set(df.max_requests) == {1, 2}
    assert {"f1", "mean_requests", "latency_p95_ms"} <= set(df.columns)


def test_failure_analysis_scenarios_and_policies():
    st, y, dom, rec, prof = _ctx()
    df = failure_analysis(st, y, dom, rec, prof, 0.5, (0.02, 0.90))
    assert set(df.policy) == {"B2", "B3", "P1"}
    assert {"baseline", "disable_dns", "disable_rdap",
            "timeouts_dns_20pct"} <= set(df.scenario)
    base = df[df.scenario == "baseline"].set_index("policy")
    dis = df[(df.scenario == "disable_dns") & (df.policy == "B2")].iloc[0]
    assert dis.acquisition_success_rate < base.acquisition_success_rate.loc["B2"]


def test_ablation_analysis_variants():
    st, y, dom, rec, prof = _ctx()
    df = ablation_analysis(st, y, dom, rec, prof, 0.5, (0.02, 0.90))
    assert set(df.variant) == {"full_score", "value_only", "no_availability",
                               "no_stability", "no_latency"}
    assert (df.req_dns >= 0).all() and (df.req_rdap >= 0).all()


def test_cv_stability_runs_and_bounds():
    st, y, dom, rec, prof = _ctx()
    # duplicate rows so folds have enough hosted samples
    st2 = {k: np.tile(v, 6) for k, v in st.items()}
    y2 = np.tile(y, 6); dom2 = dom * 6
    s = cv_stability(st2, y2, dom2, 0.5, k=3)
    assert set(s) == {"dns", "rdap"}
    assert all(0.0 <= v <= 1.0 for v in s.values())
