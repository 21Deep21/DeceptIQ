import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

BUNDLE = Path("model_v2/model.joblib")
POLICY = Path("model_v2/policy_evaluation.json")
pytestmark = pytest.mark.skipif(
    not (BUNDLE.exists() and POLICY.exists()),
    reason="model_v2 bundle or policy evaluation missing")

# Measured URL-only p0 = 0.2816 -> inside band (0.02, 0.75)
UCL = "http://www.fsf.org/licensing/education"


def _svc(cfg_mods=None):
    from appconfig import get_config
    from services.acquisition_service import AcquisitionService
    cfg = copy.deepcopy(get_config())
    for keys, val in (cfg_mods or []):
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return AcquisitionService(str(BUNDLE), str(POLICY),
                              "/nonexistent/calibrators.joblib", cfg)


def _dns_ok(domain, timeout=5):
    return SimpleNamespace(a_record_count=2, a_record_ttl=300, has_mx=True,
                           ns_count=2, ns_diversity=1, error=None)


def _rdap_ok(domain, timeout=8):
    return SimpleNamespace(status_code="ok", registrar="Example Registrar",
                           creation_date="2020-01-01T00:00:00Z",
                           expiration_date="2030-01-01T00:00:00Z",
                           status=["active"], nameservers=["a.ns.test"],
                           error=None, raw_json="{}")


def _rdap_dead(domain, timeout=8):
    return SimpleNamespace(status_code="not_found", registrar=None,
                           creation_date=None, expiration_date=None,
                           status=[], nameservers=[], error="no record",
                           raw_json=None)


def _fail(*a, **k):
    raise AssertionError("live query must not be called in this test")


def test_confident_ip_url_never_escalates(monkeypatch):
    import services.acquisition_service as mod
    monkeypatch.setattr(mod, "query_dns", _fail)
    monkeypatch.setattr(mod, "query_rdap", _fail)
    r = _svc().analyze("http://192.0.2.10:8080/login.php")
    assert r["acquisition"]["escalated"] is False
    assert r["acquisition"]["requests"] == []
    assert r["acquisition"]["final_condition"] == "none"
    assert r["verdict"] == "phishing" and r["threshold"] == pytest.approx(0.0392, abs=1e-3)


def test_uncertain_hosted_escalates_dns_first(monkeypatch):
    import services.acquisition_service as mod
    monkeypatch.setattr(mod, "query_dns", _dns_ok)
    monkeypatch.setattr(mod, "query_rdap", _rdap_ok)
    r = _svc().analyze(UCL)          # none-condition p ~0.42 -> inside (0.02, 0.75)
    a = r["acquisition"]
    assert a["escalated"] is True
    assert a["requests"][0]["group"] == "dns"
    assert 1 <= len(a["requests"]) <= 2
    assert a["final_condition"] in ("dns_only", "full")
    assert r["model"]["name"] == "random_forest_v2_fusion"
    assert r["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "shap" in r and "evidence_total" in r["shap"]
    assert "ioc" in r


def test_rdap_failure_recorded_and_verdict_still_returned(monkeypatch):
    import services.acquisition_service as mod
    monkeypatch.setattr(mod, "query_dns", _dns_ok)   # dns may follow rdap
    monkeypatch.setattr(mod, "query_rdap", _rdap_dead)
    svc = _svc()
    svc._order = ["rdap", "dns"]                      # force rdap first
    r = svc.analyze(UCL)
    a = r["acquisition"]
    assert a["requests"] and a["requests"][0]["group"] == "rdap"
    assert a["requests"][0]["status"] == "not_found"  # failure is a recorded state
    assert r["verdict"] in ("safe", "phishing")       # failure-immune


def test_latency_budget_blocks_acquisition(monkeypatch):
    import services.acquisition_service as mod
    monkeypatch.setattr(mod, "query_dns", _dns_ok)
    monkeypatch.setattr(mod, "query_rdap", _rdap_ok)
    svc = _svc(cfg_mods=[(["acquisition", "latency_budget_ms"], 1.0)])
    r = svc.analyze(UCL)               # dns p95 5688ms >> 1ms budget
    assert r["acquisition"]["requests"] == []
    assert r["acquisition"]["final_condition"] == "none"


def test_severity_bands_match_v1():
    svc = _svc()
    assert svc.severity_of(0.1) == "LOW" and svc.severity_of(0.3) == "MEDIUM"
    assert svc.severity_of(0.7) == "HIGH" and svc.severity_of(0.95) == "CRITICAL"
