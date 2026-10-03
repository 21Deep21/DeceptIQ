import json
from pathlib import Path

import pytest

BUNDLE = Path("model/model.joblib")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(),
                                reason="deployed v1 model bundle missing")


class FakeAcq:
    def __init__(self):
        self.calls = []

    def analyze(self, url):
        self.calls.append(url)
        return {"url": url, "cache_hit": False, "probability": 0.5, "verdict": "phishing",
                "severity": "MEDIUM", "threshold": 0.0392,
                "model": {"name": "random_forest_v2_fusion", "version": "v2@t"},
                "shap": {"backend": "shap", "increasing": [], "decreasing": []},
                "ioc": {"components": {}, "flags": []},
                "acquisition": {"policy": "P1", "band": [0.02, 0.75],
                                "initial_probability": 0.4, "escalated": True,
                                "requests": [{"group": "dns", "status": "ok",
                                              "duration_ms": 120.0}],
                                "final_condition": "dns_only",
                                "final_probability": 0.5,
                                "total_latency_ms": 120.0, "note": "fake"}}


@pytest.fixture(scope="module")
def service():
    from appconfig import get_config
    from services.prediction_service import PredictionService
    return PredictionService(str(BUNDLE), get_config())


def _app(tmp_path, service, acq=None, model_v2_path=None):
    from app import create_app
    overrides = {"paths": {"database_path": str(tmp_path / "scans.db"),
                           "alert_log_path": str(tmp_path / "alerts.log"),
                           "log_dir": str(tmp_path / "logs")}}
    if model_v2_path:
        overrides["model_v2"] = {"model_path": model_v2_path,
                                 "dir": str(tmp_path / "mv2")}
    return create_app(overrides=overrides, configure_logging=False,
                      service=service, acq_service=acq)


def test_cascade_predict_labeled_v2_and_persisted(tmp_path, service):
    fake = FakeAcq()
    c = _app(tmp_path, service, acq=fake).test_client()
    r = c.post("/predict", json={"url": "https://example.com/",
                                 "include_evidence": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["model"]["name"] == "random_forest_v2_fusion"
    assert body["acquisition"]["policy"] == "P1"
    assert body["verdict"] == "phishing"
    assert fake.calls == ["https://example.com/"]
    hist = c.get("/history").get_json()["scans"][0]
    assert hist["model_version"] == "v2@t"
    assert hist["threshold"] == pytest.approx(0.0392, abs=1e-9)
    assert (tmp_path / "alerts.log").exists()          # phishing -> alert


def test_default_predict_unchanged_v1(tmp_path, service):
    fake = FakeAcq()
    c = _app(tmp_path, service, acq=fake).test_client()
    body = c.post("/predict", json={"url": "https://example.com/"}).get_json()
    assert body["model"]["name"] == "random_forest"    # v1 untouched
    assert "acquisition" not in body
    assert fake.calls == []


def test_cascade_unavailable_returns_503(tmp_path, service):
    c = _app(tmp_path, service, model_v2_path="/nonexistent/bundle.joblib").test_client()
    r = c.post("/predict", json={"url": "https://example.com/",
                                 "include_evidence": True})
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "cascade_unavailable"


def test_health_reports_cascade(tmp_path, service):
    c = _app(tmp_path, service, acq=FakeAcq()).test_client()
    assert c.get("/health").get_json()["cascade"] == "available"
