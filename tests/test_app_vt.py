from pathlib import Path

import pytest

BUNDLE = Path("model/model.joblib")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(),
                                reason="deployed model bundle missing")


class FakeVTClient:
    def __init__(self, payload):
        self.payload = payload
        self.lookup_calls = []

    def status(self):
        return {"status": "enabled", "reason": "fake"}

    def lookup(self, url):
        self.lookup_calls.append(url)
        return dict(self.payload)


@pytest.fixture(scope="module")
def service():
    from appconfig import get_config
    from services.prediction_service import PredictionService
    return PredictionService(str(BUNDLE), get_config())


def _app(tmp_path, service, vt_client=None):
    from app import create_app
    return create_app(
        overrides={
            "paths": {"database_path": str(tmp_path / "scans.db"),
                      "alert_log_path": str(tmp_path / "alerts.log"),
                      "log_dir": str(tmp_path / "logs")},
        },
        configure_logging=False,
        service=service,
        vt_client=vt_client,
    )


def test_predict_includes_virustotal_when_requested(tmp_path, service):
    payload = {"status": "ok", "note": "fake VT result",
               "engines_detected": 3, "engines_total": 90}
    fake = FakeVTClient(payload)
    client = _app(tmp_path, service, vt_client=fake).test_client()
    r = client.post("/predict", json={"url": "https://example.com/",
                                      "include_external": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["virustotal"] == payload
    assert body["verdict"] in ("safe", "phishing")  # local verdict independent
    assert fake.lookup_calls == ["https://example.com/"]


def test_predict_omits_virustotal_by_default(tmp_path, service):
    fake = FakeVTClient({"status": "ok"})
    client = _app(tmp_path, service, vt_client=fake).test_client()
    r = client.post("/predict", json={"url": "https://example.com/"})
    assert r.status_code == 200
    assert "virustotal" not in r.get_json()
    assert fake.lookup_calls == []


def test_predict_virustotal_disabled_without_key(tmp_path, service, monkeypatch):
    import app as app_module
    monkeypatch.delenv("VT_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "load_env_file", lambda *a, **k: {})
    client = _app(tmp_path, service).test_client()  # real client, no key
    r = client.post("/predict", json={"url": "https://example.com/",
                                      "include_external": True})
    assert r.status_code == 200
    vt = r.get_json()["virustotal"]
    assert vt["status"] == "disabled"
    assert "VT_API_KEY" in vt["reason"]


def test_health_reports_virustotal_status(tmp_path, service):
    fake = FakeVTClient({"status": "ok"})
    client = _app(tmp_path, service, vt_client=fake).test_client()
    body = client.get("/health").get_json()
    assert body["virustotal"] == {"status": "enabled", "reason": "fake"}


def test_batch_never_calls_virustotal(tmp_path, service):
    fake = FakeVTClient({"status": "ok"})
    client = _app(tmp_path, service, vt_client=fake).test_client()
    r = client.post("/predict_batch", json={"urls": ["https://example.com/"]})
    assert r.status_code == 200
    assert fake.lookup_calls == []  # external intel is single-URL only (documented)
