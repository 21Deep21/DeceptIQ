import json
import sqlite3
from pathlib import Path

import pytest

BUNDLE = Path("model/model.joblib")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(),
                                reason="deployed model bundle missing")


@pytest.fixture(scope="module")
def service():
    from appconfig import get_config
    from services.prediction_service import PredictionService
    return PredictionService(str(BUNDLE), get_config())


@pytest.fixture()
def app(tmp_path, service):
    from app import create_app
    application = create_app(
        overrides={
            "paths": {
                "database_path": str(tmp_path / "scans.db"),
                "alert_log_path": str(tmp_path / "alerts.log"),
                "log_dir": str(tmp_path / "logs"),
            },
            "alerts": {"enabled": True},
        },
        configure_logging=False,
        service=service,
    )
    application.extensions["limiter"].enabled = False
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def test_root_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"html" in r.data.lower()


def test_health_online(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "online"
    assert body["model"] == "lightgbm"
    assert body["threshold"] == pytest.approx(0.7833, abs=1e-3)
    assert body["cache"]["entries"] >= 0


def test_api_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.get_json()
    assert body["model"]["name"] == "lightgbm"
    assert body["threshold"]["deployed"] == pytest.approx(0.7833, abs=1e-3)
    assert body["test_results"]["calibrated_at_threshold"]["recall"] == pytest.approx(0.997, abs=1e-3)
    assert "final" in body["dataset"]


def test_predict_safe(client):
    r = client.post("/predict", json={"url": "https://example.com/"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] == "safe"
    assert body["severity"] == "LOW"
    assert body["probability"] < body["threshold"]
    assert isinstance(body["scan_id"], int) and body["scan_id"] >= 1
    assert body["scanned_at"]
    assert body["enrichment"] is None
    assert body["model"]["probability_is_calibrated"] is True
    assert body["shap"]["increasing"] is not None
    assert body["ioc"]["components"]["host"] == "example.com"


def test_predict_phishing_and_alert_written(client, tmp_path):
    r = client.post("/predict", json={"url": "http://192.0.2.10:8080/login.php"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] == "phishing"
    assert body["severity"] == "CRITICAL"
    alert_path = tmp_path / "alerts.log"
    assert alert_path.exists()
    content = alert_path.read_text()
    assert content.startswith("CEF:0|")
    assert "192.0.2.10" in content
    assert "PHISHING_URL_DETECTED" in content


def test_predict_malformed_url_422(client):
    r = client.post("/predict", json={"url": "not a url"})
    assert r.status_code == 422
    assert r.get_json()["error"]["code"] == "malformed_url"


def test_predict_invalid_bodies_400(client):
    assert client.post("/predict", data="not json",
                       content_type="application/json").status_code == 400
    assert client.post("/predict", json={"wrong": "field"}).status_code == 400
    assert client.post("/predict", json={"url": ""}).status_code == 400
    assert client.post("/predict", json={"url": 123}).status_code == 400


def test_predict_userinfo_password_not_echoed(client):
    r = client.post("/predict", json={"url": "https://user:secretpw@example.com/login"})
    assert r.status_code == 200
    body = r.get_json()
    assert "secretpw" not in json.dumps(body)
    assert body["url"] == "https://user:[REDACTED]@example.com/login"


def test_predict_query_password_not_echoed_or_stored(client, tmp_path):
    r = client.post("/predict",
                    json={"url": "https://example.com/l?password=hunter2&x=1"})
    assert r.status_code == 200
    assert "hunter2" not in json.dumps(r.get_json())
    conn = sqlite3.connect(str(tmp_path / "scans.db"))
    try:
        blob = str(conn.execute("SELECT url, ioc_json FROM scans").fetchall())
    finally:
        conn.close()
    assert "hunter2" not in blob


def test_predict_batch_mixed(client):
    urls = ["https://example.com/", "http://192.0.2.10:8080/login.php",
            "not a url", "https://example.org/"]
    r = client.post("/predict_batch", json={"urls": urls})
    assert r.status_code == 200
    body = r.get_json()
    assert body["summary"] == {"total": 4, "valid": 3, "errors": 1,
                               "phishing": 1, "safe": 2}
    assert [item["status"] for item in body["results"]] == ["ok", "ok", "error", "ok"]
    assert body["results"][2]["error"]["code"] == "malformed_url"
    ok_items = [item for item in body["results"] if item["status"] == "ok"]
    assert all(isinstance(item["scan_id"], int) for item in ok_items)


def test_predict_batch_over_limit_400(client):
    r = client.post("/predict_batch", json={"urls": ["https://example.com/"] * 51})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "batch_limit_exceeded"


def test_predict_batch_invalid_payload_400(client):
    assert client.post("/predict_batch", json={}).status_code == 400
    assert client.post("/predict_batch", json={"urls": []}).status_code == 400
    assert client.post("/predict_batch", json={"urls": "nope"}).status_code == 400


def test_history_records_filters_order(client):
    client.post("/predict", json={"url": "https://example.com/"})               # scan 1
    client.post("/predict", json={"url": "http://192.0.2.10:8080/login.php"})   # scan 2

    body = client.get("/history").get_json()
    assert body["count"] == 2 and body["total_matching"] == 2
    row = body["scans"][0]  # default order desc: latest scan first
    for field in ("scan_id", "scanned_at", "url", "verdict", "probability",
                  "severity", "threshold", "model_version", "top_shap", "ioc"):
        assert field in row
    assert row["scan_id"] == 2 and row["verdict"] == "phishing"
    assert row["top_shap"]["increasing"]

    body = client.get("/history?order=asc").get_json()
    assert body["scans"][0]["scan_id"] == 1
    assert body["scans"][0]["verdict"] == "safe"

    body = client.get("/history?verdict=phishing").get_json()
    assert body["count"] == 1
    assert all(s["verdict"] == "phishing" for s in body["scans"])

    body = client.get("/history?severity=CRITICAL").get_json()
    assert body["count"] == 1

    body = client.get("/history?q=192.0.2").get_json()
    assert body["count"] == 1

    body = client.get("/history?limit=1").get_json()
    assert body["count"] == 1 and body["total_matching"] == 2


def test_history_invalid_filter_400(client):
    assert client.get("/history?verdict=bogus").status_code == 400
    assert client.get("/history?severity=EXTREME").status_code == 400
    assert client.get("/history?order=sideways").status_code == 400


def test_rate_limit_returns_429(tmp_path, service):
    from app import create_app
    application = create_app(
        overrides={
            "paths": {"database_path": str(tmp_path / "scans.db"),
                      "alert_log_path": str(tmp_path / "alerts.log"),
                      "log_dir": str(tmp_path / "logs")},
            "api": {"rate_limit": "2 per minute"},
        },
        configure_logging=False,
        service=service,
    )
    c = application.test_client()
    assert c.post("/predict", json={"url": "https://example.com/"}).status_code == 200
    assert c.post("/predict", json={"url": "https://example.org/"}).status_code == 200
    r = c.post("/predict", json={"url": "https://example.net/"})
    assert r.status_code == 429
    assert r.get_json()["error"]["code"] == "rate_limited"
    # exempt endpoints remain reachable after the limit trips:
    assert c.get("/health").status_code == 200
