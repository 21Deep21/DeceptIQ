"""Option B: confidence-honest verdict display (FLAGGED FOR REVIEW).

Display-level only: the API verdict string, alerting, and persistence are
unchanged. The console shows an amber FLAGGED FOR REVIEW badge for alerted
URLs whose probability is below the configurable review threshold.
"""
from pathlib import Path

import pytest

BUNDLE = Path("model/model.joblib")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="v1 bundle missing")


@pytest.fixture(scope="module")
def service():
    from appconfig import get_config
    from services.prediction_service import PredictionService
    return PredictionService(str(BUNDLE), get_config())


def _client(tmp_path, service, review=0.60):
    from app import create_app
    app = create_app(
        overrides={"paths": {"database_path": str(tmp_path / "s.db"),
                             "alert_log_path": str(tmp_path / "a.log"),
                             "log_dir": str(tmp_path / "logs")},
                   "review_threshold": review},
        configure_logging=False, service=service)
    app.extensions["limiter"].enabled = False
    return app.test_client()


def test_health_exposes_review_threshold(tmp_path, service):
    body = _client(tmp_path, service).get("/health").get_json()
    assert body["review_threshold"] == 0.60


def test_api_verdict_string_unchanged(tmp_path, service):
    # The API contract keeps 'safe'/'phishing'; FLAGGED FOR REVIEW is a
    # frontend display state computed from probability + review_threshold.
    body = _client(tmp_path, service).post(
        "/predict", json={"url": "https://example.com/"}).get_json()
    assert body["verdict"] in ("safe", "phishing")


def test_frontend_implements_review_display(tmp_path, service):
    c = _client(tmp_path, service)
    js = c.get("/static/js/script.js").get_data(as_text=True)
    assert "FLAGGED FOR REVIEW" in js
    assert "REVIEW_THRESHOLD" in js
    assert "verdictBadge(res.verdict, res.probability)" in js
    assert "verdictBadge(r.verdict, r.probability)" in js
    css = c.get("/static/css/style.css").get_data(as_text=True)
    assert ".v-review" in css


def test_config_carries_review_threshold():
    from appconfig import get_config
    assert float(get_config().get("review_threshold", 0.60)) == 0.60
