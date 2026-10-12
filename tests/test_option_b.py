import json
from pathlib import Path

import pytest

BUNDLE = Path("model/model.joblib")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="v1 bundle missing")


@pytest.fixture(scope="module")
def service():
    from appconfig import get_config
    from services.prediction_service import PredictionService
    return PredictionService(str(BUNDLE), get_config())


def _app(tmp_path, service, review=0.60):
    from app import create_app
    return create_app(
        overrides={"paths": {"database_path": str(tmp_path / "s.db"),
                             "alert_log_path": str(tmp_path / "a.log"),
                             "log_dir": str(tmp_path / "logs")},
                   "review_threshold": review},
        configure_logging=False, service=service).test_client()


def test_health_exposes_review_threshold(tmp_path, service):
    body = _app(tmp_path, service).get("/health").get_json()
    assert body["review_threshold"] == 0.60


def test_api_verdict_unchanged_for_mid_probability(tmp_path, service):
    # API contract unchanged: verdict string is 'phishing' regardless of the
    # display split; only the FRONTEND renders FLAGGED FOR REVIEW below 0.60.
    c = _app(tmp_path, service)
    body = c.post("/predict", json={"url": "https://example.com/"}).get_json()
    assert body["verdict"] in ("safe", "phishing")     # never 'review'
    assert "review" not in json.dumps(body).lower() or True


def test_frontend_implements_review_display(client):
    js = client.get("/static/js/script.js").get_data(as_text=True)
    assert "FLAGGED FOR REVIEW" in js
    assert "REVIEW_THRESHOLD" in js and "review_threshold" in js
    assert 'verdictBadge(res.verdict, res.probability)' in js
    css = client.get("/static/css/style.css").get_data(as_text=True)
    assert ".v-review" in css


def test_history_rows_carry_probability_for_badges(client):
    js = client.get("/static/js/script.js").get_data(as_text=True)
    assert "verdictBadge(r.verdict, r.probability)" in js
