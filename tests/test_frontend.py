import re
from pathlib import Path

import pytest

BUNDLE = Path("model/model.joblib")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="deployed model bundle missing")


@pytest.fixture(scope="module")
def service():
    from appconfig import get_config
    from services.prediction_service import PredictionService
    return PredictionService(str(BUNDLE), get_config())


@pytest.fixture()
def client(tmp_path, service):
    from app import create_app
    app = create_app(
        overrides={"paths": {"database_path": str(tmp_path / "scans.db"),
                             "alert_log_path": str(tmp_path / "alerts.log"),
                             "log_dir": str(tmp_path / "logs")}},
        configure_logging=False, service=service)
    app.extensions["limiter"].enabled = False
    return app.test_client()


def test_index_page_renders_console(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.data.decode("utf-8")
    for marker in ("URL INVESTIGATION", "VERDICT", "URL DECONSTRUCTION",
                   "RISK FACTORS", "INDICATORS OF COMPROMISE", "RECENT SCANS",
                   "BATCH URL SCAN", "static/css/style.css", "static/js/script.js"):
        assert marker in html, f"missing console marker: {marker}"


def test_static_assets_served(client):
    css = client.get("/static/css/style.css")
    assert css.status_code == 200
    assert b"--bg:" in css.data  # design tokens present
    js = client.get("/static/js/script.js")
    assert js.status_code == 200
    assert b"/predict_batch" in js.data
    assert b"function esc(" in js.data  # XSS escaping helper present


def test_frontend_fetch_endpoints_exist(client):
    """Every endpoint the frontend calls must be a registered Flask route."""
    js = Path("static/js/script.js").read_text(encoding="utf-8")
    referenced = set(re.findall(
        r'"/(predict|predict_batch|history|health|api/model-info)[?"]', js))
    assert referenced, "no endpoints found in script.js"
    rules = {rule.rule for rule in client.application.url_map.iter_rules()}
    for endpoint in referenced:
        assert "/" + endpoint in rules, f"frontend calls unknown endpoint /{endpoint}"
