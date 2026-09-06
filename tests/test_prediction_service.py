import os
from pathlib import Path

import pytest

from features.url_utils import MalformedURLError
from services.prediction_service import BoundedLRU, PredictionService

BUNDLE = Path("model/model.joblib")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(),
                                reason="deployed model bundle missing")


@pytest.fixture(scope="module")
def service():
    from appconfig import get_config
    return PredictionService(str(BUNDLE), get_config())


def test_bounded_lru_eviction_order_and_stats():
    cache = BoundedLRU(maxsize=3)
    for i in range(3):
        cache.put(f"k{i}", f"v{i}")
    assert cache.get("k0") == "v0"          # k0 becomes most recently used
    cache.put("k3", "v3")                   # evicts k1 (least recently used)
    assert cache.get("k1") is None
    assert cache.get("k0") == "v0"
    assert cache.get("k2") == "v2"
    assert cache.get("k3") == "v3"
    stats = cache.stats()
    assert stats["entries"] == 3 and stats["max_entries"] == 3
    assert stats["hits"] == 4 and stats["misses"] == 1


def test_severity_bands(service):
    assert service.severity_of(0.0) == "LOW"
    assert service.severity_of(0.2499) == "LOW"
    assert service.severity_of(0.25) == "MEDIUM"
    assert service.severity_of(0.5999) == "MEDIUM"
    assert service.severity_of(0.60) == "HIGH"
    assert service.severity_of(0.8999) == "HIGH"
    assert service.severity_of(0.90) == "CRITICAL"
    assert service.severity_of(1.0) == "CRITICAL"


def test_predict_safe_url_fields(service):
    r = service.predict("https://example.com/")
    assert r["verdict"] == "safe"
    assert 0.0 <= r["probability"] < r["threshold"]
    assert r["threshold"] == pytest.approx(service.threshold, abs=1e-9)
    assert r["model"]["name"] == service.model_name
    assert r["model"]["probability_is_calibrated"] is True
    assert r["shap"]["increasing"] and r["shap"]["decreasing"]
    assert r["ioc"]["components"]["registrable_domain"] == "example.com"
    assert r["enrichment"] is None


def test_predict_phishing_url_fields(service):
    r = service.predict("http://192.0.2.10:8080/login.php")
    assert r["verdict"] == "phishing"
    assert r["probability"] >= r["threshold"]
    assert r["severity"] in ("HIGH", "CRITICAL")
    assert r["ioc"]["components"]["is_ip"] is True


def test_predict_is_cached_by_normalized_url(service):
    r1 = service.predict("https://cache-probe.example.org/")
    r2 = service.predict("  https://cache-probe.example.org/  ")  # same key
    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is True
    assert r1["probability"] == r2["probability"]


def test_predict_malformed_raises(service):
    with pytest.raises(MalformedURLError):
        service.predict("not a url")


def test_predict_batch_mixed_statuses(service):
    results = service.predict_batch(
        ["https://example.com/", "not a url", "http://192.0.2.10:8080/login.php", 123]
    )
    assert [r["status"] for r in results] == ["ok", "error", "ok", "error"]
    assert results[1]["error"]["code"] == "malformed_url"
    assert results[3]["error"]["code"] == "invalid_input"
    assert results[2]["verdict"] == "phishing"


def test_predict_batch_populates_cache(service):
    service.predict_batch(["https://batch-cache-probe.example.org/x"])
    r = service.predict("https://batch-cache-probe.example.org/x")
    assert r["cache_hit"] is True


@pytest.mark.skipif(os.environ.get("RUN_NETWORK_TESTS") != "1",
                    reason="network test - set RUN_NETWORK_TESTS=1 to enable")
def test_predict_enrichment_example_com(service):
    r = service.predict("https://example.com/", include_enrichment=True)
    e = r["enrichment"]
    assert e["cached"] is False
    assert e["whois"]["available"] is True
    assert e["whois"]["registrar"]
    assert e["dns"]["a_record_count"] is not None
    assert e["dns"]["a_record_count"] >= 1
