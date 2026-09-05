from services.scan_store import ScanStore


def test_record_and_list_roundtrip(tmp_path):
    store = ScanStore(str(tmp_path / "scans.db"))
    scan_id, scanned_at = store.record(
        url="https://example.com/login", verdict="phishing", probability=0.9123,
        severity="CRITICAL", threshold=0.7833, model_version="lightgbm@2026",
        shap_backend="shap",
        top_shap={"increasing": [{"feature": "has_ip_hostname", "contribution": 0.26}],
                  "decreasing": []},
        ioc={"components": {"host": "example.com"}, "flags": [{"flag": "ip_host"}]},
    )
    assert scan_id == 1 and scanned_at
    rows, total = store.list()
    assert total == 1 and len(rows) == 1
    row = rows[0]
    assert row["scan_id"] == 1
    assert row["url"] == "https://example.com/login"
    assert row["verdict"] == "phishing"
    assert row["probability"] == 0.9123
    assert row["severity"] == "CRITICAL"
    assert row["threshold"] == 0.7833
    assert row["top_shap"]["increasing"][0]["feature"] == "has_ip_hostname"
    assert row["ioc"]["components"]["host"] == "example.com"


def test_filters_search_order_and_like_escaping(tmp_path):
    store = ScanStore(str(tmp_path / "scans.db"))
    for url, verdict in (
        ("https://alpha.example.com/", "safe"),
        ("http://192.0.2.10:8080/login.php", "phishing"),
        ("https://example.org/100%promo", "safe"),   # literal % in stored text
    ):
        store.record(url=url, verdict=verdict, probability=0.5, severity="MEDIUM",
                     threshold=0.78, model_version="m", shap_backend="shap",
                     top_shap={}, ioc={})
    rows, total = store.list(verdict="phishing")
    assert total == 1 and rows[0]["url"].startswith("http://192.0.2.10")
    rows, total = store.list(q="192.0.2")
    assert total == 1
    rows, total = store.list(q="100%")   # % treated literally, not as wildcard
    assert total == 1 and rows[0]["url"].endswith("100%promo")
    rows, total = store.list(order="asc")
    assert rows[0]["url"] == "https://alpha.example.com/"
    rows, total = store.list(order="desc")
    assert rows[0]["url"].endswith("100%promo")      # highest id first
    rows, total = store.list(limit=2)
    assert len(rows) == 2 and total == 3
