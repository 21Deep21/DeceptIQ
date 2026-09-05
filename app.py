"""Flask application - SOC-style phishing URL analysis console backend.

Routes:
  GET  /                investigation console page (UI arrives in Phase 7)
  GET  /health          service status (rate-limit exempt)
  GET  /api/model-info  model/threshold/dataset metadata (rate-limit exempt)
  POST /predict         analyze one URL
  POST /predict_batch   analyze up to api.batch_limit URLs (default 50)
  GET  /history         scan history (limit/verdict/severity/q/order)

Run:  python app.py    (development server; see config.yaml `server`)

Security posture:
  * URLs submitted by users are sanitized (userinfo passwords and
    credential-like query values redacted) before being echoed,
    logged, or stored;
  * rate limiting per client IP (default 30/minute, configurable);
  * no secrets in code/config - environment variables only;
  * the analyzed URL is NEVER visited.
"""

from __future__ import annotations

import copy
import json
import logging
import pathlib
import time
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from appconfig import get_config, load_env_file, setup_app_logging
from features.ioc_extraction import sanitize_url_for_logging
from features.url_utils import MalformedURLError, normalize_url
from services.alert_logger import AlertLogger
from services.prediction_service import PredictionService
from services.scan_store import ScanStore, utc_now_iso
from services.virustotal_service import VirusTotalClient

logger = logging.getLogger("app")

VALID_VERDICTS = ("phishing", "safe")
VALID_SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_json_file(path: str) -> Optional[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def create_app(config_path: Optional[str] = None,
               overrides: Optional[Dict[str, Any]] = None,
               configure_logging: bool = True,
               service: Optional[PredictionService] = None,
               vt_client: Optional[VirusTotalClient] = None) -> Flask:
    """Application factory.

    `service` lets tests share one PredictionService across app
    instances; production leaves it None (loaded from the bundle).
    """
    cfg = get_config(config_path)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    if configure_logging:
        setup_app_logging(cfg)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(
        cfg.get("api", {}).get("request_body_max_bytes", 65536))
    try:
        app.json.sort_keys = False  # stable field order (Flask >= 2.3)
    except Exception:
        app.config["JSON_SORT_KEYS"] = False

    if service is None:
        bundle_path = str(cfg["model"]["model_path"])
        if not pathlib.Path(bundle_path).exists():
            raise RuntimeError(
                f"model bundle not found: {bundle_path} - run Phase 3 first "
                f"(python -m model.finalize)"
            )
        service = PredictionService(bundle_path, cfg)
    store = ScanStore(str(cfg["paths"]["database_path"]))
    alerts = AlertLogger(cfg)
    load_env_file()  # optional .env secrets -> os.environ (values never logged)
    vt = vt_client or VirusTotalClient(cfg)

    api_cfg = cfg.get("api", {})
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[str(api_cfg.get("rate_limit", "30 per minute"))],
    )
    app.extensions["limiter"] = limiter
    app.extensions["prediction_service"] = service
    app.extensions["scan_store"] = store

    batch_limit = int(api_cfg.get("batch_limit", 50))
    history_default_limit = int(api_cfg.get("history_default_limit", 50))
    history_max_limit = int(api_cfg.get("history_max_limit", 500))
    history_cfg = cfg.get("history", {})
    top_shap_inc = int(history_cfg.get("top_shap_increasing", 5))
    top_shap_dec = int(history_cfg.get("top_shap_decreasing", 3))
    enrichment_default = bool(cfg.get("enrichment", {}).get("enabled_by_default", False))
    external_default = bool(cfg.get("enrichment", {}).get("include_external_by_default", False))

    # ------------------------------------------------------------ helpers

    def _record_scan(result: Dict[str, Any]) -> Tuple[int, str]:
        shap = result.get("shap") or {}
        ioc = result.get("ioc") or {}
        return store.record(
            url=result["url"],
            verdict=result["verdict"],
            probability=result["probability"],
            severity=result["severity"],
            threshold=service.threshold,
            model_version=service.model_version,
            shap_backend=service.explainer.backend,
            top_shap={
                "increasing": (shap.get("increasing") or [])[:top_shap_inc],
                "decreasing": (shap.get("decreasing") or [])[:top_shap_dec],
            },
            ioc={"components": ioc.get("components"), "flags": ioc.get("flags")},
        )

    def _maybe_alert(result: Dict[str, Any], scan_id: int) -> None:
        if result["verdict"] == "phishing":
            alerts.alert(
                url=result["url"],
                probability=result["probability"],
                severity=result["severity"],
                threshold=service.threshold,
                model=service.model_version,
                client_ip=request.remote_addr or "-",
                scan_id=scan_id,
            )

    # ------------------------------------------------------------ routes

    @app.get("/")
    @limiter.exempt
    def index():
        return render_template("index.html")

    @app.get("/health")
    @limiter.exempt
    def health():
        return jsonify({
            "status": "online",
            "virustotal": vt.status(),
            "model": service.model_name,
            "model_version": service.model_version,
            "threshold": service.threshold,
            "shap_backend": service.explainer.backend_label,
            "cache": service.cache_stats,
            "time": utc_now_iso(),
        })

    @app.get("/api/model-info")
    @limiter.exempt
    def model_info():
        meta = _load_json_file(str(cfg["model"]["metadata_path"])) or {}
        thr = _load_json_file(str(cfg["model"]["threshold_path"])) or {}
        stats = _load_json_file(str(cfg["paths"]["dataset_stats_file"])) or {}
        return jsonify({
            "model": {
                "name": service.model_name,
                "version": service.model_version,
                "selection_rationale": meta.get("selection_rationale"),
                "calibration": meta.get("calibration"),
                "library_versions": meta.get("library_versions"),
                "shap_backend": service.explainer.backend_label,
            },
            "threshold": {
                "deployed": service.threshold,
                "policy": thr.get("rule"),
                "target_recall": thr.get("target_recall"),
                "min_precision": thr.get("min_precision"),
                "validation_operating_point": thr.get("operating_point_validation"),
                "severity_bands": service.model_info["severity_bands"],
            },
            "test_results": meta.get("test_results"),
            "dataset": {
                "final": stats.get("final"),
                "sources": (stats.get("download_report") or {}).get("sources"),
                "notes": stats.get("notes"),
            },
        })

    @app.post("/predict")
    def predict():
        t0 = time.perf_counter()
        data = request.get_json(silent=True)
        if (not isinstance(data, dict) or not isinstance(data.get("url"), str)
                or not data["url"].strip()):
            return jsonify({"error": {
                "code": "invalid_request",
                "message": "JSON body must be an object with a non-empty string field 'url'",
            }}), 400
        include_enrichment = bool(data.get("include_enrichment", enrichment_default))
        include_external = bool(data.get("include_external", external_default))
        try:
            result = service.predict(data["url"], include_enrichment=include_enrichment)
        except MalformedURLError as exc:
            logger.info("predict rejected malformed url=%r: %s",
                        sanitize_url_for_logging(data["url"]), exc)
            return jsonify({"error": {"code": "malformed_url",
                                      "message": str(exc)}}), 422
        scan_id, scanned_at = _record_scan(result)
        _maybe_alert(result, scan_id)
        result["scan_id"] = scan_id
        result["scanned_at"] = scanned_at
        if include_external:
            # External intelligence (opt-in), strictly separate from the local
            # ML verdict; not stored in scan history (time-varying external data).
            result["virustotal"] = vt.lookup(normalize_url(data["url"]))
            logger.info("virustotal url=%s status=%s", result["url"],
                        result["virustotal"].get("status"))
        logger.info("predict url=%s verdict=%s probability=%.4f severity=%s "
                    "cache_hit=%s latency_ms=%.0f",
                    result["url"], result["verdict"], result["probability"],
                    result["severity"], result["cache_hit"],
                    (time.perf_counter() - t0) * 1000)
        return jsonify(result)

    @app.post("/predict_batch")
    def predict_batch():
        t0 = time.perf_counter()
        data = request.get_json(silent=True)
        urls = data.get("urls") if isinstance(data, dict) else None
        if not isinstance(urls, list):
            return jsonify({"error": {
                "code": "invalid_request",
                "message": "JSON body must be an object with a 'urls' array",
            }}), 400
        if not urls:
            return jsonify({"error": {"code": "invalid_request",
                                      "message": "'urls' array is empty"}}), 400
        if len(urls) > batch_limit:
            return jsonify({"error": {
                "code": "batch_limit_exceeded",
                "message": (f"batch of {len(urls)} URLs exceeds the configured "
                            f"limit of {batch_limit}"),
            }}), 400

        results = service.predict_batch(urls)
        valid = 0
        phishing = 0
        for item in results:
            if item.get("status") != "ok":
                continue
            valid += 1
            scan_id, scanned_at = _record_scan(item)
            item["scan_id"] = scan_id
            item["scanned_at"] = scanned_at
            _maybe_alert(item, scan_id)
            if item["verdict"] == "phishing":
                phishing += 1
        summary = {
            "total": len(results), "valid": valid, "errors": len(results) - valid,
            "phishing": phishing, "safe": valid - phishing,
        }
        logger.info("predict_batch total=%d valid=%d phishing=%d errors=%d "
                    "latency_ms=%.0f", summary["total"], valid, phishing,
                    summary["errors"], (time.perf_counter() - t0) * 1000)
        return jsonify({"results": results, "summary": summary})

    @app.get("/history")
    def history():
        raw_limit = request.args.get("limit", type=int)
        limit = history_default_limit if raw_limit is None else raw_limit
        limit = max(1, min(limit, history_max_limit))
        verdict = request.args.get("verdict")
        if verdict is not None and verdict not in VALID_VERDICTS:
            return jsonify({"error": {
                "code": "invalid_filter",
                "message": f"verdict must be one of {list(VALID_VERDICTS)}"}}), 400
        severity = request.args.get("severity")
        if severity is not None and severity not in VALID_SEVERITIES:
            return jsonify({"error": {
                "code": "invalid_filter",
                "message": f"severity must be one of {list(VALID_SEVERITIES)}"}}), 400
        order = request.args.get("order", "desc")
        if order not in ("asc", "desc"):
            return jsonify({"error": {
                "code": "invalid_filter",
                "message": "order must be 'asc' or 'desc'"}}), 400
        q = request.args.get("q")
        scans, total = store.list(limit=limit, verdict=verdict, severity=severity,
                                  q=q, order=order)
        return jsonify({
            "scans": scans, "count": len(scans), "total_matching": total,
            "limit": limit,
            "filters": {"verdict": verdict, "severity": severity, "q": q,
                        "order": order},
        })

    # ------------------------------------------------------------ errors

    @app.errorhandler(429)
    def rate_limited(_exc):
        return jsonify({"error": {
            "code": "rate_limited",
            "message": "rate limit exceeded - slow down"}}), 429

    @app.errorhandler(413)
    def payload_too_large(_exc):
        return jsonify({"error": {
            "code": "payload_too_large",
            "message": "request body exceeds the configured maximum size"}}), 413

    @app.errorhandler(404)
    def not_found(exc):
        if request.path.startswith(("/predict", "/history", "/api", "/health")):
            return jsonify({"error": {"code": "not_found",
                                      "message": "unknown endpoint"}}), 404
        return exc.get_response()

    logger.info("Flask app ready: model=%s threshold=%.4f backend=%s db=%s "
                "rate_limit=%s batch_limit=%d",
                service.model_name, service.threshold,
                service.explainer.backend_label, cfg["paths"]["database_path"],
                api_cfg.get("rate_limit"), batch_limit)
    return app


def main() -> None:
    cfg = get_config()
    app = create_app()
    server = cfg.get("server", {})
    app.run(host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 5000)),
            debug=bool(server.get("debug", False)),
            threaded=True)


if __name__ == "__main__":
    main()
