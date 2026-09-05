"""Prediction orchestration service (Phase 5).

Per URL (fully offline unless enrichment is requested):
  validation/normalization (features.url_utils) -> calibrated probability
  (bundle's calibrated model) -> threshold verdict -> severity band ->
  SHAP explanation (services.shap_service) -> IOC extraction
  (features.ioc_extraction) -> optional WHOIS/DNS enrichment.

The URL is never visited. Only URL strings and public metadata are
analyzed.

CACHING: the deterministic core (probability, verdict, severity, SHAP,
IOCs) is cached in a bounded LRU keyed by the NORMALIZED URL. WHOIS,
DNS and any future external intelligence are NEVER cached (time
sensitive). A cache hit still produces a fresh scan record and alert:
the cache saves computation, not events.

SECURITY: URLs returned/stored are sanitized - userinfo passwords and
credential-like query values are redacted before anything leaves this
service (API responses, logs, database).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib

from features.dns_features import query_dns
from features.ioc_extraction import extract_iocs, sanitize_url_for_logging
from features.url_utils import MalformedURLError, normalize_url
from features.whois_features import query_whois
from services.shap_service import ShapExplainer

logger = logging.getLogger("app.prediction")

_BUNDLE_FORMAT_VERSION = 1

_WHOIS_NOTE = ("WHOIS unavailability is NOT a phishing signal - privacy "
               "protection, rate limits, unsupported registries or network "
               "failure all produce no data.")
_DNS_NOTE = ("DNS failure is NOT a phishing signal - it indicates "
             "unavailability only.")


class BoundedLRU:
    """Thread-safe bounded LRU cache (OrderedDict-based)."""

    def __init__(self, maxsize: int):
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"entries": len(self._data), "max_entries": self._maxsize,
                    "hits": self._hits, "misses": self._misses}

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def _invalid_input_echo(value: Any) -> str:
    """Never echo raw non-string input; strings are sanitized."""
    if isinstance(value, str):
        return sanitize_url_for_logging(value)
    return f"<invalid input: {type(value).__name__}>"


def _whois_to_dict(info) -> Dict[str, Any]:
    return {
        "available": bool(info.registrar or info.creation_date or info.expiration_date),
        "domain": info.domain,
        "registrar": info.registrar,
        "creation_date": info.creation_date.isoformat() if info.creation_date else None,
        "expiration_date": (info.expiration_date.isoformat()
                            if info.expiration_date else None),
        "domain_age_days": info.domain_age_days,
        "days_to_expiry": info.days_to_expiry,
        "error": info.error,
    }


def _dns_enrich(host: str, registered: Optional[str], timeout: float) -> Dict[str, Any]:
    """A records reflect the HOST; MX/NS reflect the REGISTRABLE domain
    (mail/name-server capability is a property of the domain)."""
    host_info = query_dns(host, timeout=timeout)
    reg_info = (query_dns(registered, timeout=timeout)
                if registered and registered != host else host_info)
    errors = []
    if host_info.error:
        errors.append(f"host: {host_info.error}")
    if reg_info is not host_info and reg_info.error:
        errors.append(f"domain: {reg_info.error}")
    return {
        "available": (host_info.a_record_count is not None
                      or reg_info.ns_count is not None
                      or reg_info.has_mx is not None),
        "host": host,
        "domain": registered or host,
        "a_record_count": host_info.a_record_count,
        "a_record_ttl": host_info.a_record_ttl,
        "has_mx": reg_info.has_mx,
        "ns_count": reg_info.ns_count,
        "ns_diversity": reg_info.ns_diversity,
        "error": "; ".join(errors) or None,
    }


class PredictionService:
    """Loads the deployed bundle once; serves predictions thereafter."""

    def __init__(self, bundle_path: str, config: Dict[str, Any],
                 shap_backend: Optional[str] = None):
        self._bundle = joblib.load(bundle_path)
        if int(self._bundle.get("format_version", -1)) != _BUNDLE_FORMAT_VERSION:
            raise RuntimeError(
                f"unsupported bundle format_version "
                f"{self._bundle.get('format_version')!r} in {bundle_path}"
            )
        self.model_name = str(self._bundle["model_name"])
        self.created_at = str(self._bundle.get("created_at", "unknown"))
        self.model_version = f"{self.model_name}@{self.created_at}"
        self.threshold = float(self._bundle["threshold"])
        self.calibrated = self._bundle["calibrated_model"]

        exp_cfg = config.get("explainability", {})
        top_k = int(exp_cfg.get("top_k_features", 10))
        backend = shap_backend or str(exp_cfg.get("shap_backend", "auto"))
        self.explainer = ShapExplainer(self._bundle["raw_pipeline"],
                                       backend=backend, top_k_default=top_k)

        sev = config.get("severity", {})
        self._low_max = float(sev.get("low_max", 0.25))
        self._medium_max = float(sev.get("medium_max", 0.60))
        self._high_max = float(sev.get("high_max", 0.90))
        if not (0.0 <= self._low_max <= self._medium_max <= self._high_max <= 1.0):
            raise ValueError(
                f"invalid severity bands: low_max={self._low_max}, "
                f"medium_max={self._medium_max}, high_max={self._high_max}"
            )
        self._enrich_timeout = float(
            config.get("network", {}).get("enrichment_timeout", 5))
        self._cache = BoundedLRU(int(config.get("cache", {}).get("max_entries", 512)))

        self.model_info: Dict[str, Any] = {
            "name": self.model_name,
            "version": self.model_version,
            "probability_is_calibrated": True,
            "calibration": "sigmoid (Platt) fitted on the validation split (Phase 3)",
            "shap_backend": self.explainer.backend_label,
            "threshold": self.threshold,
            "severity_bands": {
                "low_max": self._low_max,
                "medium_max": self._medium_max,
                "high_max": self._high_max,
            },
            "library_versions": self._bundle.get("library_versions", {}),
        }
        logger.info("prediction service ready: model=%s threshold=%.4f backend=%s",
                    self.model_name, self.threshold, self.explainer.backend_label)

    # ------------------------------------------------------------ public

    @property
    def cache_stats(self) -> Dict[str, int]:
        return self._cache.stats()

    def severity_of(self, probability: float) -> str:
        """Probability bands, INDEPENDENT of the classification threshold
        (documented decision D1): severity communicates risk magnitude,
        the verdict communicates the operational decision."""
        p = float(probability)
        if p < self._low_max:
            return "LOW"
        if p < self._medium_max:
            return "MEDIUM"
        if p < self._high_max:
            return "HIGH"
        return "CRITICAL"

    def predict(self, url: str, include_enrichment: bool = False) -> Dict[str, Any]:
        """Analyze one URL. Raises MalformedURLError for invalid input."""
        norm = normalize_url(url)
        core = self._cache.get(norm)
        cache_hit = core is not None
        if core is None:
            core = self._compute_cores([norm])[0]
            self._cache.put(norm, core)
        result = dict(core)
        result["cache_hit"] = cache_hit
        result["threshold"] = self.threshold
        result["model"] = self.model_info
        result["enrichment"] = (self._enrichment(core["ioc"]["components"])
                                if include_enrichment else None)
        return result

    def predict_batch(self, urls: Sequence[Any]) -> List[Dict[str, Any]]:
        """Vectorized batch analysis. One invalid entry produces an error
        item; valid entries are still analyzed (never all-or-nothing).
        Network enrichment is intentionally NOT performed in batch mode.
        """
        entries: List[Optional[Dict[str, Any]]] = [None] * len(urls)
        valid: List[Tuple[int, str, Optional[Dict[str, Any]]]] = []
        for index, raw in enumerate(urls):
            if not isinstance(raw, str) or not raw.strip():
                entries[index] = {
                    "url": _invalid_input_echo(raw),
                    "status": "error",
                    "error": {"code": "invalid_input",
                              "message": "URL must be a non-empty string"},
                }
                continue
            try:
                norm = normalize_url(raw)
            except MalformedURLError as exc:
                entries[index] = {
                    "url": sanitize_url_for_logging(raw),
                    "status": "error",
                    "error": {"code": "malformed_url", "message": str(exc)},
                }
                continue
            valid.append((index, norm, self._cache.get(norm)))

        uncached = [norm for _, norm, core in valid if core is None]
        new_cores = self._compute_cores(uncached) if uncached else []
        core_by_url: Dict[str, Dict[str, Any]] = dict(zip(uncached, new_cores))
        for norm, core in core_by_url.items():
            self._cache.put(norm, core)

        for index, norm, cached in valid:
            core = cached if cached is not None else core_by_url[norm]
            result = dict(core)
            result["status"] = "ok"
            result["cache_hit"] = cached is not None
            result["threshold"] = self.threshold
            result["model"] = self.model_info
            result["enrichment"] = None
            entries[index] = result
        return [entry for entry in entries if entry is not None]

    # ------------------------------------------------------------ private

    def _compute_cores(self, norm_urls: List[str]) -> List[Dict[str, Any]]:
        probas = self.calibrated.predict_proba(list(norm_urls))[:, 1]
        explanations = self.explainer.explain(list(norm_urls))
        cores: List[Dict[str, Any]] = []
        for url, proba, explanation in zip(norm_urls, probas, explanations):
            probability = float(proba)
            ioc = extract_iocs(url)
            shap = explanation.to_dict()
            shap["note"] = (
                "SHAP values are exact TreeSHAP log-odds contributions of the raw "
                "model (not probabilities). Positive contributions increase "
                "phishing risk, negative contributions reduce it. The verdict is "
                "decided by the calibrated probability and the deployed threshold."
            )
            cores.append({
                "url": ioc["components"]["url"],  # sanitized (credentials redacted)
                "probability": probability,
                "verdict": "phishing" if probability >= self.threshold else "safe",
                "severity": self.severity_of(probability),
                "shap": shap,
                "ioc": ioc,
            })
        return cores

    def _enrichment(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort WHOIS + DNS. NEVER cached, NEVER a verdict signal."""
        registered = components.get("registrable_domain")
        host = components.get("host")
        is_ip = bool(components.get("is_ip"))
        if is_ip or not registered:
            whois_part: Dict[str, Any] = {
                "available": False, "domain": host,
                "error": ("IP-literal host: WHOIS applies to registered "
                          "domains, not IP addresses"),
            }
            dns_part: Dict[str, Any] = {
                "available": False, "host": host, "domain": host,
                "a_record_count": None, "a_record_ttl": None, "has_mx": None,
                "ns_count": None, "ns_diversity": None,
                "error": "IP-literal host: forward DNS resolution does not apply",
            }
        else:
            whois_part = _whois_to_dict(
                query_whois(registered, timeout=self._enrich_timeout))
            dns_part = _dns_enrich(host, registered, self._enrich_timeout)
        whois_part["note"] = _WHOIS_NOTE
        dns_part["note"] = _DNS_NOTE
        return {"whois": whois_part, "dns": dns_part, "cached": False}
