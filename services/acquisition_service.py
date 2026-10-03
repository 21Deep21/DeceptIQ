"""Live evidence-cascade service (v2.0 Phase E).

Productization of the EVALUATED P1 policy: for a submitted URL, compute the
URL-only probability with the fusion model (none condition); if the host is
a registrable domain and the probability lies inside the frozen uncertainty
band, acquire DNS/RDAP evidence LIVE (reliability-ranked order, strict
timeouts, bounded requests, confidence early-stopping, conservative p95
budget eligibility - mirroring model/acquisition.replay_policy exactly),
then produce the final verdict from the fusion model with the acquired
evidence, plus exact TreeSHAP over the fused feature space.

SAFETY: the analyzed host is never contacted. DNS goes to resolvers, RDAP
to the IANA bootstrap redirector - the same passive third-party services
measured in Phase A.

The deployed v1 stack is untouched: this service is used ONLY when /predict
is called with include_evidence=true; the response is labeled with the v2
model name and threshold (0.0392). Calibrators: per-condition
(validation-fitted, Phase E) when available, else the bundle's shared one.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from features.dns_features import query_dns
from features.evidence_features import (
    EVIDENCE_FEATURE_NAMES,
    N_EVIDENCE_FEATURES,
    evidence_vector_from_records,
)
from features.feature_extraction import FEATURE_NAMES
from features.ioc_extraction import extract_iocs
from features.rdap_features import query_rdap
from features.url_utils import (
    hostname_of,
    is_ip_hostname,
    normalize_url,
    registrable_domain_of_host,
)

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

logger = logging.getLogger("app.acquisition")

GROUPS = ("dns", "rdap")
_MAX_AGE_DAYS, _MAX_EXPIRY_DAYS = 12000, 36500


def _state_name(acquired: set) -> str:
    d, r = "dns" in acquired, "rdap" in acquired
    return "full" if d and r else "dns_only" if d else "rdap_only" if r else "none"


class AcquisitionService:
    """Loads the model_v2 bundle + frozen policy; serves live cascades."""

    def __init__(self, bundle_path: str, policy_path: str,
                 calibrators_path: Optional[str], cfg: Dict[str, Any]):
        self._bundle = joblib.load(bundle_path)
        self.model_name = str(self._bundle["model_name"])
        self.model_version = f"{self.model_name}@{self._bundle.get('created_at', '?')}"
        self.threshold = float(self._bundle["threshold"])
        self._rf = self._bundle["rf_model"]
        self._builder = self._bundle["url_builder"]

        pol = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        self.band = tuple(pol["band"])
        profs = pol.get("profiles", {})
        self._profiles = {g: profs.get(g, {}) for g in GROUPS}
        self._order = sorted(GROUPS,
                             key=lambda g: -float(self._profiles[g].get("score", 0.0)))
        logger.info("cascade policy: band=%s order=%s (frozen from Phase C)",
                    self.band, self._order)

        acq = dict(cfg.get("acquisition", {}))
        self._max_requests = int(acq.get("max_requests", 2))
        lb = acq.get("latency_budget_ms")
        self._latency_budget_ms = float(lb) if lb is not None else None
        ecfg = dict(cfg.get("evidence", {}))
        self._dns_timeout = float(ecfg.get("dns_timeout", 5))
        self._rdap_timeout = float(ecfg.get("rdap_timeout", 8))

        sev = cfg.get("severity", {})
        self._low = float(sev.get("low_max", 0.25))
        self._med = float(sev.get("medium_max", 0.60))
        self._high = float(sev.get("high_max", 0.90))

        self._calibrators: Dict[str, Any] = {}
        if calibrators_path and Path(calibrators_path).exists():
            try:
                self._calibrators = joblib.load(calibrators_path)
                logger.info("per-condition calibrators loaded: %s",
                            sorted(self._calibrators))
            except Exception as exc:
                logger.warning("condition calibrators unreadable (%s); "
                               "using shared calibrator", exc)
        self._shared = self._bundle["calibrated_model"]

        self._feature_names = (list(self._builder.feature_names_)
                               + list(EVIDENCE_FEATURE_NAMES))
        self._n_features = len(self._feature_names)
        self._top_k = int(cfg.get("explainability", {}).get("top_k_features", 10))
        self._tree_explainer = None
        if _SHAP_OK:
            try:
                te = shap.TreeExplainer(self._rf)
                _ = np.asarray(te.shap_values(
                    np.zeros((1, self._n_features), dtype=np.float64),
                    check_additivity=False)).shape
                self._tree_explainer = te
                logger.info("fusion SHAP ready (%d features, probability space)",
                            self._n_features)
            except Exception as exc:
                logger.warning("fusion SHAP unavailable: %s", exc)
        logger.info("acquisition service ready: %s @ threshold %.4f",
                    self.model_name, self.threshold)

    # ------------------------------------------------------------ helpers

    def severity_of(self, p: float) -> str:
        if p < self._low:
            return "LOW"
        if p < self._med:
            return "MEDIUM"
        if p < self._high:
            return "HIGH"
        return "CRITICAL"

    def _matrix(self, norm_url: str, E_row: np.ndarray):
        X_url = self._builder.transform([norm_url])
        return hstack([X_url, csr_matrix(E_row.reshape(1, -1))], format="csr")

    def _predict(self, norm_url: str, E_row: np.ndarray, condition: str):
        X = self._matrix(norm_url, E_row)
        cal = self._calibrators.get(condition) or self._shared
        return float(np.asarray(cal.predict_proba(X))[0, 1]), X

    def _live_dns(self, domain: str):
        t0 = time.perf_counter()
        info = query_dns(domain, timeout=self._dns_timeout)
        dur = (time.perf_counter() - t0) * 1000.0
        any_present = any(v is not None for v in
                          (info.a_record_count, info.ns_count, info.has_mx))
        if any_present:                                   # mirrors collect_evidence
            fields = {"a_record_count": info.a_record_count,
                      "a_record_ttl": info.a_record_ttl,
                      "has_mx": info.has_mx,
                      "ns_count": info.ns_count,
                      "ns_diversity": info.ns_diversity}
            return {"status": "ok", "fields": fields}, dur
        err = info.error or ""
        status = ("not_found" if "NXDOMAIN" in err
                  else "timeout" if "Timeout" in err else "error")
        return {"status": status, "fields": None}, dur

    def _live_rdap(self, domain: str):
        t0 = time.perf_counter()
        info = query_rdap(domain, timeout=self._rdap_timeout)
        dur = (time.perf_counter() - t0) * 1000.0
        if info.status_code == "ok":
            fields = {"registrar": info.registrar,
                      "creation_date": info.creation_date,
                      "expiration_date": info.expiration_date,
                      "status": info.status,
                      "nameservers": info.nameservers}
            return {"status": "ok", "fields": fields}, dur
        return {"status": info.status_code or "error", "fields": None}, dur

    def _explain(self, X, final_p: float) -> Optional[Dict[str, Any]]:
        if self._tree_explainer is None:
            return None
        # this shap build requires a dense ndarray for isnan validation:
        # convert the sparse fused matrix before TreeSHAP
        Xd = np.asarray(X.todense(), dtype=np.float64)
        sv = np.asarray(self._tree_explainer.shap_values(Xd,
                                                         check_additivity=False),
                        dtype=np.float64)
        if sv.ndim == 3 and sv.shape[:2] == X.shape and sv.shape[2] >= 2:
            sv = sv[:, :, -1]                              # positive-class slice
        phi = sv[0]
        ev = self._tree_explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            ev = ev[-1]
        output = float(self._rf.predict_proba(Xd)[0, 1])
        recon = abs(float(ev) + float(phi.sum()) - output)
        contribs = []
        for j, v in enumerate(phi):
            if abs(v) < 1e-9:
                continue
            val = float(X[0, j])
            if j < len(FEATURE_NAMES):
                display = f"{val:g}"
            elif j < len(self._builder.feature_names_):
                display = f"present (tfidf={val:.3f})" if val > 0 else "absent"
            else:
                display = f"{val:g}"
            contribs.append({"feature": self._feature_names[j], "value": val,
                             "display_value": display, "contribution": float(v),
                             "direction": "increasing" if v > 0 else "decreasing"})
        contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        n_url = len(self._builder.feature_names_)
        return {
            "url": "",  # filled by caller (sanitized)
            "backend": "shap",
            "backend_label": "shap.TreeExplainer (exact TreeSHAP, fusion model)",
            "space": "probability",
            "base_value": round(float(ev), 6),
            "raw_margin": round(output, 6),
            "reconstruction_error": round(recon, 9),
            "lexical_total": round(float(phi[:len(FEATURE_NAMES)].sum()), 6),
            "ngram_total": round(float(phi[len(FEATURE_NAMES):n_url].sum()), 6),
            "evidence_total": round(float(phi[n_url:].sum()), 6),
            "increasing": contribs[:self._top_k] if False else [
                c for c in contribs if c["contribution"] > 0][:self._top_k],
            "decreasing": [c for c in contribs if c["contribution"] < 0][:self._top_k],
            "note": ("SHAP values are exact TreeSHAP contributions to the fusion "
                     "model's predicted probability (sklearn RF has no log-odds "
                     "margin). The last 14 features are the evidence block "
                     "(incl. missingness indicators). Model evidence, NOT the "
                     "calibrated probability itself; the verdict comes from the "
                     "calibrated probability and the v2 threshold."),
        }

    # ------------------------------------------------------------ public

    def analyze(self, url: str) -> Dict[str, Any]:
        norm = normalize_url(url)
        host = hostname_of(norm)
        domain = None
        if not is_ip_hostname(host):
            registered = registrable_domain_of_host(host)
            domain = registered or None

        E = np.zeros(N_EVIDENCE_FEATURES)
        p0, _X = self._predict(norm, E, "none")
        requests: List[Dict[str, Any]] = []
        acquired: Dict[str, dict] = {}
        charged = 0.0
        confident = (p0 < self.band[0]) or (p0 > self.band[1])

        if domain is not None and not confident:
            for g in self._order:
                if len(requests) >= self._max_requests:
                    break
                p95 = float(self._profiles[g].get("p95_ms") or 0.0)
                if (self._latency_budget_ms is not None
                        and (self._latency_budget_ms - charged) < p95):
                    break
                rec, dur = (self._live_dns(domain) if g == "dns"
                            else self._live_rdap(domain))
                requests.append({"group": g, "status": rec["status"],
                                 "duration_ms": round(dur, 1)})
                charged += dur
                acquired[g] = rec
                cond = _state_name(set(acquired))
                E = evidence_vector_from_records(acquired.get("dns"),
                                                 acquired.get("rdap"))
                p_cur, _X = self._predict(norm, E, cond)
                if (p_cur < self.band[0]) or (p_cur > self.band[1]):
                    break

        cond = _state_name(set(acquired))
        final_p, X = self._predict(norm, E, cond)
        verdict = "phishing" if final_p >= self.threshold else "safe"
        logger.info("cascade decision: domain=%s p0=%.4f band=%s confident=%s "
                    "requests=%d condition=%s final_p=%.4f verdict=%s",
                    domain, p0, self.band, confident, len(requests), cond,
                    final_p, verdict)
        ioc = extract_iocs(norm)
        sanitized = ioc["components"]["url"]

        result: Dict[str, Any] = {
            "url": sanitized,
            "cache_hit": False,          # live cascade is never cached (v1 contract)
            "probability": final_p,
            "verdict": verdict,
            "severity": self.severity_of(final_p),
            "threshold": self.threshold,
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "probability_is_calibrated": True,
                "calibration": ("per-condition sigmoid (validation-fitted)"
                                if self._calibrators else
                                "shared sigmoid (full-evidence fitted)"),
                "threshold": self.threshold,
                "severity_bands": {"low_max": self._low, "medium_max": self._med,
                                   "high_max": self._high},
            },
            "ioc": ioc,
            "acquisition": {
                "policy": "P1 (live)",
                "band": list(self.band),
                "initial_probability": round(p0, 6),
                "escalated": bool(requests),
                "requests": requests,
                "final_condition": cond,
                "final_probability": round(final_p, 6),
                "total_latency_ms": round(charged, 1),
                "profiles": {g: {"availability": self._profiles[g].get("availability"),
                                 "p95_ms": self._profiles[g].get("p95_ms")}
                             for g in GROUPS},
                "note": ("URL-only first stage; DNS/RDAP acquired live only when "
                         "uncertain (frozen band, reliability-ranked, early-stopped, "
                         "budget-checked). Costs are measured request durations. "
                         "The analyzed host is never contacted; DNS/RDAP are passive "
                         "third-party services. Display enrichment flags are ignored "
                         "in cascade mode."),
            },
        }
        sh = self._explain(X, final_p)
        if sh is not None:
            sh["url"] = sanitized
            result["shap"] = sh
        return result
