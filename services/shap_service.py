"""SHAP explainability service for the deployed model.

Two interchangeable EXACT TreeSHAP backends:
  * "shap"   - shap.TreeExplainer (the shap package; depends on numba).
  * "native" - LightGBM's built-in TreeSHAP (booster.predict(pred_contrib=True)),
               the same algorithm implemented in LightGBM's C++ core; no
               shap/numba dependency. Automatic fallback when the shap
               package is unavailable (e.g. no wheel for this Python).

The active backend is selected at construction with a health check and
is REPORTED in every explanation - never assumed.

Explanation space (documented, verified numerically per URL):
  base_value + sum(phi_i) == raw LightGBM margin (log-odds)
  base_value = expected margin over the training data.
  phi_i > 0  -> feature evidence INCREASES phishing risk (log-odds)
  phi_i < 0  -> feature evidence DECREASES phishing risk
The DEPLOYED probability is a sigmoid (Platt) calibration of the raw
margin - a monotone mapping, so the direction of each feature's
evidence is preserved. SHAP values are NOT probabilities.

Inputs must be normalisable URL strings (features.url_utils.
normalize_url is applied first; malformed input raises
MalformedURLError and is the caller's responsibility to report).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from features.feature_extraction import FEATURE_NAMES
from features.ioc_extraction import sanitize_url_for_logging
from features.url_utils import normalize_url

logger = logging.getLogger(__name__)

try:
    import shap  # noqa: N813
    SHAP_IMPORT_ERROR = ""
except ImportError as exc:  # expected on Python versions without numba wheels
    shap = None
    SHAP_IMPORT_ERROR = str(exc)

DENSE_COUNT = len(FEATURE_NAMES)
MIN_ABS_CONTRIBUTION = 1e-9
RECONSTRUCTION_WARN_TOL = 0.01

BACKEND_LABELS = {
    "shap": "shap.TreeExplainer (exact TreeSHAP)",
    "native": "LightGBM native TreeSHAP (booster pred_contrib)",
}


@dataclass
class FeatureContribution:
    name: str
    value: float
    display_value: str
    shap: float


@dataclass
class UrlExplanation:
    url: str
    backend: str
    base_value: float
    raw_margin: Optional[float]
    reconstruction_error: Optional[float]
    lexical_total: float
    ngram_total: float
    increasing: List[FeatureContribution] = field(default_factory=list)
    decreasing: List[FeatureContribution] = field(default_factory=list)

    @property
    def backend_label(self) -> str:
        return BACKEND_LABELS.get(self.backend, self.backend)

    def to_dict(self) -> Dict[str, Any]:
        def _c(c: FeatureContribution) -> Dict[str, Any]:
            return {
                "feature": c.name,
                "value": round(c.value, 6),
                "display_value": c.display_value,
                "contribution": round(c.shap, 6),
                "direction": "increasing" if c.shap > 0 else "decreasing",
            }
        return {
            "url": self.url,
            "backend": self.backend,
            "backend_label": self.backend_label,
            "base_value": round(self.base_value, 6),
            "raw_margin": None if self.raw_margin is None else round(self.raw_margin, 6),
            "reconstruction_error": (None if self.reconstruction_error is None
                                     else round(self.reconstruction_error, 9)),
            "lexical_total": round(self.lexical_total, 6),
            "ngram_total": round(self.ngram_total, 6),
            "increasing": [_c(c) for c in self.increasing],
            "decreasing": [_c(c) for c in self.decreasing],
        }


class ShapExplainer:
    """Explains the RAW pipeline of the deployed bundle (not the
    calibrated wrapper): calibrated probability decides the verdict,
    SHAP explains the evidence of the underlying model."""

    @property
    def backend_label(self) -> str:
        """Human-readable label of the ACTIVE backend (reported, never assumed)."""
        return BACKEND_LABELS.get(self.backend, self.backend)

    def __init__(self, raw_pipeline, backend: str = "auto", top_k_default: int = 10):
        self.builder = raw_pipeline.named_steps["features"]
        self.model = raw_pipeline.named_steps["model"]
        self.feature_names = [str(n) for n in self.builder.feature_names_]
        self._n_features = len(self.feature_names)
        self._top_k_default = int(top_k_default)
        self._tree_explainer = None
        self.backend = self._resolve_backend(backend)

    # ------------------------------------------------------------ backend

    def _require_native_capable(self, why: str) -> None:
        if not (hasattr(self.model, "booster_") and self.model.booster_ is not None):
            raise RuntimeError(
                f"{why}: the native TreeSHAP backend requires a fitted LightGBM "
                f"model (got {type(self.model).__name__}). Install the shap "
                f"package or use a LightGBM pipeline."
            )

    def _resolve_backend(self, requested: str) -> str:
        if requested not in ("auto", "shap", "native"):
            raise ValueError(f"backend must be auto|shap|native, got {requested!r}")
        if requested == "native":
            self._require_native_capable("native backend requested")
            return "native"
        if shap is None:
            if requested == "shap":
                raise RuntimeError(
                    f"shap backend requested but the shap package is unavailable: "
                    f"{SHAP_IMPORT_ERROR}"
                )
            self._require_native_capable("auto backend needs a fallback")
            logger.info("shap package unavailable (%s) - using LightGBM native TreeSHAP",
                        SHAP_IMPORT_ERROR or "not installed")
            return "native"
        try:  # health check (also triggers numba JIT once, at startup)
            t0 = time.time()
            te = shap.TreeExplainer(self.model)
            sv = te.shap_values(np.zeros((1, self._n_features), dtype=np.float64))
            _ = np.asarray(sv).shape
            logger.info("shap.TreeExplainer ready (health check + JIT: %.1fs)",
                        time.time() - t0)
            self._tree_explainer = te
            return "shap"
        except Exception as exc:
            if requested == "shap":
                raise RuntimeError(f"shap backend failed its health check: {exc}") from exc
            logger.warning("shap.TreeExplainer unusable (%s) - falling back to "
                           "LightGBM native TreeSHAP", exc)
            self._require_native_capable("auto backend fallback")
            return "native"

    # ------------------------------------------------------------ compute

    def _compute_shap(self, Xd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(shap_values [n, m], base_values [n]) in log-odds space."""
        if self.backend == "shap":
            sv = self._tree_explainer.shap_values(Xd)
            if isinstance(sv, (list, tuple)):  # older APIs: [class0, class1]
                sv = sv[-1]
            sv = np.asarray(sv, dtype=np.float64)
            if sv.ndim != 2 or sv.shape != Xd.shape:
                raise RuntimeError(f"unexpected shap_values shape {sv.shape} "
                                   f"for input {Xd.shape}")
            ev = self._tree_explainer.expected_value
            if isinstance(ev, (list, tuple, np.ndarray)):
                ev = ev[-1]
            return sv, np.full(Xd.shape[0], float(ev), dtype=np.float64)
        contribs = np.asarray(
            self.model.booster_.predict(Xd, pred_contrib=True), dtype=np.float64
        )
        expected = (Xd.shape[0], Xd.shape[1] + 1)
        if contribs.shape != expected:
            raise RuntimeError(f"unexpected pred_contrib shape {contribs.shape}, "
                               f"expected {expected}")
        return contribs[:, :-1], contribs[:, -1]

    def _raw_margins(self, Xd: np.ndarray) -> Optional[np.ndarray]:
        if not (hasattr(self.model, "booster_") and self.model.booster_ is not None):
            return None
        return np.asarray(self.model.booster_.predict(Xd, raw_score=True),
                          dtype=np.float64)

    # ------------------------------------------------------------ public

    def explain(self, urls: Sequence[str], top_k: Optional[int] = None) -> List[UrlExplanation]:
        """Per-URL SHAP explanations (urls are normalised first)."""
        k = int(top_k or self._top_k_default)
        norm_urls = [normalize_url(u) for u in urls]  # raises MalformedURLError early
        X = self.builder.transform(norm_urls)
        Xd = X.toarray().astype(np.float64)
        shap_matrix, base = self._compute_shap(Xd)
        margins = self._raw_margins(Xd)

        results: List[UrlExplanation] = []
        for i, url in enumerate(norm_urls):
            phi = shap_matrix[i]
            total = float(phi.sum())
            if margins is not None:
                margin = float(margins[i])
                recon = abs(base[i] + total - margin)
                if recon > RECONSTRUCTION_WARN_TOL:
                    logger.warning("SHAP reconstruction mismatch for %r: %.4f "
                                   "(possible backend space mismatch)",
                                   sanitize_url_for_logging(url), recon)
            else:
                margin, recon = None, None

            contribs: List[FeatureContribution] = []
            for j in range(self._n_features):
                v = float(phi[j])
                if abs(v) < MIN_ABS_CONTRIBUTION:
                    continue
                val = float(Xd[i, j])
                if j < DENSE_COUNT:
                    display = f"{val:g}"
                else:
                    display = f"present (tfidf={val:.3f})" if val > 0 else "absent"
                contribs.append(FeatureContribution(name=self.feature_names[j],
                                                    value=val,
                                                    display_value=display,
                                                    shap=v))
            contribs.sort(key=lambda c: abs(c.shap), reverse=True)
            results.append(UrlExplanation(
                url=sanitize_url_for_logging(url),  # reported URL is sanitized
                backend=self.backend,
                base_value=float(base[i]),
                raw_margin=margin,
                reconstruction_error=recon,
                lexical_total=float(phi[:DENSE_COUNT].sum()),
                ngram_total=float(phi[DENSE_COUNT:].sum()),
                increasing=[c for c in contribs if c.shap > 0][:k],
                decreasing=[c for c in contribs if c.shap < 0][:k],
            ))
        return results


if __name__ == "__main__":  # demo: python -m services.shap_service <urls...>
    import argparse

    import joblib

    from appconfig import get_config, setup_console_logging
    from features.url_utils import MalformedURLError

    def _cli(argv=None) -> int:
        setup_console_logging()
        p = argparse.ArgumentParser(
            description="SHAP explanation demo on the DEPLOYED model bundle")
        p.add_argument("urls", nargs="+")
        p.add_argument("--backend", default="auto", choices=["auto", "shap", "native"])
        p.add_argument("--top-k", type=int, default=None)
        args = p.parse_args(argv)

        cfg = get_config()
        bundle = joblib.load(cfg["model"]["model_path"])
        top_k = args.top_k or int(cfg.get("explainability", {}).get("top_k_features", 10))
        explainer = ShapExplainer(bundle["raw_pipeline"], backend=args.backend,
                                  top_k_default=top_k)
        threshold = float(bundle["threshold"])
        model_name = str(bundle["model_name"])

        for raw_url in args.urls:
            try:
                norm = normalize_url(raw_url)
            except MalformedURLError as exc:
                print(f"ERROR  {raw_url}: {exc}")
                continue
            proba = float(bundle["calibrated_model"].predict_proba([norm])[0, 1])
            verdict = "PHISHING" if proba >= threshold else "SAFE"
            exp = explainer.explain([norm])[0]
            print()
            print("=" * 72)
            print(f"URL            {exp.url}")
            print(f"PROBABILITY    {proba:.4f} (calibrated)    "
                  f"VERDICT: {verdict} (threshold {threshold:.4f})")
            print(f"MODEL          {model_name} | backend: {exp.backend_label}")
            if exp.raw_margin is not None:
                print(f"MARGIN         raw={exp.raw_margin:+.4f}  base={exp.base_value:+.4f}  "
                      f"reconstruction error={exp.reconstruction_error:.2e}")
            print(f"EVIDENCE       lexical {exp.lexical_total:+.4f} | "
                  f"character n-grams {exp.ngram_total:+.4f}  (log-odds space)")
            print()
            print("FEATURES INCREASING PHISHING RISK")
            for c in exp.increasing or [None]:
                print(f"  {c.name:<30} {c.display_value:<24} shap={c.shap:+.4f}"
                      if c else "  (none)")
            print("FEATURES REDUCING PHISHING RISK")
            for c in exp.decreasing or [None]:
                print(f"  {c.name:<30} {c.display_value:<24} shap={c.shap:+.4f}"
                      if c else "  (none)")
        return 0

    raise SystemExit(_cli())
