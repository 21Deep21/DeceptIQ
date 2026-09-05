"""Optional VirusTotal cross-check (external intelligence) - Phase 6.

DESIGN (defensive, privacy-preserving):
  * API key: read ONLY from the VT_API_KEY environment variable (a
    git-ignored .env file may provide it via appconfig.load_env_file).
    The key is never stored in config.yaml, never written into code,
    and never logged.
  * LOOKUP ONLY: existing VT reports are queried by URL identifier.
    URLs are NEVER submitted to VirusTotal for scanning, so this
    integration never triggers a third-party visit of the analyzed
    URL (project rule: never visit malicious URLs).
  * CREDENTIALS NEVER TRANSMITTED: userinfo is removed and
    credential-like query values redacted BEFORE the request leaves
    the host (features.ioc_extraction.strip_url_credentials).
  * Strict timeout; network failures, HTTP errors and quota limits are
    reported as unavailability - NEVER as phishing evidence.
  * LOCAL THROTTLING protects the free-tier quota (4 requests/minute):
    requests closer together than the configured interval are refused
    (or briefly delayed) with an honest status.
  * VirusTotal is NOT ground truth: results are third-party engine
    opinions, always reported separately from the local ML verdict and
    never stored in scan history (time-varying external data).
"""

from __future__ import annotations

import base64
import datetime as dt
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import requests

from features.ioc_extraction import strip_url_credentials

logger = logging.getLogger("app.virustotal")

ENV_KEY = "VT_API_KEY"
_STAT_KEYS = ("harmless", "malicious", "suspicious", "undetected", "timeout")
_MAX_THROTTLE_WAIT = 5.0  # seconds the throttle may make a caller wait

_NOTE = (
    "External intelligence from VirusTotal (third-party engine opinions), "
    "reported separately from the local ML verdict. Lookup only - URLs are "
    "never submitted for scanning. Not ground truth."
)


def url_identifier(url: str) -> str:
    """VirusTotal v3 URL identifier: unpadded base64url of the URL string."""
    return base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")


class VirusTotalClient:
    """Lookup-only VirusTotal v3 client. Never raises; every outcome is a
    structured status dict (ok / not_found / rate_limited / error /
    disabled)."""

    def __init__(self, cfg: Dict[str, Any], session: Optional[requests.Session] = None):
        vt = cfg.get("virustotal", {})
        self._cfg_enabled = bool(vt.get("enabled", True))
        self._base = str(vt.get("base_url", "https://www.virustotal.com/api/v3")).rstrip("/")
        self._timeout = float(vt.get("timeout_seconds", 8))
        self._min_interval = float(vt.get("min_request_interval_seconds", 15))
        self._key = os.environ.get(ENV_KEY, "").strip()
        self._session = session if session is not None else requests.Session()
        self._lock = threading.Lock()
        self._last_request: Optional[float] = None
        self.available = self._cfg_enabled and bool(self._key)
        if self._cfg_enabled and not self._key:
            logger.info("VirusTotal disabled: %s environment variable not set "
                        "(the application works fully without it)", ENV_KEY)

    # ------------------------------------------------------------ status

    def status(self) -> Dict[str, Any]:
        if not self._cfg_enabled:
            return {"status": "disabled", "reason": "disabled in config.yaml"}
        if not self._key:
            return {"status": "disabled", "reason": f"{ENV_KEY} environment variable not set"}
        return {"status": "enabled", "reason": "API key present (value never logged)"}

    # ------------------------------------------------------------ lookup

    def lookup(self, normalized_url: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"status": None, "note": _NOTE}
        if not self.available:
            result["status"] = "disabled"
            result["reason"] = ("disabled in config.yaml" if not self._cfg_enabled
                                else f"{ENV_KEY} environment variable not set")
            return result

        # privacy: strip credentials BEFORE anything leaves this host
        lookup_url = strip_url_credentials(normalized_url)
        result["lookup_url"] = lookup_url
        url_id = url_identifier(lookup_url)

        throttle_reason = self._throttle()
        if throttle_reason:
            result["status"] = "rate_limited"
            result["reason"] = throttle_reason
            return result

        try:
            resp = self._session.get(
                f"{self._base}/urls/{url_id}",
                headers={"x-apikey": self._key, "accept": "application/json"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            result["status"] = "error"
            result["reason"] = f"network failure ({type(exc).__name__})"
            logger.info("VirusTotal request failed (%s) - reported as unavailable",
                        type(exc).__name__)
            return result

        code = int(resp.status_code)
        if code == 404:
            result["status"] = "not_found"
            result["reason"] = ("no VirusTotal report exists for this exact URL "
                                "(lookup only - URLs are never submitted for scanning)")
            return result
        if code == 401:
            result["status"] = "error"
            result["reason"] = "invalid or expired API key (HTTP 401)"
            return result
        if code == 429:
            result["status"] = "rate_limited"
            result["reason"] = "VirusTotal quota exceeded (HTTP 429)"
            return result
        if code != 200:
            result["status"] = "error"
            result["reason"] = f"unexpected HTTP status {code}"
            return result

        try:
            payload = resp.json()
        except ValueError:
            result["status"] = "error"
            result["reason"] = "invalid JSON in VirusTotal response"
            return result

        attrs = ((payload.get("data") or {}).get("attributes") or {})
        raw_stats = attrs.get("last_analysis_stats") or {}
        stats = {k: int(raw_stats.get(k, 0) or 0) for k in _STAT_KEYS}
        ts = attrs.get("last_analysis_date")
        try:
            last_date = (dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
                         .isoformat(timespec="seconds")) if ts else None
        except (TypeError, ValueError, OSError):
            last_date = None

        result.update(
            status="ok",
            url_identifier=url_id,
            last_analysis_stats=stats,
            engines_detected=stats["malicious"] + stats["suspicious"],
            engines_total=sum(stats.values()),
            reputation=int(attrs.get("reputation", 0) or 0),
            last_analysis_date=last_date,
        )
        return result

    # ------------------------------------------------------------ throttle

    def _throttle(self) -> Optional[str]:
        """Enforce the minimum interval between VT requests.

        Returns None when the request may proceed, or a reason string
        when the local throttle refused. Never blocks the caller for
        more than _MAX_THROTTLE_WAIT seconds.
        """
        with self._lock:
            now = time.monotonic()
            if self._last_request is not None:
                earliest = self._last_request + self._min_interval
                if now < earliest:
                    wait = earliest - now
                    if wait > _MAX_THROTTLE_WAIT:
                        return (f"local throttle: next request possible in "
                                f"{wait:.0f}s (free-tier quota protection)")
                    time.sleep(wait)
            self._last_request = time.monotonic()
        return None
