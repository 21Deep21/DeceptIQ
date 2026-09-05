"""CEF-inspired defensive security alert log (data/alerts.log).

One structured event per scan classified as PHISHING (never for safe
verdicts). The format is CEF-INSPIRED, not strictly validated CEF:

  CEF:0|Vendor|Product|Version|SignatureID|Name|Severity|Extension

The extension carries the sanitized URL, calibrated probability,
threshold, model version, client IP and scan id. This is a defensive
detection record for SOC/SIEM workflows; it contains no secrets.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

CEF_SEVERITY = {"LOW": 3, "MEDIUM": 5, "HIGH": 8, "CRITICAL": 10}


def _cef_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


class AlertLogger:
    def __init__(self, cfg: Dict[str, Any]):
        alerts_cfg = cfg.get("alerts", {})
        self.enabled = bool(alerts_cfg.get("enabled", True))
        self._vendor = str(alerts_cfg.get("vendor", "PhishingUrlAnalyzer"))
        self._product = str(alerts_cfg.get("product", "URL Analysis Platform"))
        self._version = str(cfg.get("application", {}).get("version", "0"))
        self._logger: Optional[logging.Logger] = None
        if self.enabled:
            path = pathlib.Path(cfg["paths"]["alert_log_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            # unique logger name per instance: prevents handler cross-talk
            # when several app instances live in one process (tests).
            log = logging.getLogger(f"alerts.cef.{uuid.uuid4().hex[:12]}")
            log.setLevel(logging.INFO)
            log.propagate = False
            handler = RotatingFileHandler(
                path,
                maxBytes=int(alerts_cfg.get("max_bytes", 5_242_880)),
                backupCount=int(alerts_cfg.get("backup_count", 3)),
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            log.addHandler(handler)
            self._logger = log

    def alert(self, *, url: str, probability: float, severity: str,
              threshold: float, model: str, client_ip: str, scan_id: int) -> None:
        if not self.enabled or self._logger is None:
            return
        cef_severity = CEF_SEVERITY.get(str(severity).upper(), 8)
        extension = (
            f"cs1Label=urlVis cs1={_cef_escape(url)} "
            f"cs2Label=probability cs2={float(probability):.4f} "
            f"cs3Label=threshold cs3={float(threshold):.4f} "
            f"cs4Label=modelVersion cs4={_cef_escape(str(model))} "
            f"src={client_ip} requestId={scan_id}"
        )
        line = (f"CEF:0|{self._vendor}|{self._product}|{self._version}"
                f"|PHISHING_URL_DETECTED|URL classified as phishing"
                f"|{cef_severity}|{extension}")
        self._logger.info(line)
