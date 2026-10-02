"""RDAP (Registration Data Access Protocol) acquisition — v2.0 Phase A.

The research draft specifies RDAP as the registration-evidence group:
structured JSON, the official successor to WHOIS, queried via the IANA
bootstrap redirector (https://rdap.org/domain/<name>).

Philosophy identical to whois_features/dns_features: best-effort, never
raises, unavailability is recorded as a STATUS — never a phishing signal.
RDAP privacy redaction simply yields absent fields (None). Statuses:
ok | not_found | rate_limited | timeout | error.

Pure acquisition module: parsing returns faithful fields; derived
features (domain age etc.) belong to the Phase B feature builder.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

RDAP_BOOTSTRAP = "https://rdap.org/domain/"


@dataclass
class RdapInfo:
    domain: str
    registrar: Optional[str] = None
    creation_date: Optional[str] = None      # ISO string as returned by the registry
    expiration_date: Optional[str] = None
    status: List[str] = field(default_factory=list)
    nameservers: List[str] = field(default_factory=list)
    status_code: Optional[str] = None        # ok | not_found | rate_limited | timeout | error
    error: Optional[str] = None
    raw_json: Optional[str] = None           # for hashing; RDAP payloads carry no credentials


def parse_rdap_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Faithful extraction from an RDAP domain object (RFC 9083).

    Redacted/absent fields stay None/[] — never imputed.
    """
    events: Dict[str, str] = {}
    for e in payload.get("events") or []:
        action, date = e.get("eventAction"), e.get("eventDate")
        if action and date:
            events[action] = date
    registrar: Optional[str] = None
    for ent in payload.get("entities") or []:
        if "registrar" in (ent.get("roles") or []):
            va = ent.get("vcardArray") or []
            if len(va) > 1 and isinstance(va[1], list):
                for item in va[1]:
                    if (isinstance(item, (list, tuple)) and len(item) > 3
                            and item[0] == "fn" and item[3]):
                        registrar = str(item[3])
                        break
            if registrar:
                break
    nameservers = sorted(
        str(ns.get("ldhName", "")).rstrip(".").lower()
        for ns in payload.get("nameservers") or [] if ns.get("ldhName")
    )
    return {
        "registrar": registrar,
        "creation_date": events.get("registration"),
        "expiration_date": events.get("expiration"),
        "status": list(payload.get("status") or []),
        "nameservers": nameservers,
    }


def query_rdap(domain: str, timeout: float = 8.0, base_url: str = RDAP_BOOTSTRAP,
               session: Optional[requests.Session] = None) -> RdapInfo:
    """Never raises. Every failure mode maps to a status with a reason."""
    sess = session if session is not None else requests.Session()
    info = RdapInfo(domain=domain)
    url = f"{base_url}{domain}"
    try:
        resp = sess.get(url, timeout=timeout,
                        headers={"accept": "application/rdap+json, application/json"})
    except requests.Timeout:
        info.status_code, info.error = "timeout", "RDAP request timed out"
        return info
    except requests.RequestException as exc:
        info.status_code, info.error = "error", f"{type(exc).__name__}: {exc}"[:200]
        return info

    code = int(resp.status_code)
    if code == 200:
        try:
            payload = resp.json()
        except ValueError:
            info.status_code, info.error = "error", "invalid JSON in RDAP response"
            return info
        for key, value in parse_rdap_payload(payload).items():
            setattr(info, key, value)
        info.status_code = "ok"
        info.raw_json = resp.text
        return info
    if code == 404:
        info.status_code, info.error = "not_found", "no RDAP record (domain not in registry)"
    elif code in (429, 503):
        info.status_code, info.error = "rate_limited", f"RDAP service pressure (HTTP {code})"
    else:
        info.status_code, info.error = "error", f"unexpected HTTP {code}"
    return info


def rdap_response_sha256(info: RdapInfo) -> Optional[str]:
    """Hash of the raw registry response, for the evidence log (draft 4)."""
    if not info.raw_json:
        return None
    return hashlib.sha256(info.raw_json.encode("utf-8")).hexdigest()


if __name__ == "__main__":  # demo: python -m features.rdap_features example.com nonexistent-xyz.invalid
    import sys

    for d in sys.argv[1:]:
        i = query_rdap(d, timeout=10)
        print(f"{d}: status={i.status_code} registrar={i.registrar!r} "
              f"created={i.creation_date} expires={i.expiration_date} "
              f"ns={i.nameservers} error={i.error!r}")
