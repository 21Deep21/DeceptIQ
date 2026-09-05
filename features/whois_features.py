"""WHOIS enrichment (best-effort, network-dependent).

ARCHITECTURE (see decision D1): WHOIS is scan-time enrichment shown in
the investigation console for a single URL. It is NOT part of the
trained model's features - bulk WHOIS collection is impractical
(rate-limited per-TLD servers) and non-reproducible (values change).

UNAVAILABLE != PHISHING: privacy protection, rate limits, unsupported
registries and network failures all produce no data. Missing values
stay None and are reported as 'Unavailable' - never as a risk signal.
"""

from __future__ import annotations

import datetime as dt
import logging
import socket
from dataclasses import dataclass
from typing import Optional

import whois as _whois  # python-whois

logger = logging.getLogger(__name__)

_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S")


@dataclass
class WhoisInfo:
    domain: str
    registrar: Optional[str] = None
    creation_date: Optional[dt.date] = None
    expiration_date: Optional[dt.date] = None
    domain_age_days: Optional[int] = None
    days_to_expiry: Optional[int] = None
    error: Optional[str] = None  # why data is unavailable (never a verdict)


def _to_date(value) -> Optional[dt.date]:
    """python-whois returns dates as datetime, list-of-datetime, or str."""
    if isinstance(value, (list, tuple)):
        for v in value:
            d = _to_date(v)
            if d:
                return d
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        for fmt in _DATE_FORMATS:
            try:
                return dt.datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None
    return None


def _clean_str(value) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)):
        for v in value:
            s = _clean_str(v)
            if s:
                return s
    return None


class _default_socket_timeout:
    """Temporarily set the process-wide default socket timeout.

    python-whois exposes no per-query timeout, so we set the socket
    default for the duration of the query and restore it afterwards.
    Known caveat (documented): this briefly mutates global state - an
    acceptable trade-off for the CLI / single-request enrichment path.
    """

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        self._old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self.seconds)
        return self

    def __exit__(self, *exc):
        socket.setdefaulttimeout(self._old)
        return False


def query_whois(domain: str, timeout: float = 5.0) -> WhoisInfo:
    """Never raises. Fields stay None when unavailable."""
    info = WhoisInfo(domain=domain)
    try:
        with _default_socket_timeout(timeout):
            record = _whois.whois(domain)
    except Exception as exc:  # WHOIS failures are EXPECTED, not exceptions
        info.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return info
    if record is None:
        info.error = "empty WHOIS response"
        return info

    info.registrar = _clean_str(getattr(record, "registrar", None))
    info.creation_date = _to_date(getattr(record, "creation_date", None))
    info.expiration_date = _to_date(getattr(record, "expiration_date", None))

    today = dt.date.today()
    if info.creation_date is not None:
        age = (today - info.creation_date).days
        info.domain_age_days = age if age >= 0 else None  # future dates = bad data
    if info.expiration_date is not None:
        remaining = (info.expiration_date - today).days
        info.days_to_expiry = remaining if remaining >= 0 else None

    if info.registrar is None and info.creation_date is None and info.expiration_date is None:
        info.error = "WHOIS returned no usable fields"
    return info


if __name__ == "__main__":  # manual demo: python -m features.whois_features example.com
    import sys

    for d in sys.argv[1:]:
        info = query_whois(d)
        print(
            f"{d}: registrar={info.registrar!r} created={info.creation_date} "
            f"age_days={info.domain_age_days} days_to_expiry={info.days_to_expiry} "
            f"error={info.error!r}"
        )
