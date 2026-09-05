"""DNS enrichment (best-effort, network-dependent).

Same architecture rule as WHOIS: scan-time enrichment, not model
features. DNS FAILURE != PHISHING: timeouts and resolver problems are
reported as UNAVAILABLE (None). An authoritative NXDOMAIN is reported
as an explicit 'domain does not resolve' error - also not a phishing
verdict by itself. An authoritative NoAnswer means the name exists but
the record type is absent (recorded as 0 / False - that IS information).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import dns.exception
import dns.resolver

from features.url_utils import registrable_domain_of_host

logger = logging.getLogger(__name__)


@dataclass
class DnsInfo:
    domain: str
    a_record_count: Optional[int] = None
    a_record_ttl: Optional[int] = None
    has_mx: Optional[bool] = None
    ns_count: Optional[int] = None
    ns_diversity: Optional[int] = None  # distinct registrable domains of the NS names
    error: Optional[str] = None


def query_dns(domain: str, timeout: float = 5.0) -> DnsInfo:
    """Never raises. Unavailable data stays None with an error note."""
    info = DnsInfo(domain=domain)
    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout    # per-nameserver timeout
    resolver.lifetime = timeout   # total query budget

    errors: List[str] = []
    try:
        _fill(resolver, domain, info, errors)
    except dns.resolver.NXDOMAIN:
        info.error = "NXDOMAIN (domain does not resolve)"
        return info
    except Exception as exc:  # enrichment must fail gracefully, never raise
        info.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return info
    if errors:
        info.error = "; ".join(errors)
    return info


def _fill(resolver, domain: str, info: DnsInfo, errors: List[str]) -> None:
    ans, err = _safe_resolve(resolver, domain, "A")
    if err == "no_record":
        info.a_record_count = 0  # authoritative: name exists, no A record
    elif err:
        errors.append(f"A: {err}")
    else:
        info.a_record_count = len(list(ans))
        try:
            info.a_record_ttl = int(ans.rrset.ttl)
        except Exception:
            pass

    ans, err = _safe_resolve(resolver, domain, "MX")
    if err == "no_record":
        info.has_mx = False  # authoritative: domain cannot receive mail
    elif err:
        errors.append(f"MX: {err}")
    else:
        info.has_mx = True

    ans, err = _safe_resolve(resolver, domain, "NS")
    if err == "no_record":
        info.ns_count = 0
        info.ns_diversity = 0
    elif err:
        errors.append(f"NS: {err}")
    else:
        ns_names = sorted(str(rdata).rstrip(".") for rdata in ans)
        info.ns_count = len(ns_names)
        info.ns_diversity = _registrable_diversity(ns_names)


def _safe_resolve(resolver, domain: str, rdtype: str):
    """(Answer, error). 'no_record' = authoritative NoAnswer.
    NXDOMAIN propagates (the domain itself does not exist)."""
    try:
        return resolver.resolve(domain, rdtype), None
    except dns.resolver.NoAnswer:
        return None, "no_record"
    except (dns.exception.Timeout, dns.resolver.NoNameservers) as exc:
        return None, type(exc).__name__


def _registrable_diversity(ns_names: List[str]) -> int:
    domains = set()
    for ns in ns_names:
        registered = registrable_domain_of_host(ns)
        if registered:
            domains.add(registered)
    return len(domains)


if __name__ == "__main__":  # manual demo: python -m features.dns_features example.com
    import sys

    for d in sys.argv[1:]:
        info = query_dns(d)
        print(
            f"{d}: a={info.a_record_count} ttl={info.a_record_ttl} mx={info.has_mx} "
            f"ns={info.ns_count} ns_diversity={info.ns_diversity} error={info.error!r}"
        )
