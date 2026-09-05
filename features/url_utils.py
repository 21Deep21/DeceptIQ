"""URL validation, normalization, and registrable-domain extraction.

The SAME normalization is used for dataset construction (offline) and
live inference, which prevents training-serving skew: the model sees
identical URL strings in both contexts.
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import tldextract

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = ("http", "https")

# interior whitespace/control characters make a URL malformed for our purposes
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f\s]")
_HOSTNAME_CHARS_RE = re.compile(r"[a-z0-9._-]+")
_IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")

# default behaviour: live Public Suffix List, cached locally after first
# download; bundled snapshot fallback when offline.
_extract = tldextract.TLDExtract()


class MalformedURLError(ValueError):
    """URL cannot be parsed/normalized safely."""


def is_ip_hostname(host: str) -> bool:
    """True for IPv4 literals and bracketed IPv6 literals ('[::1]')."""
    if not host:
        return False
    if host.startswith("[") and host.endswith("]") and ":" in host:
        return True
    if not _IPV4_RE.fullmatch(host):
        return False
    return all(0 <= int(octet) <= 255 for octet in host.split("."))


def normalize_url(url: str) -> str:
    """Validate and normalize a URL string.

    Rules:
      * strip surrounding whitespace; reject interior whitespace/control chars
      * reject URLs longer than MAX_URL_LENGTH
      * accept only http/https schemes
      * lowercase scheme and hostname; punycode-encode unicode hostnames
      * validate hostname shape (no empty labels, sane character set)
      * preserve userinfo, port, path, query, fragment (IOC value)

    Raises MalformedURLError for anything unsafe. Userinfo is preserved
    deliberately: an '@' in a URL is a classic phishing obfuscation
    signal we want the model to see.
    """
    if not isinstance(url, str):
        raise MalformedURLError("URL must be a string")
    s = url.strip()
    if not s:
        raise MalformedURLError("empty URL")
    if _CONTROL_RE.search(s):
        raise MalformedURLError("whitespace or control character inside URL")
    if len(s) > MAX_URL_LENGTH:
        raise MalformedURLError(f"URL longer than {MAX_URL_LENGTH} characters")

    parts = urlsplit(s)
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise MalformedURLError(
            f"unsupported scheme: {parts.scheme or '(missing)'} - http/https only"
        )
    if not parts.netloc:
        raise MalformedURLError("missing host")
    try:
        port = parts.port
    except ValueError as exc:
        raise MalformedURLError(f"invalid port: {exc}") from exc

    userinfo, at_sign, hostport = parts.netloc.rpartition("@")
    if hostport.startswith("["):  # IPv6 literal, possibly with port
        close = hostport.find("]")
        if close == -1:
            raise MalformedURLError("malformed IPv6 literal")
        host = hostport[1:close].lower()
        if not host or ":" not in host:
            raise MalformedURLError("malformed IPv6 literal")
        norm_host = f"[{host}]"
    else:
        host = hostport.partition(":")[0].lower()
        if not host:
            raise MalformedURLError("empty host")
        if is_ip_hostname(host):
            norm_host = host
        else:
            try:
                candidate = (
                    host.encode("idna").decode("ascii") if not host.isascii() else host
                )
            except (UnicodeError, ValueError) as exc:
                raise MalformedURLError(f"invalid hostname: {exc}") from exc
            if not _HOSTNAME_CHARS_RE.fullmatch(candidate):
                raise MalformedURLError("invalid characters in hostname")
            if ".." in candidate or candidate.startswith(".") or candidate.endswith("."):
                raise MalformedURLError("malformed hostname labels")
            norm_host = candidate

    netloc = (userinfo + "@") if at_sign else ""
    netloc += norm_host
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))


def hostname_of(url: str) -> str:
    """Lowercase ASCII host (no userinfo/port). IPv6 keeps brackets.

    Raises MalformedURLError for invalid input."""
    norm = normalize_url(url)
    hostport = urlsplit(norm).netloc.rpartition("@")[2]
    if hostport.startswith("["):
        return hostport.rpartition("]")[0] + "]"
    return hostport.partition(":")[0]


def registrable_domain_of_host(host: str) -> Optional[str]:
    """Registrable domain of a bare hostname ('sub.example.co.uk' ->
    'example.co.uk'), or None for IP literals / hosts without a suffix."""
    if not host or is_ip_hostname(host):
        return None
    ext = _extract(host)
    if ext.suffix and ext.domain:
        return f"{ext.domain}.{ext.suffix}"
    return None


def extract_registrable_domain(url: str) -> str:
    """Registrable domain of a URL, or the IP literal when the host is an IP.

    Raises MalformedURLError when the host has no registrable domain
    (e.g. 'localhost', bare public suffix): such records cannot be
    grouped safely for domain-aware evaluation and are dropped during
    dataset cleaning."""
    host = hostname_of(url)
    if is_ip_hostname(host):
        return host
    registered = registrable_domain_of_host(host)
    if registered is None:
        raise MalformedURLError(f"no registrable domain in host '{host}'")
    return registered


def subdomain_count(url: str) -> int:
    """Labels below the registrable domain (0 for IP literals/bare hosts)."""
    host = hostname_of(url)
    if is_ip_hostname(host):
        return 0
    ext = _extract(host)
    if not (ext.suffix and ext.domain):
        return 0
    return len([label for label in ext.subdomain.split(".") if label])
