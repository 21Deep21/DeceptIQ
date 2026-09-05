"""IOC (Indicator of Compromise) extraction and URL deconstruction.

Defensive analysis only: this module decomposes a URL STRING into its
structural components and reports notable characteristics. The URL is
never visited, resolved or requested. NOTHING here is proof of
phishing - every flag is an observational indicator intended to help a
human analyst investigate, and flag wording says so explicitly.

SECURITY / DATA-MINIMISATION:
  * passwords embedded in userinfo (user:pass@host) are redacted to
    '[REDACTED]' in ALL structured output;
  * query parameter values whose NAME is credential-like (password,
    passwd, pass, pwd, secret) are redacted to '[REDACTED]' as well,
    because API responses, logs and the SQLite history echo/store
    these values. session/token values are KEPT - they are IOC
    evidence, and the spec's own example displays 'session=123';
  * the URL carried in the result ('components.url') and returned to
    callers is the sanitized form. Best-effort limitation (documented):
    percent-encoded credential values may partially appear inside
    encoded-sequence previews;
  * use sanitize_url_for_logging() before writing URLs to logs - it
    redacts userinfo passwords AND credential-like query values and
    never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

from features.feature_extraction import SHORTENER_DOMAINS
from features.url_utils import (
    MalformedURLError,
    hostname_of,
    is_ip_hostname,
    normalize_url,
    registrable_domain_of_host,
    subdomain_count,
)

EXECUTABLE_EXTENSIONS = frozenset({
    ".exe", ".scr", ".bat", ".cmd", ".com", ".apk", ".msi", ".jar",
    ".vbs", ".ps1", ".hta",
})
SCRIPT_EXTENSIONS = frozenset({".php", ".asp", ".aspx", ".jsp", ".cgi", ".pl"})

SUSPICIOUS_PARAM_NAMES = frozenset({
    "session", "token", "auth", "login", "user", "username", "pass",
    "password", "email", "account", "verify", "validate", "redirect",
    "url", "next", "continue", "return", "dest", "goto", "target",
    "callback", "ref",
})

# parameter NAMES whose VALUES are redacted everywhere (data minimisation)
CREDENTIAL_PARAM_NAMES = frozenset({"password", "passwd", "pass", "pwd", "secret"})
_CREDENTIAL_QUERY_RE = re.compile(r"(?i)(^|&)(password|passwd|pass|pwd|secret)=([^&]*)")

_PERCENT_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_DOUBLE_ENCODE_RE = re.compile(r"%25[0-9A-Fa-f]{2}")
STANDARD_PORTS = (80, 443)


@dataclass(frozen=True)
class IocThresholds:
    long_path_chars: int = 100
    many_query_params: int = 5
    many_subdomains: int = 3
    many_path_segments: int = 5
    max_encoded_samples: int = 5


def _truncate(s: str, limit: int = 64) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _redact_query(query: str) -> str:
    """Replace credential-like query parameter VALUES with [REDACTED]."""
    return _CREDENTIAL_QUERY_RE.sub(r"\1\2=[REDACTED]", query)


def sanitize_url_for_logging(url: str) -> str:
    """Redact userinfo passwords and credential-like query values.

    Best-effort and NEVER raises: on any parsing problem the input is
    returned unchanged (an unparseable string cannot leak a parsed
    credential).
    """
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            userinfo, _, hostport = netloc.rpartition("@")
            user, sep, _password = userinfo.partition(":")
            redacted = f"{user}:[REDACTED]" if (sep and _password) else (user or "[REDACTED]")
            netloc = f"{redacted}@{hostport}"
        query = _redact_query(parts.query) if parts.query else parts.query
        return urlunsplit(parts._replace(netloc=netloc, query=query))
    except Exception:
        return url


def _path_extension(path: str) -> Optional[str]:
    if not path:
        return None
    last = path.rsplit("/", 1)[-1]
    if not last or "." not in last:
        return None
    return last[last.rfind("."):].lower()


def extract_iocs(url: str, thresholds: Optional[IocThresholds] = None) -> Dict[str, Any]:
    """Decompose a URL into components, IOC rows, and indicator flags.

    Raises MalformedURLError for invalid input (callers decide how to
    report - nothing crashes).
    """
    th = thresholds or IocThresholds()
    norm = normalize_url(url)
    parts = urlsplit(norm)

    scheme = parts.scheme
    port = parts.port
    path = parts.path
    query = parts.query
    fragment = parts.fragment

    host = hostname_of(norm)  # lowercase; brackets kept for IPv6
    raw_host = host[1:-1] if host.startswith("[") else host
    is_ip = is_ip_hostname(host)  # host WITH brackets detects IPv6 too

    userinfo, at_sign, _hostport = parts.netloc.rpartition("@")
    username: Optional[str] = None
    userinfo_redacted: Optional[str] = None
    if at_sign:
        user, sep, password = userinfo.partition(":")
        username = user or None
        if sep and password:
            userinfo_redacted = f"{user}:[REDACTED]"
        else:
            userinfo_redacted = user or "[REDACTED]"

    registered: Optional[str] = None if is_ip else registrable_domain_of_host(raw_host)
    tld = registered.rpartition(".")[2] if registered else None
    subdomain: Optional[str] = None
    if registered and raw_host.endswith("." + registered):
        subdomain = raw_host[: -(len(registered) + 1)] or None
    sub_count = subdomain_count(norm)

    parameters: List[Dict[str, str]] = []
    for name, value in parse_qsl(query, keep_blank_values=True):
        if name.lower() in CREDENTIAL_PARAM_NAMES:
            value = "[REDACTED]"  # data minimisation: never echo/store credentials
        parameters.append({"name": name, "value": value})
    redacted_query = _redact_query(query) if query else query

    encoded_matches = _PERCENT_RE.findall(norm)
    encoded: List[Dict[str, str]] = []
    seen = set()
    for enc in encoded_matches:
        if enc not in seen:
            seen.add(enc)
            encoded.append({"encoded": enc, "decoded": unquote(enc)})
    double_encoded = bool(_DOUBLE_ENCODE_RE.search(norm))

    ext = _path_extension(path)

    # ---------------------------------------------------------- flags
    flags: List[Dict[str, str]] = []

    def add(flag: str, detail: str) -> None:
        flags.append({"flag": flag, "detail": detail})

    if is_ip:
        add("ip_host", "Host is a raw IP literal - no domain registration or domain "
                       "reputation applies (indicator only)")
    if at_sign:
        add("userinfo_present", "userinfo before '@' can disguise the real host; the true "
                                "host is AFTER the '@' (indicator only)")
    if port is not None and port not in STANDARD_PORTS:
        add("unusual_port", f"Non-standard port {port} (standard: 80/443) - indicator only")
    if encoded_matches:
        sample = ", ".join(f"{e['encoded']}->{e['decoded']!r}" for e in encoded[: th.max_encoded_samples])
        add("percent_encoding", f"{len(encoded_matches)} percent-encoded sequence(s); "
                                f"decoded samples: {sample}")
    if double_encoded:
        add("double_encoding", "Double-encoded sequence(s) (%25xx) detected - possible "
                               "obfuscation (indicator only)")
    if len(parameters) >= th.many_query_params:
        add("many_query_params", f"{len(parameters)} query parameters (threshold "
                                 f"{th.many_query_params}) - indicator only")
    suspicious_names = sorted({p["name"].lower() for p in parameters
                               if p["name"].lower() in SUSPICIOUS_PARAM_NAMES})
    if suspicious_names:
        add("suspicious_parameter_names",
            f"parameter name(s) commonly seen in credential lures / open redirects: "
            f"{', '.join(suspicious_names)} (indicator only)")
    if path and len(path) >= th.long_path_chars:
        add("long_path", f"Path is {len(path)} characters (threshold {th.long_path_chars}) "
                         f"- indicator only")
    if path and path.count("/") >= th.many_path_segments:
        add("many_path_segments", f"{path.count('/')} path segments (threshold "
                                  f"{th.many_path_segments}) - indicator only")
    if sub_count >= th.many_subdomains:
        add("many_subdomains", f"{sub_count} subdomain labels (threshold {th.many_subdomains}) "
                               f"- indicator only")
    if registered is not None and registered in SHORTENER_DOMAINS:
        add("url_shortener", f"{registered} is a known URL shortener - the final target is "
                             f"hidden behind a redirect (indicator only)")
    if any(label.startswith("xn--") for label in raw_host.split(".")):
        add("punycode_host", "Punycode (xn--) labels: internationalised domain that can "
                             "resemble ASCII lookalikes (homograph consideration; indicator only)")
    if ext in EXECUTABLE_EXTENSIONS:
        add("executable_extension", f"Path ends in executable file type '{ext}' - indicator only")
    elif ext in SCRIPT_EXTENSIONS:
        add("script_extension", f"Path ends in server-side script extension '{ext}' "
                                f"(informational; common on legitimate sites too)")

    # ------------------------------------------------------- IOC table
    iocs: List[Dict[str, str]] = [
        {"type": "Scheme", "value": scheme, "status": "Standard", "note": ""},
        {"type": "Userinfo",
         "value": userinfo_redacted if at_sign else "Not present",
         "status": "Present - review" if at_sign else "-",
         "note": "real host follows '@'" if at_sign else ""},
        {"type": "Domain",
         "value": registered or "Unavailable (IP host)",
         "status": ("URL shortener" if (registered and registered in SHORTENER_DOMAINS)
                    else ("Observed" if registered else "-")),
         "note": ""},
        {"type": "Subdomain", "value": subdomain or "None",
         "status": "Observed" if subdomain else "-",
         "note": f"{sub_count} label(s) below the registrable domain"},
        {"type": "TLD", "value": tld or "-", "status": "Observed" if tld else "-", "note": ""},
        {"type": "IP", "value": raw_host if is_ip else "Unavailable",
         "status": "Raw IP host" if is_ip else "-", "note": ""},
        {"type": "Port", "value": str(port) if port is not None else "Not present",
         "status": ("Standard (80/443)" if (port is None or port in STANDARD_PORTS)
                    else "Non-standard"), "note": ""},
        {"type": "Path", "value": path or "None", "status": "Observed",
         "note": f"{len(path)} characters" if path else ""},
    ]
    if parameters:
        for p in parameters[:10]:
            susp = p["name"].lower() in SUSPICIOUS_PARAM_NAMES
            iocs.append({
                "type": "Parameter", "value": f"{p['name']}={_truncate(p['value'])}",
                "status": "Suspicious name" if susp else "Observed",
                "note": ("name commonly seen in credential lures / open redirects "
                         "(indicator only)") if susp else "",
            })
        if len(parameters) > 10:
            iocs.append({"type": "Parameter", "value": f"... {len(parameters) - 10} more",
                         "status": "Observed", "note": ""})
    else:
        iocs.append({"type": "Parameters", "value": "None", "status": "-", "note": ""})
    iocs.append({"type": "Fragment", "value": fragment or "Not present",
                 "status": "Observed" if fragment else "-", "note": ""})
    if encoded:
        sample = ", ".join(f"{e['encoded']}->{e['decoded']!r}"
                           for e in encoded[: th.max_encoded_samples])
        iocs.append({"type": "Encoded",
                     "value": f"{len(encoded_matches)} sequence(s): {sample}",
                     "status": "Detected", "note": "decoded previews shown"})
    else:
        iocs.append({"type": "Encoded", "value": "None", "status": "-", "note": ""})

    return {
        "components": {
            "url": sanitize_url_for_logging(norm),   # sanitized: userinfo + query creds
            "scheme": scheme,
            "userinfo": userinfo_redacted,
            "username": username,
            "host": host,
            "subdomain": subdomain,
            "registrable_domain": registered,
            "tld": tld,
            "is_ip": is_ip,
            "ip": raw_host if is_ip else None,
            "port": port,
            "path": path,
            "query": redacted_query,
            "fragment": fragment,
        },
        "parameters": parameters,
        "iocs": iocs,
        "flags": flags,
    }


def strip_url_credentials(url: str) -> str:
    """Prepare a URL for transmission to a third-party service.

    * userinfo is REMOVED entirely: it is a client-side obfuscation
      trick, not part of the resource identity, and removing it yields
      the true resource URL without any secret;
    * credential-like query values are REDACTED (not removed), because
      query parameters ARE part of the resource identity - the third
      party sees exactly the form we display, so a report lookup for a
      credential-bearing URL may legitimately return not_found. This
      privacy-first trade-off is accepted and documented.
    Best-effort; never raises.
    """
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            netloc = netloc.rpartition("@")[2]  # keep host[:port] only
        query = _redact_query(parts.query) if parts.query else parts.query
        return urlunsplit(parts._replace(netloc=netloc, query=query))
    except Exception:
        return url


if __name__ == "__main__":  # demo: python -m features.ioc_extraction <urls...>
    import sys

    for u in sys.argv[1:]:
        try:
            r = extract_iocs(u)
        except MalformedURLError as exc:
            print(f"ERROR  {u}: {exc}")
            continue
        c = r["components"]
        print()
        print("=" * 72)
        print(f"URL: {u}")
        print("COMPONENTS")
        for key in ("scheme", "userinfo", "host", "subdomain", "registrable_domain",
                    "tld", "ip", "port", "path", "query", "fragment"):
            print(f"  {key:<20} {c[key]!r}")
        print("IOC TABLE")
        print(f"  {'TYPE':<12} {'VALUE':<44} STATUS")
        for row in r["iocs"]:
            print(f"  {row['type']:<12} {_truncate(row['value'], 44):<44} {row['status']}")
        print("FLAGS")
        if r["flags"]:
            for f in r["flags"]:
                print(f"  - {f['flag']}: {f['detail']}")
        else:
            print("  (none)")
