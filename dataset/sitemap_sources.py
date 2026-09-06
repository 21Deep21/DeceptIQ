"""Legitimate deep-path URL harvesting via public sitemaps (Phase 1R).

WHY THIS EXISTS: the original benign class consisted only of constructed
'homepage' URLs (ranking lists publish domains, not URLs). Measured SHAP
evidence showed the model therefore learned 'any path implies malice'
(path_length was a near-decisive feature). This module collects REAL
legitimate URLs WITH deep paths so the benign class resembles real
traffic (homepages AND deep links).

METHOD (defensive, documented):
  1. For each curated domain, request ONLY https://<domain>/robots.txt -
     the standard sitemap-discovery mechanism - and parse 'Sitemap:'
     lines. Conventional sitemap paths are tried as fallback.
  2. Fetch the declared sitemap files (urlset or sitemapindex, plain or
     .gz), with bounded count/size and politeness delays.
  3. Extract <loc> page URLs. The discovered page URLs are dataset
     STRINGS - they are NEVER requested, visited or rendered.

No fabrication: every URL comes verbatim from a fetched sitemap.
Failures per domain are reported, never hidden.
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_SITEMAP_LINE_RE = re.compile(r"(?im)^\s*sitemap\s*:\s*(\S+)")
_ROBOTS_MAX_BYTES = 512 * 1024


def robots_sitemap_lines(robots_text: str) -> List[str]:
    """Absolute sitemap URLs declared in a robots.txt ('Sitemap:' lines)."""
    return _SITEMAP_LINE_RE.findall(robots_text or "")


def maybe_gunzip(data: bytes) -> bytes:
    """Decompress .gz sitemap files (magic-byte sniff); passthrough otherwise."""
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def parse_sitemap(data: bytes, max_locs: int = 3000
                  ) -> Tuple[List[str], List[str], bool, Optional[str]]:
    """Parse one sitemap document.

    Returns (page_urls, child_sitemap_urls, is_index, error_note).
    Namespace-agnostic: elements whose LOCAL name is 'loc' are collected;
    page-vs-child is decided by the ROOT element (urlset -> pages,
    sitemapindex -> children). Streaming via iterparse to bound memory;
    stops after max_locs entries. Parse failures are returned as an
    error note - never raised.
    """
    try:
        xml = maybe_gunzip(data)
    except (OSError, EOFError) as exc:
        return [], [], False, f"gzip decompression failed: {exc}"

    pages: List[str] = []
    children: List[str] = []
    root_local = ""
    error: Optional[str] = None
    try:
        for event, elem in ET.iterparse(io.BytesIO(xml), events=("start", "end")):
            local = elem.tag.rsplit("}", 1)[-1] if isinstance(elem.tag, str) else ""
            if event == "start" and not root_local:
                root_local = local
            if event == "end" and local == "loc":
                text = (elem.text or "").strip()
                if text:
                    if root_local == "sitemapindex":
                        children.append(text)
                    else:
                        pages.append(text)
                elem.clear()
                if len(pages) >= max_locs or len(children) >= max_locs:
                    break
    except ET.ParseError as exc:
        error = f"XML parse error: {exc}"
    return pages, children, root_local == "sitemapindex", error


def _get_bytes(session: requests.Session, url: str, net: Dict[str, Any],
               max_bytes: int) -> Optional[bytes]:
    """Bounded GET returning bytes, or None on any failure.

    Returns None (not an exception) by design: a missing robots.txt or
    sitemap is a NORMAL outcome handled by the caller (fallback paths /
    domain-failure reporting). The failure reason is logged at DEBUG.
    """
    timeout = (int(net.get("feed_connect_timeout", 10)),
               int(net.get("feed_read_timeout", 30)))
    try:
        with session.get(url, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            chunks: List[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=1 << 16):
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    logger.debug("sitemap file exceeds size cap, truncated: %s", url)
                    break
            return b"".join(chunks)
    except Exception as exc:
        logger.debug("GET %s failed: %s", url, exc)
        return None


def _discover_sitemaps(session: requests.Session, domain: str,
                       sm_cfg: Dict[str, Any], net: Dict[str, Any]) -> List[str]:
    """robots.txt 'Sitemap:' lines, else conventional fallback paths."""
    robots = _get_bytes(session, f"https://{domain}/robots.txt", net, _ROBOTS_MAX_BYTES)
    if robots is not None:
        lines = robots_sitemap_lines(robots.decode("utf-8", errors="replace"))
        if lines:
            return lines
    paths = sm_cfg.get("fallback_sitemap_paths",
                       ["sitemap.xml", "sitemap_index.xml", "sitemap-index.xml"])
    return [f"https://{domain}/{str(p).lstrip('/')}" for p in paths]


def collect_sitemap_urls(cfg: Dict[str, Any],
                         session: Optional[requests.Session] = None
                         ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Harvest benign deep-link URLs from curated domains' sitemaps.

    Returns (entries, report). entries: {'url', 'label'=0,
    'source'='sitemap', 'threat_type'='benign'}. Never raises;
    per-domain failures are counted and reported.
    """
    sm_cfg = dict(cfg.get("sitemaps", {}))
    net = dict(cfg.get("network", {}))
    domains = [str(d).strip().lower() for d in sm_cfg.get("domains", []) if str(d).strip()]
    delay = float(sm_cfg.get("request_delay_seconds", 0.4))
    max_file = int(sm_cfg.get("max_file_mb", 20)) * 1024 * 1024
    max_locs = int(sm_cfg.get("max_locs_per_file", 3000))
    per_domain_cap = int(sm_cfg.get("max_urls_per_domain", 200))
    max_files = int(sm_cfg.get("max_sitemap_files_per_domain", 3))
    max_children = int(sm_cfg.get("max_children_per_index", 15))

    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = net.get("user_agent", "phishing-url-analyzer")

    report: Dict[str, Any] = {
        "domains_total": len(domains), "domains_ok": 0, "domains_failed": 0,
        "failures": [], "sitemaps_fetched": 0, "urls_collected": 0,
        "per_domain_counts": {},
    }
    entries: List[Dict[str, Any]] = []
    seen: set = set()
    processed = 0

    logger.info(
        "sitemap harvest starting: %d domains (takes several minutes; only robots.txt "
        "and sitemap files are fetched - page URLs are never visited)",
        len(domains),
    )

    for domain in domains:
        got = 0
        candidates = _discover_sitemaps(session, domain, sm_cfg, net)
        for sm_url in candidates[:max_files]:
            data = _get_bytes(session, sm_url, net, max_file)
            if data is None:
                continue
            report["sitemaps_fetched"] += 1
            pages, children, is_index, err = parse_sitemap(data, max_locs)
            if err:
                logger.debug("sitemap parse issue for %s: %s", sm_url, err)
            targets = list(pages)
            if is_index:
                for child in children[:max_children]:
                    cdata = _get_bytes(session, child, net, max_file)
                    if cdata is None:
                        continue
                    report["sitemaps_fetched"] += 1
                    cpages, _c2, c_is_index, _cerr = parse_sitemap(cdata, max_locs)
                    if not c_is_index:
                        targets.extend(cpages)
                    if len(targets) >= per_domain_cap:
                        break
                    time.sleep(delay)
            for u in targets:
                u = u.strip()
                if not u.lower().startswith(("http://", "https://")):
                    continue
                if u in seen:
                    continue
                seen.add(u)
                entries.append({"url": u, "label": 0,
                                "source": "sitemap", "threat_type": "benign"})
                got += 1
                if got >= per_domain_cap:
                    break
            if got >= per_domain_cap:
                break
            time.sleep(delay)
        if got:
            report["domains_ok"] += 1
            report["per_domain_counts"][domain] = got
        else:
            report["domains_failed"] += 1
            report["failures"].append(domain)
        processed += 1
        if processed % 20 == 0:
            logger.info("sitemap progress: %d/%d domains processed, %d URLs collected so far",
                        processed, len(domains), len(entries))
        time.sleep(delay)

    report["urls_collected"] = len(entries)
    if report["failures"]:
        shown = ", ".join(report["failures"][:20])
        more = ("" if len(report["failures"]) <= 20
                else f" (+{len(report['failures']) - 20} more)")
        logger.info("sitemap domains yielding no URLs: %s%s", shown, more)
    logger.info("sitemap harvest: %d URLs from %d/%d domains (%d sitemap files fetched)",
                len(entries), report["domains_ok"], report["domains_total"],
                report["sitemaps_fetched"])
    return entries, report