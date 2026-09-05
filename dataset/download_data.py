"""Acquire raw threat-intelligence and legitimacy feeds (Phase 1).

SCOPE / SAFETY: this module downloads plain-text FEED FILES from their
official publishers. It never requests, visits, or renders any URL that
appears inside those feeds. URLs are parsed as strings only.

Sources (no account/API key required unless noted):
  * OpenPhish free community feed .... phishing URLs (label 1)
  * URLhaus (abuse.ch) ............... malware-DISTRIBUTION URLs
                                        (label 1, threat_type
                                        'malware_distribution' - distinct
                                        from phishing)
  * PhishTank ....................... requires manual registration +
                                        download; place the CSV archive at
                                        data/raw/phishtank.csv to ingest
  * Tranco .......................... legitimate top-domain ranking
                                        (label 0), via public API
  * Majestic Million ................ legitimate fallback ranking

Every failure is reported honestly and recorded in the dataset stats.
Nothing is ever fabricated to reach a target count.
"""

from __future__ import annotations

import csv
import io
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from appconfig import get_config, setup_console_logging

logger = logging.getLogger(__name__)


class SourceUnavailableError(RuntimeError):
    """A feed source could not be downloaded/parsed (reported, never fatal)."""


# ------------------------------------------------------------------ HTTP

def _new_session(cfg: Dict[str, Any]) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = cfg["network"]["user_agent"]
    return s


def _timeouts(cfg: Dict[str, Any]) -> Tuple[int, int]:
    net = cfg["network"]
    return int(net["feed_connect_timeout"]), int(net["feed_read_timeout"])


def _get_text(session: requests.Session, url: str, cfg: Dict[str, Any]) -> str:
    """GET a text feed with retries, timeouts, and a hard size cap."""
    net = cfg["network"]
    retries = int(net.get("feed_retries", 3))
    backoff = float(net.get("feed_retry_backoff", 2.0))
    max_bytes = int(net.get("max_feed_mb", 200)) * 1024 * 1024
    last_error: Exception = SourceUnavailableError("not attempted")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, timeout=_timeouts(cfg), stream=True) as resp:
                resp.raise_for_status()
                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=1 << 18):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > max_bytes:
                        raise SourceUnavailableError(
                            f"feed exceeds {max_bytes // (1024 * 1024)} MB cap"
                        )
                return b"".join(chunks).decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            logger.warning("GET %s attempt %d/%d failed: %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise SourceUnavailableError(
        f"GET {url} failed after {retries} attempts: {last_error}"
    )


def _save_raw(cfg: Dict[str, Any], filename: str, text: str) -> Path:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / filename
    out.write_text(text, encoding="utf-8")
    logger.info("saved raw feed copy: %s (%d bytes)", out, out.stat().st_size)
    return out


# ------------------------------------------------------------------ parsers

def _parse_url_lines(text: str, source: str, threat_type: str) -> List[Dict[str, Any]]:
    entries = []
    for line in text.splitlines():
        u = line.strip()
        if u[:7].lower() == "http://" or u[:8].lower() == "https://":
            entries.append(
                {"url": u, "label": 1, "source": source, "threat_type": threat_type}
            )
    return entries


def parse_openphish(text: str) -> List[Dict[str, Any]]:
    return _parse_url_lines(text, source="openphish", threat_type="phishing")


def parse_urlhaus(text: str) -> List[Dict[str, Any]]:
    return _parse_url_lines(text, source="urlhaus", threat_type="malware_distribution")


def parse_tranco_csv(text: str) -> List[str]:
    domains = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 2 and row[0].strip().isdigit():
            domains.append(row[1].strip().lower())
    return domains


def parse_majestic_csv(text: str) -> List[str]:
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None:
        return []
    try:
        idx = header.index("Domain")
    except ValueError:
        return []
    return [
        row[idx].strip().lower()
        for row in reader
        if len(row) > idx and row[idx].strip()
    ]


def parse_phishtank_csv(path: Path) -> List[Dict[str, Any]]:
    """Ingest an optional, manually downloaded PhishTank archive CSV."""
    import pandas as pd

    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "url" not in df.columns:
        raise SourceUnavailableError(f"PhishTank CSV at {path} has no 'url' column")
    if "verified" in df.columns:
        mask = df["verified"].astype(str).str.strip().str.lower().isin(
            ["yes", "y", "true", "1"]
        )
        df = df[mask]
    entries = []
    for u in df["url"].dropna():
        u = str(u).strip()
        if u:
            entries.append(
                {"url": u, "label": 1, "source": "phishtank", "threat_type": "phishing"}
            )
    return entries


# ------------------------------------------------------------------ fetchers

def fetch_openphish(cfg: Dict[str, Any], session: requests.Session) -> List[Dict[str, Any]]:
    text = _get_text(session, cfg["sources"]["openphish_feed_url"], cfg)
    entries = parse_openphish(text)
    if not entries:
        raise SourceUnavailableError("OpenPhish feed parsed to 0 URLs")
    _save_raw(cfg, "openphish_feed.txt", text)
    return entries


def fetch_urlhaus(cfg: Dict[str, Any], session: requests.Session) -> List[Dict[str, Any]]:
    errors: List[str] = []
    for key in ("urlhaus_text_recent_url", "urlhaus_text_url"):
        url = cfg["sources"][key]
        try:
            text = _get_text(session, url, cfg)
            entries = parse_urlhaus(text)
            if entries:
                _save_raw(cfg, "urlhaus_text.txt", text)
                return entries
            errors.append(f"{url}: parsed to 0 URLs")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    try:  # last resort: abuse.ch API (POST returns plain text)
        resp = session.post(cfg["sources"]["urlhaus_api_text_url"], data={}, timeout=_timeouts(cfg))
        resp.raise_for_status()
        entries = parse_urlhaus(resp.text)
        if entries:
            _save_raw(cfg, "urlhaus_text.txt", resp.text)
            return entries
        errors.append("api: parsed to 0 URLs")
    except Exception as exc:
        errors.append(f"api: {exc}")
    raise SourceUnavailableError("all URLhaus endpoints failed: " + " | ".join(errors))


def fetch_tranco_domains(cfg: Dict[str, Any], session: requests.Session) -> List[str]:
    """Tranco API flow: create list -> poll until ready -> download CSV."""
    base = str(cfg["sources"]["tranco_api_base"]).rstrip("/")
    timeout = _timeouts(cfg)

    resp = None
    last = None
    for kwargs in ({"json": {}}, {"data": ""}):
        r = session.post(f"{base}/lists", timeout=timeout, **kwargs)
        if r.ok:
            resp = r
            break
        last = r
    if resp is None:
        raise SourceUnavailableError(f"Tranco list creation failed: HTTP {last.status_code if last is not None else '?'}")
    list_id = (resp.json() or {}).get("list_id")
    if not list_id:
        raise SourceUnavailableError("Tranco did not return a list_id")

    deadline = time.time() + int(cfg["network"].get("tranco_poll_timeout", 120))
    download_hint = None
    while time.time() < deadline:
        info = session.get(f"{base}/lists/{list_id}", timeout=timeout)
        info.raise_for_status()
        data = info.json()
        status = str(data.get("status", "")).lower()
        download_hint = data.get("download")
        if status in ("done", "completed", "finished"):
            break
        if status in ("error", "failed"):
            raise SourceUnavailableError(f"Tranco list build failed: {data}")
        time.sleep(5)
    else:
        raise SourceUnavailableError("Tranco list build timed out")

    candidates = []
    if isinstance(download_hint, str) and download_hint:
        candidates.append(
            download_hint if download_hint.startswith("http")
            else f"https://tranco-list.eu{download_hint}"
        )
    candidates.append(f"{base}/lists/{list_id}/download")
    for c in candidates:
        try:
            text = _get_text(session, c, cfg)
            domains = parse_tranco_csv(text)
            if domains:
                _save_raw(cfg, "tranco.csv", text)
                return domains
        except Exception as exc:
            logger.warning("Tranco download candidate failed: %s (%s)", c, exc)
    raise SourceUnavailableError("could not download Tranco CSV")


def fetch_majestic_domains(cfg: Dict[str, Any], session: requests.Session) -> List[str]:
    text = _get_text(session, cfg["sources"]["majestic_csv_url"], cfg)
    domains = parse_majestic_csv(text)
    if not domains:
        raise SourceUnavailableError("Majestic CSV parsed to 0 domains")
    _save_raw(cfg, "majestic_million.csv", text)
    return domains


def construct_legitimate_entries(
    domains: List[str], source_name: str, target: int, pool: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """Build benign entries from a domain RANKING list.

    Disclosed transformation (not fabrication): ranking lists publish
    DOMAINS, not URLs, so the benign URL is constructed as
    'https://<domain>/'. Known bias (benign rows carry no path/query
    content) is recorded in the dataset stats notes.
    """
    pool_domains = [d for d in domains if d][: int(pool)]
    n = min(int(target), len(pool_domains))
    chosen = rng.sample(pool_domains, n) if n < len(pool_domains) else list(pool_domains)
    return [
        {"url": f"https://{d}/", "label": 0, "source": source_name, "threat_type": "benign"}
        for d in chosen
    ]


# ------------------------------------------------------------------ orchestration

def collect_entries(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds = cfg["dataset"]
    rng = random.Random(int(ds["random_seed"]))
    session = _new_session(cfg)
    report: Dict[str, Any] = {"sources": {}}
    malicious: List[Dict[str, Any]] = []

    for name, fetcher in (("openphish", fetch_openphish), ("urlhaus", fetch_urlhaus)):
        try:
            entries = fetcher(cfg, session)
            malicious.extend(entries)
            report["sources"][name] = {"status": "ok", "raw_records": len(entries)}
            logger.info("%s: %d raw URLs", name, len(entries))
        except Exception as exc:
            report["sources"][name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300]}
            logger.warning("%s unavailable: %s", name, exc)
        time.sleep(1)  # politeness gap between feeds

    # PhishTank: manual download only (archive requires registration)
    pt_path = Path(str(ds.get("phishtank_csv", "")))
    if str(pt_path) and pt_path.exists():
        try:
            pt_entries = parse_phishtank_csv(pt_path)
            malicious.extend(pt_entries)
            report["sources"]["phishtank"] = {"status": "ok", "raw_records": len(pt_entries)}
        except Exception as exc:
            report["sources"]["phishtank"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    else:
        report["sources"]["phishtank"] = {
            "status": "skipped",
            "reason": f"no CSV at {pt_path} (PhishTank archive requires manual registration/download)",
        }

    # Legitimate side: Tranco primary, Majestic fallback
    legit: List[Dict[str, Any]] = []
    try:
        domains = fetch_tranco_domains(cfg, session)
        legit = construct_legitimate_entries(
            domains, "tranco", int(ds["target_legitimate"]), int(ds["legitimate_domain_pool"]), rng
        )
        report["sources"]["tranco"] = {"status": "ok", "raw_records": len(domains), "selected": len(legit)}
    except Exception as exc:
        report["sources"]["tranco"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300]}
        logger.warning("Tranco unavailable (%s) - falling back to Majestic Million", exc)
        try:
            domains = fetch_majestic_domains(cfg, session)
            legit = construct_legitimate_entries(
                domains, "majestic", int(ds["target_legitimate"]), int(ds["legitimate_domain_pool"]), rng
            )
            report["sources"]["majestic"] = {"status": "ok (fallback)", "raw_records": len(domains), "selected": len(legit)}
        except Exception as exc2:
            report["sources"]["majestic"] = {"status": "failed", "reason": f"{type(exc2).__name__}: {exc2}"[:300]}

    return malicious + legit, report


def log_report(stats: Dict[str, Any]) -> None:
    f = stats["final"]
    logger.info("=" * 68)
    logger.info("DATASET REPORT (all numbers are actual measured values)")
    for name, info in stats.get("download_report", {}).get("sources", {}).items():
        logger.info("source %-10s %s", name, info)
    c = stats["cleaning"]
    logger.info(
        "input=%d malformed_removed=%d duplicates_removed=%d label_conflicts=%d",
        c["input"], c["malformed_removed"], c["duplicates_removed"], c["label_conflicts_resolved"],
    )
    if c["malformed_reasons"]:
        logger.info("malformed reasons: %s", c["malformed_reasons"])
    logger.info("dropped_by_domain_cap=%d", stats["dropped_by_domain_cap"])
    s = stats["sampling"]
    logger.info(
        "selected: malicious=%d benign=%d (targets: %d/%d, available: %d/%d)",
        s["malicious_selected"], s["benign_selected"],
        stats["config"]["target_phishing"], stats["config"]["target_legitimate"],
        s["malicious_available"], s["benign_available"],
    )
    logger.info(
        "final: total=%d labels=%s sources=%s threat_types=%s unique_domains=%d",
        f["total"], f["by_label"], f["by_source"], f["by_threat_type"],
        f["unique_registrable_domains"],
    )
    logger.info("wrote %s and %s", stats["output_file"], stats["stats_file"])
    logger.info("=" * 68)


def main() -> int:
    setup_console_logging()
    logger.info("dataset acquisition starting (feed FILES only; feed URLs are never visited)")
    cfg = get_config()
    entries, report = collect_entries(cfg)
    if not entries:
        logger.error("no entries collected from any source - cannot build dataset")
        return 1
    from dataset.build_dataset import build_from_entries  # lazy import avoids circularity

    stats = build_from_entries(entries, cfg, download_report=report)
    if stats["final"]["total"] == 0:
        logger.error("dataset is empty after cleaning - nothing written")
        return 1
    log_report(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
