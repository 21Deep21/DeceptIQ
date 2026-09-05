"""Dataset cleaning, normalization, balancing, and persistence (Phase 1).

Pipeline (offline once raw feeds are on disk):
  1. normalize every URL (features/url_utils.normalize_url)
  2. drop malformed URLs (counted and reported - never a crash)
  3. extract the registrable domain (tldextract); hosts without one dropped
  4. exact-URL dedup; on cross-label duplicates the MALICIOUS label wins
     (the threat feed is authoritative for that exact URL)
  5. per-(class, domain) URL cap for domain diversity
  6. seeded class balancing to target sizes - ALL available records are
     used when fewer than target exist; counts reported, never padded
  7. seeded shuffle; write urls.csv + dataset_stats.json

DOMAIN-LEVEL LEAKAGE: registrable_domain is the GROUPING VARIABLE for
the domain-aware train/test split and StratifiedGroupKFold CV in Phase
2. Random URL-level splitting would put different URLs of the same
domain into both train and test (e.g. evil-example.com/login in train,
evil-example.com/account in test) and inflate performance. Grouped
splitting prevents this.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from appconfig import get_config, setup_console_logging
from features.url_utils import MalformedURLError, extract_registrable_domain, normalize_url

logger = logging.getLogger(__name__)

CSV_COLUMNS = ["url", "label", "source", "threat_type", "registrable_domain"]


def clean_entries(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize, drop malformed, extract domains, dedup, resolve label conflicts."""
    stats: Dict[str, Any] = {
        "input": len(entries),
        "malformed_removed": 0,
        "malformed_reasons": {},
        "duplicates_removed": 0,
        "label_conflicts_resolved": 0,
        "after_cleaning": 0,
    }
    by_url: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        try:
            url = normalize_url(str(e.get("url", "")))
            domain = extract_registrable_domain(url)
        except MalformedURLError as exc:
            stats["malformed_removed"] += 1
            reason = str(exc).split(":", 1)[0] or "malformed"
            stats["malformed_reasons"][reason] = stats["malformed_reasons"].get(reason, 0) + 1
            continue
        record = {
            "url": url,
            "label": int(e.get("label", 0)),
            "source": str(e.get("source", "unknown")),
            "threat_type": str(e.get("threat_type", "")),
            "registrable_domain": domain,
        }
        existing = by_url.get(url)
        if existing is None:
            by_url[url] = record
        elif existing["label"] != record["label"]:
            # exact same URL reported as both benign and malicious:
            # the threat feed is authoritative for that exact URL
            if record["label"] == 1:
                by_url[url] = record
            stats["label_conflicts_resolved"] += 1
        else:
            stats["duplicates_removed"] += 1
    cleaned = list(by_url.values())
    stats["after_cleaning"] = len(cleaned)
    return cleaned, stats


def apply_domain_cap(entries: List[Dict[str, Any]], cap: int) -> Tuple[List[Dict[str, Any]], int]:
    """Keep at most `cap` URLs per (label, registrable_domain)."""
    if cap <= 0:
        return entries, 0
    counts: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for e in entries:
        key = f"{int(e['label'])}|{e['registrable_domain']}"
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= cap:
            kept.append(e)
        else:
            dropped += 1
    return kept, dropped


def sample_balanced(
    entries: List[Dict[str, Any]], n_malicious: int, n_benign: int, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Seeded balancing. Never fabricates: if fewer records exist than the
    target, ALL of them are used and the shortfall is reported."""
    rng = random.Random(seed)
    malicious = [e for e in entries if e["label"] == 1]
    benign = [e for e in entries if e["label"] == 0]
    n_mal = min(n_malicious, len(malicious))
    n_ben = min(n_benign, len(benign))
    stats = {
        "malicious_available": len(malicious),
        "benign_available": len(benign),
        "malicious_selected": n_mal,
        "benign_selected": n_ben,
    }
    sel_mal = rng.sample(malicious, n_mal) if n_mal < len(malicious) else list(malicious)
    sel_ben = rng.sample(benign, n_ben) if n_ben < len(benign) else list(benign)
    result = sel_mal + sel_ben
    rng.shuffle(result)
    return result, stats


def build_from_entries(
    entries: List[Dict[str, Any]], cfg: Dict[str, Any], download_report: Dict[str, Any] = None
) -> Dict[str, Any]:
    ds = cfg["dataset"]
    seed = int(ds["random_seed"])
    target_mal = int(ds["target_phishing"])
    target_ben = int(ds["target_legitimate"])
    cap = int(ds["max_urls_per_domain"])

    cleaned, cleaning_stats = clean_entries(entries)
    capped, dropped_by_cap = apply_domain_cap(cleaned, cap)
    final, sampling_stats = sample_balanced(capped, target_mal, target_ben, seed)

    df = pd.DataFrame(final, columns=CSV_COLUMNS)
    out_csv = Path(cfg["paths"]["processed_dataset_file"])
    out_stats = Path(cfg["paths"]["dataset_stats_file"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    stats = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "config": {
            "target_phishing": target_mal,
            "target_legitimate": target_ben,
            "max_urls_per_domain": cap,
            "legitimate_domain_pool": int(ds["legitimate_domain_pool"]),
        },
        "download_report": download_report or {},
        "cleaning": cleaning_stats,
        "dropped_by_domain_cap": dropped_by_cap,
        "sampling": sampling_stats,
        "final": {
            "total": int(len(df)),
            "by_label": {str(k): int(v) for k, v in df["label"].value_counts().items()},
            "by_source": {str(k): int(v) for k, v in df["source"].value_counts().items()},
            "by_threat_type": {str(k): int(v) for k, v in df["threat_type"].value_counts().items()},
            "unique_registrable_domains": int(df["registrable_domain"].nunique()),
            "unique_registrable_domains_by_label": {
                str(k): int(v)
                for k, v in df.groupby("label")["registrable_domain"].nunique().items()
            },
        },
        "notes": _notes(download_report),
        "output_file": str(out_csv),
        "stats_file": str(out_stats),
    }
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d rows)", out_csv, len(df))
    logger.info("wrote %s", out_stats)
    return stats


def _notes(download_report: Dict[str, Any]) -> List[str]:
    notes = [
        "Legitimate URLs are constructed as 'https://<domain>/' from domain ranking "
        "lists (Tranco/Majestic publish domains, not URLs). Known bias: benign rows "
        "carry no path/query content; path-based features will look stronger than on "
        "real traffic. Documented as a dataset limitation.",
        "URLhaus records are malware-distribution URLs, not strictly phishing. They "
        "are labelled malicious (1) with threat_type 'malware_distribution' to keep "
        "the distinction explicit.",
        "registrable_domain is the grouping variable for domain-aware splitting in "
        "Phase 2 (prevents same-domain train/test leakage).",
        "URLs on legitimately ranked domains that appear in threat feeds (compromised "
        "hosts) are kept with their malicious label: the URL is malicious even when "
        "the domain is popular. Grouped splitting keeps such domains in one fold.",
    ]
    for name, info in (download_report or {}).get("sources", {}).items():
        if isinstance(info, dict) and info.get("status") in ("failed", "skipped"):
            notes.append(f"Source '{name}' {info.get('status')}: {info.get('reason', 'unknown')}")
    return notes


def main() -> int:
    """Rebuild the dataset OFFLINE from raw feed copies in data/raw."""
    setup_console_logging()
    cfg = get_config()
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    from dataset import download_data as dl  # local import avoids circularity

    ds = cfg["dataset"]
    rng = random.Random(int(ds["random_seed"]))
    entries: List[Dict[str, Any]] = []
    for fname, parser in (
        ("openphish_feed.txt", dl.parse_openphish),
        ("urlhaus_text.txt", dl.parse_urlhaus),
    ):
        p = raw_dir / fname
        if p.exists():
            entries.extend(parser(p.read_text(encoding="utf-8", errors="replace")))
    pt = raw_dir / "phishtank.csv"
    if pt.exists():
        entries.extend(dl.parse_phishtank_csv(pt))
    for fname, parser, source_name in (
        ("tranco.csv", dl.parse_tranco_csv, "tranco"),
        ("majestic_million.csv", dl.parse_majestic_csv, "majestic"),
    ):
        p = raw_dir / fname
        if p.exists():
            domains = parser(p.read_text(encoding="utf-8", errors="replace"))
            entries.extend(
                dl.construct_legitimate_entries(
                    domains, source_name, int(ds["target_legitimate"]),
                    int(ds["legitimate_domain_pool"]), rng,
                )
            )
    if not entries:
        logger.error("no raw feed files in %s - run 'python -m dataset.download_data' first", raw_dir)
        return 1
    stats = build_from_entries(
        entries, cfg, download_report={"mode": "offline rebuild from data/raw"}
    )
    dl.log_report(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
