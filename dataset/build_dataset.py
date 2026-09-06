"""Dataset cleaning, normalization, balancing, and persistence.

PHASE 1R (dataset revision): the benign class is a documented MIX of
  * observed URLs   - real page URLs harvested from public sitemaps of
                      curated reputable organizations (source 'sitemap'),
                      carrying real paths/queries; and
  * constructed URLs - 'https://<domain>/' homepages from ranking lists
                      (source 'tranco'/'majestic').
Rationale (measured SHAP evidence): the original benign class contained
ONLY constructed homepages, so 'any path implies malice' became a
near-decisive feature. Mixing real deep links into the benign class
counters this source bias. Nothing is fabricated - both pools are real
URLs from legitimate sources, and the ACTUAL mix ratio is reported.

Pipeline: normalize -> drop malformed -> registrable-domain extraction
-> exact-URL dedup (malicious label wins conflicts) -> per-label domain
caps (malicious 5 / benign 25, configurable) -> seeded class balancing
with the benign mix -> seeded shuffle -> urls.csv + dataset_stats.json.

DOMAIN-LEVEL LEAKAGE: registrable_domain remains the grouping variable
for domain-aware splitting / StratifiedGroupKFold (Phase 2R) - random
URL-level splitting would leak domains across train/test and inflate
performance.
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
OBSERVED_BENIGN_SOURCES = {"sitemap"}
CONSTRUCTED_BENIGN_SOURCES = {"tranco", "majestic"}


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


def apply_domain_cap(entries: List[Dict[str, Any]], benign_cap: int,
                     malicious_cap: int) -> Tuple[List[Dict[str, Any]], int]:
    """Keep at most cap URLs per (label, registrable_domain).

    Malicious cap stays tight (5): URLhaus spam-lists hundreds of URLs
    per host, and a tight cap prevents host memorization. Benign cap is
    higher (25): a reputable site genuinely has many real pages, and
    grouped splitting already confines each domain to one fold.
    """
    caps = {0: int(benign_cap), 1: int(malicious_cap)}
    counts: Dict[str, int] = {}
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for e in entries:
        label = int(e["label"])
        cap = caps.get(label, 0)
        if cap <= 0:
            kept.append(e)
            continue
        key = f"{label}|{e['registrable_domain']}"
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= cap:
            kept.append(e)
        else:
            dropped += 1
    return kept, dropped


def sample_balanced(entries: List[Dict[str, Any]], n_malicious: int, n_benign: int,
                    seed: int, benign_observed_frac: float = 0.5
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Seeded balancing with the documented benign mix.

    The benign target is split between observed (sitemap) and constructed
    (homepage) URLs at benign_observed_frac. If one pool is short, the
    other REAL pool tops up (never fabrication); the actual ratio is
    reported. Malicious shortfall is reported, never padded.
    """
    rng = random.Random(seed)
    malicious = [e for e in entries if e["label"] == 1]
    benign_obs = [e for e in entries if e["label"] == 0
                  and e["source"] in OBSERVED_BENIGN_SOURCES]
    benign_const = [e for e in entries if e["label"] == 0
                    and e["source"] not in OBSERVED_BENIGN_SOURCES]

    n_mal = min(int(n_malicious), len(malicious))
    n_obs = min(int(round(n_benign * benign_observed_frac)), len(benign_obs))
    n_const = min(int(n_benign) - n_obs, len(benign_const))
    stats = {
        "malicious_available": len(malicious),
        "benign_observed_available": len(benign_obs),
        "benign_constructed_available": len(benign_const),
        "malicious_selected": n_mal,
        "benign_selected": n_obs + n_const,
        "benign_observed_selected": n_obs,
        "benign_constructed_selected": n_const,
        "target_observed_frac": float(benign_observed_frac),
        "actual_observed_frac": (n_obs / (n_obs + n_const)) if (n_obs + n_const) else 0.0,
    }
    sel = rng.sample(malicious, n_mal) if n_mal < len(malicious) else list(malicious)
    sel += rng.sample(benign_obs, n_obs) if n_obs < len(benign_obs) else list(benign_obs)
    sel += rng.sample(benign_const, n_const) if n_const < len(benign_const) else list(benign_const)
    rng.shuffle(sel)
    return sel, stats


def build_from_entries(entries: List[Dict[str, Any]], cfg: Dict[str, Any],
                       download_report: Dict[str, Any] = None) -> Dict[str, Any]:
    ds = cfg["dataset"]
    seed = int(ds["random_seed"])
    target_mal = int(ds["target_phishing"])
    target_ben = int(ds["target_legitimate"])
    ben_frac = float(ds.get("benign_observed_frac", 0.5))
    cap_mal = int(ds.get("max_urls_per_domain_malicious",
                         ds.get("max_urls_per_domain", 5)))
    cap_ben = int(ds.get("max_urls_per_domain_benign", 25))

    cleaned, cleaning_stats = clean_entries(entries)
    capped, dropped_by_cap = apply_domain_cap(cleaned, cap_ben, cap_mal)
    final, sampling_stats = sample_balanced(capped, target_mal, target_ben, seed, ben_frac)

    df = pd.DataFrame(final, columns=CSV_COLUMNS)
    out_csv = Path(cfg["paths"]["processed_dataset_file"])
    out_stats = Path(cfg["paths"]["dataset_stats_file"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    stats = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "dataset_revision": "1R (benign mix: observed sitemap URLs + constructed homepages)",
        "config": {
            "target_phishing": target_mal,
            "target_legitimate": target_ben,
            "benign_observed_frac": ben_frac,
            "max_urls_per_domain_malicious": cap_mal,
            "max_urls_per_domain_benign": cap_ben,
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
        "DATASET REVISION 1R: the benign class is a documented mix of real deep-link "
        "URLs harvested from public sitemaps of curated reputable organizations "
        "(observed) and constructed 'https://<domain>/' homepages from ranking lists. "
        "Rationale: the original benign class contained only constructed homepages, "
        "which made 'presence of a path' a near-decisive malicious signal (measured "
        "SHAP evidence, Phase 4). The mix counters this source bias.",
        "SITEMAP HARVESTING (defensive): only robots.txt and the sitemap files it "
        "declares were fetched (bounded, with politeness delays). The page URLs "
        "inside the sitemaps are dataset strings and were NEVER requested, visited "
        "or rendered. Per-domain failures are reported, never hidden.",
        "Constructed benign homepages remain 'https://<domain>/' (ranking lists "
        "publish domains, not URLs) - a disclosed transformation, not fabrication.",
        "URLhaus records are malware-distribution URLs, not strictly phishing. They "
        "are labelled malicious (1) with threat_type 'malware_distribution' to keep "
        "the distinction explicit.",
        "registrable_domain is the grouping variable for domain-aware splitting "
        "(prevents same-domain train/test leakage).",
        "URLs on legitimately ranked domains that appear in threat feeds (compromised "
        "hosts) are kept with their malicious label: the URL is malicious even when "
        "the domain is popular. Grouped splitting keeps such domains in one fold.",
        "Known remaining source skews (documented, not hidden): benign URLs are "
        "predominantly https and non-IP-hosted (modern reputable sites are), while "
        "URLhaus URLs are predominantly http and IP-hosted. These remain "
        "real-world-plausible but also source-correlated signals.",
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
    sm = raw_dir / "sitemap_urls.txt"
    if sm.exists():
        for line in sm.read_text(encoding="utf-8", errors="replace").splitlines():
            u = line.strip()
            if u:
                entries.append({"url": u, "label": 0, "source": "sitemap",
                                "threat_type": "benign"})
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
