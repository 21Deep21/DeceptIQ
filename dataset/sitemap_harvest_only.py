"""Standalone Phase 1R sitemap harvest (network, ~10-30 minutes).

Fetches ONLY robots.txt + sitemap files for the curated domains in
config.yaml and saves the harvested page URLs to
data/raw/sitemap_urls.txt. Page URLs are never visited.
Run this when raw feeds are already cached; then rebuild the dataset
OFFLINE with:  python -m dataset.build_dataset
"""

from pathlib import Path

import requests

from appconfig import get_config, setup_console_logging
from dataset.sitemap_sources import collect_sitemap_urls


def main() -> int:
    setup_console_logging()
    cfg = get_config()
    session = requests.Session()
    session.headers["User-Agent"] = cfg["network"]["user_agent"]

    entries, report = collect_sitemap_urls(cfg, session)
    out = Path("data/raw/sitemap_urls.txt")
    out.write_text("\n".join(e["url"] for e in entries) + "\n", encoding="utf-8")

    print("=" * 60)
    print(f"saved {len(entries)} URLs to {out}")
    print(f"domains ok/failed: {report['domains_ok']}/{report['domains_failed']}"
          f"  sitemap files fetched: {report['sitemaps_fetched']}")
    if report["failures"]:
        print("domains yielding nothing (first 20):",
              ", ".join(report["failures"][:20]))
    print("=" * 60)
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
