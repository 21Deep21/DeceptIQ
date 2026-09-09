"""Daily OpenPhish feed accumulation (v1.1, improvement #1).

WHY: the free OpenPhish feed is a ~300-URL snapshot of CURRENTLY active
phishing URLs; the deployed dataset captured a single snapshot (only 57
true phishing URLs = 1.1% of the malicious class). Accumulating one
snapshot per day builds a real phishing corpus over weeks - exact
duplicates across days are removed by clean_entries at build time.

SCOPE/SAFETY: fetches ONLY the feed file (page URLs are never visited)
and stores it under data/raw/openphish_YYYYMMDD.txt (UTC date). No
backfill (the free feed has no history). Honest failure: if the feed is
unreachable, nothing is written and exit code 1 is returned.

Run manually:      python -m dataset.accumulate_feeds [--force]
Cron (09:00 UTC):  0 9 * * * cd ~/projects/phishing-url-analyzer && \
    .venv/bin/python -m dataset.accumulate_feeds >> logs/accumulate.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from appconfig import get_config, setup_console_logging

logger = logging.getLogger("app.accumulate")

_DATED_RE = re.compile(r"openphish_\d{8}\.txt$")


def snapshot_path(raw_dir: Path, day: dt.date) -> Path:
    """Dated snapshot path: data/raw/openphish_YYYYMMDD.txt (UTC date)."""
    return raw_dir / f"openphish_{day.strftime('%Y%m%d')}.txt"


def accumulate(cfg: Dict[str, Any], force: bool = False,
               session: Optional[requests.Session] = None
               ) -> Tuple[int, Optional[Path]]:
    """Fetch today's feed snapshot. Returns (exit_code, path_or_None)."""
    from dataset.download_data import _get_text, _new_session, parse_openphish

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date()
    out = snapshot_path(raw_dir, today)

    if out.exists() and not force:
        logger.info("snapshot for %s already exists (%d bytes) - keeping it "
                    "(use --force to refetch)", today, out.stat().st_size)
        return 0, out

    if session is None:
        session = _new_session(cfg)
    try:
        text = _get_text(session, cfg["sources"]["openphish_feed_url"], cfg)
    except Exception as exc:
        logger.error("OpenPhish feed unavailable (%s) - nothing written", exc)
        return 1, None

    entries = parse_openphish(text)
    if not entries:
        logger.error("feed parsed to 0 URLs - nothing written (honest failure)")
        return 1, None
    out.write_text(text, encoding="utf-8")

    snapshots = [p for p in sorted(raw_dir.glob("openphish_*.txt"))
                 if _DATED_RE.match(p.name)]
    total = sum(len(parse_openphish(p.read_text(encoding="utf-8", errors="replace")))
                for p in snapshots)
    logger.info("saved %s (%d raw URLs); %d dated snapshots on disk, %d accumulated "
                "raw URLs total (duplicates removed at build time)",
                out.name, len(entries), len(snapshots), total)
    return 0, out


def main(argv=None) -> int:
    setup_console_logging()
    p = argparse.ArgumentParser(description="Daily OpenPhish snapshot accumulation")
    p.add_argument("--force", action="store_true",
                   help="refetch even if today's snapshot exists")
    args = p.parse_args(argv)
    code, _ = accumulate(get_config(), force=args.force)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
