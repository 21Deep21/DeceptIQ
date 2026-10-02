"""One-time logged evidence acquisition (v2.0 Phase A).

For every unique registrable domain in urls.csv (IP-literal hosts skipped
— domain evidence does not apply), query each configured passive group
(DNS, RDAP) ONCE and append a JSONL record with duration and status.
Policies (Phase C/D) are later evaluated by REPLAYING these recorded
costs, so this log IS the operational measurement (draft section 6).

Resume-capable: domains already present in a group's log are skipped, so
interrupted runs continue where they stopped. Progress lines every N
domains (lesson learned from the sitemap harvest). Run the full pass in
the background (nohup) — expect hours, not minutes.

Usage:
    python -m dataset.collect_evidence --limit 30     # pilot
    python -m dataset.collect_evidence                # full pass (background!)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

from appconfig import get_config, setup_console_logging
from features.dns_features import query_dns
from features.rdap_features import query_rdap, rdap_response_sha256
from features.url_utils import is_ip_hostname

logger = logging.getLogger(__name__)

QueryFn = Callable[[str], Tuple[str, Optional[Dict[str, Any]], Optional[str]]]
# returns (status, fields_or_None, response_sha256_or_None)


def _load_done(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["domain"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def collect_evidence(domains: List[str], queries: Dict[str, QueryFn], out_dir: Path,
                     delay: float = 0.5, progress_every: int = 50,
                     limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Acquire+log evidence per group. Never raises; resume-capable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Any]] = {}
    for group, qfn in queries.items():
        path = out_dir / f"{group}.jsonl"
        done = _load_done(path)
        todo = [d for d in domains if d not in done]
        resumed_past = len(domains) - len(todo)
        if limit is not None:
            todo = todo[:limit]
        statuses: Dict[str, int] = {}
        logger.info("group %s: %d domains total | %d already collected (resume) "
                    "| %d to acquire now", group, len(domains), resumed_past, len(todo))
        for i, dom in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                status, fields, sha = qfn(dom)
            except Exception as exc:  # the collector must never die mid-run
                status, fields, sha = "error", None, None
                logger.debug("query failed for %s: %s", dom, exc)
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            record = {
                "domain": dom, "group": group,
                "requested_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
                "duration_ms": duration_ms, "status": status,
                "fields": fields, "response_sha256": sha, "collector": "evidence/1",
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            statuses[status] = statuses.get(status, 0) + 1
            if progress_every and i % progress_every == 0:
                logger.info("%s progress: %d/%d this run (statuses: %s)",
                            group, i, len(todo), statuses)
            if delay:
                time.sleep(delay)
        summary[group] = {"attempted": len(todo), "statuses": statuses,
                          "resumed_past": resumed_past}
    return summary


def _make_dns_query(timeout: float) -> QueryFn:
    def q(domain: str):
        info = query_dns(domain, timeout=timeout)
        any_present = any(v is not None for v in
                          (info.a_record_count, info.ns_count, info.has_mx))
        if any_present:
            fields = {"a_record_count": info.a_record_count,
                      "a_record_ttl": info.a_record_ttl,
                      "has_mx": info.has_mx,
                      "ns_count": info.ns_count,
                      "ns_diversity": info.ns_diversity}
            return "ok", fields, None
        err = info.error or ""
        if "NXDOMAIN" in err:
            return "not_found", None, None
        if "Timeout" in err:
            return "timeout", None, None
        return "error", None, None
    return q


def _make_rdap_query(timeout: float, session: requests.Session) -> QueryFn:
    def q(domain: str):
        info = query_rdap(domain, timeout=timeout, session=session)
        if info.status_code == "ok":
            fields = {"registrar": info.registrar,
                      "creation_date": info.creation_date,
                      "expiration_date": info.expiration_date,
                      "status": info.status,
                      "nameservers": info.nameservers}
            return "ok", fields, rdap_response_sha256(info)
        return info.status_code or "error", None, None
    return q


def main(argv=None) -> int:
    setup_console_logging()
    ap = argparse.ArgumentParser(
        description="Phase A evidence acquisition (one-time, logged, resume-capable)")
    ap.add_argument("--groups", default="dns,rdap")
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot mode: only first N pending domains per group")
    ap.add_argument("--delay", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = get_config()
    ecfg = cfg.get("evidence", {})
    out_dir = Path(ecfg.get("dir", "data/processed/evidence"))
    delay = (args.delay if args.delay is not None
             else float(ecfg.get("request_delay_seconds", 0.5)))
    progress = int(ecfg.get("progress_every", 50))
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    df = pd.read_csv(cfg["paths"]["processed_dataset_file"])
    domains = sorted({str(d) for d in df["registrable_domain"].dropna()
                      if not is_ip_hostname(str(d))})
    logger.info("evidence acquisition: %d unique registrable domains "
                "(IP-literal hosts excluded by design)", len(domains))

    session = requests.Session()
    session.headers["User-Agent"] = cfg["network"]["user_agent"]
    queries: Dict[str, QueryFn] = {}
    if "dns" in groups:
        queries["dns"] = _make_dns_query(float(ecfg.get("dns_timeout", 5)))
    if "rdap" in groups:
        queries["rdap"] = _make_rdap_query(float(ecfg.get("rdap_timeout", 8)), session)
    unknown = set(groups) - set(queries)
    if unknown:
        raise SystemExit(f"unknown evidence groups: {sorted(unknown)}")

    summary = collect_evidence(domains, queries, out_dir, delay=delay,
                               progress_every=progress, limit=args.limit)
    for group, s in summary.items():
        logger.info("%s done: attempted=%d statuses=%s", group, s["attempted"], s["statuses"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
