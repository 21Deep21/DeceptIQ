"""Evidence-derived features and missingness indicators (v2.0 Phase B).

Builds the fusion feature block from the FROZEN Phase A evidence logs
(data/processed/evidence/{dns,rdap}.jsonl; read-only; LAST record per
domain wins - the re-attempt rule from Phase A).

DESIGN (research draft sections 5.4/5.5, failure handling):
  Three mutually exclusive indicators per group distinguish DOMAIN FACTS
  from ACQUISITION FAILURES:
    ok        -> evidence acquired; numeric features populated
    not_found -> domain-level fact (NXDOMAIN / no registry record);
                 itself potential signal, kept distinct
    failed    -> timeout / rate_lated / error: OUR-side failure, no
                 information about the domain (rate_limited is a pacing
                 artifact per the Phase A patch, so it maps here)
  All indicators zero + numerics zero -> evidence NOT acquired (IP-literal
  hosts, masked training variants, un-escalated runtime cases).
  Numeric features are zero-filled when unavailable; indicators let the
  model distinguish 'missing' from 'actually zero'.

Derived ages use the log's requested_at timestamp as reference
(operational honesty) and are clamped to documented ranges against parse
garbage. Known impurities (documented, accepted): ok-with-unparseable
dates contribute age 0; has_mx maps unknown (None) and authoritative
no (False) to the same 0 at this granularity.

Registrar identity is deliberately NOT a model feature (high-cardinality
target-encoding leakage risk at this scale); it remains display evidence
in the logs. TEMPORAL CAVEAT (recorded in model_v2 metadata): evidence
was acquired after the feed snapshot, so not_found partly reflects
phishing-infrastructure takedown lag and may overstate runtime value.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EVIDENCE_FEATURE_NAMES: Tuple[str, ...] = (
    # DNS group (8)
    "dns_ok", "dns_not_found", "dns_failed",
    "dns_a_record_count", "dns_a_record_ttl", "dns_has_mx",
    "dns_ns_count", "dns_ns_diversity",
    # RDAP group (6)
    "rdap_ok", "rdap_not_found", "rdap_failed",
    "rdap_domain_age_days", "rdap_days_to_expiry", "rdap_nameserver_count",
)

N_EVIDENCE_FEATURES = len(EVIDENCE_FEATURE_NAMES)
GROUPS = ("dns", "rdap")
GROUP_COLUMNS: Dict[str, Tuple[int, int]] = {"dns": (0, 8), "rdap": (8, 14)}

_FAILED_STATUSES = {"timeout", "rate_limited", "error"}
_MAX_AGE_DAYS = 12000      # ~33 years: guard against parse garbage
_MAX_EXPIRY_DAYS = 36500   # ~100 years


def _status_indicators(status: Optional[str]) -> Tuple[int, int, int]:
    if status == "ok":
        return 1, 0, 0
    if status == "not_found":
        return 0, 1, 0
    if status in _FAILED_STATUSES:
        return 0, 0, 1
    return 0, 0, 0          # never queried / unknown -> all zero


def _parse_iso(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _clamp(v, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _dns_vector(rec: dict) -> List[float]:
    ok, nf, fail = _status_indicators(rec.get("status"))
    f = rec.get("fields") or {}
    if not ok:
        return [ok, nf, fail, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [ok, nf, fail,
            float(f.get("a_record_count") or 0),
            float(f.get("a_record_ttl") or 0),
            float(1 if f.get("has_mx") else 0),
            float(f.get("ns_count") or 0),
            float(f.get("ns_diversity") or 0)]


def _rdap_vector(rec: dict) -> List[float]:
    ok, nf, fail = _status_indicators(rec.get("status"))
    if not ok:
        return [ok, nf, fail, 0.0, 0.0, 0.0]
    f = rec.get("fields") or {}
    ref = _parse_iso(rec.get("requested_at"))
    created = _parse_iso(f.get("creation_date"))
    expires = _parse_iso(f.get("expiration_date"))
    age = float(_clamp((ref - created).days, 0, _MAX_AGE_DAYS)) if (ref and created) else 0.0
    expiry = float(_clamp((expires - ref).days, 0, _MAX_EXPIRY_DAYS)) if (ref and expires) else 0.0
    ns = float(len(f.get("nameservers") or []))
    return [ok, nf, fail, age, expiry, ns]


def _read_latest(log_path: Path) -> Dict[str, dict]:
    """Last record per domain (the Phase A re-attempt rule)."""
    latest: Dict[str, dict] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "domain" in rec:
            latest[rec["domain"]] = rec
    return latest


def load_evidence_matrix(log_dir) -> Dict[str, np.ndarray]:
    """domain -> 14-dim evidence feature vector (last record wins)."""
    log_dir = Path(log_dir)
    dns = _read_latest(log_dir / "dns.jsonl")
    rdap = _read_latest(log_dir / "rdap.jsonl")
    out: Dict[str, np.ndarray] = {}
    for domain in set(dns) | set(rdap):
        out[domain] = np.asarray(_dns_vector(dns.get(domain, {}))
                                 + _rdap_vector(rdap.get(domain, {})),
                                 dtype=np.float64)
    logger.info("evidence matrix loaded: %d domains x %d features "
                "(logs: %s)", len(out), N_EVIDENCE_FEATURES, log_dir)
    return out


def evidence_matrix_for_domains(domains, matrix: Dict[str, np.ndarray]) -> np.ndarray:
    """(n, 14); domains absent from the logs -> all zeros (IP-literal
    hosts / never-queried domains -> the 'not acquired' state)."""
    zero = np.zeros(N_EVIDENCE_FEATURES, dtype=np.float64)
    domains = list(domains)
    if not domains:
        return np.zeros((0, N_EVIDENCE_FEATURES), dtype=np.float64)
    return np.vstack([matrix.get(str(d), zero) for d in domains])


def mask_groups(E: np.ndarray, groups: Iterable[str]) -> np.ndarray:
    """Copy with the named groups' columns zeroed (training-time controlled
    feature-group masking). The masked state is IDENTICAL to 'evidence not
    acquired' - one unified representation for all unavailability causes."""
    out = np.array(E, dtype=np.float64, copy=True)
    for g in groups:
        if g not in GROUP_COLUMNS:
            raise ValueError(f"unknown evidence group {g!r}")
        lo, hi = GROUP_COLUMNS[g]
        out[:, lo:hi] = 0.0
    return out


def evidence_log_fingerprint(log_dir) -> str:
    """sha256 over the frozen log files' bytes - recorded in the model_v2
    bundle so the fusion model is bound to its exact evidence snapshot."""
    log_dir = Path(log_dir)
    h = hashlib.sha256()
    for name in sorted(p.name for p in log_dir.glob("*.jsonl")):
        h.update(name.encode("utf-8"))
        h.update((log_dir / name).read_bytes())
    return h.hexdigest()
