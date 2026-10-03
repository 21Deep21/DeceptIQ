"""Acquisition controller and policies (v2.0 Phase C).

Replay-based evaluation per the frozen-costs design (Phase A): NO live
network requests during evaluation. With two evidence groups there are
exactly four possible evidence states per URL (none/dns/rdap/full); the
evaluator precomputes all four probability vectors in one batched pass
over the fusion model, and every policy replay is a state machine over
those probabilities plus the RECORDED per-domain costs and outcomes from
the frozen evidence logs. A replayed acquisition IS the log record: its
status becomes the evidence state's content, its duration_ms is charged.

Policies (research draft section 5.7):
  B1 url_only     - never escalates (fastest, lowest cost)
  B2 all_features - acquires every applicable group for every URL
  B3 cascade      - acquires the FIXED second-stage set (all groups) for
                    uncertain URLs only (confidence-gating baseline)
  P1 reliability  - for uncertain URLs, acquires groups one at a time in
                    reliability-score order, re-checking confidence after
                    each; respects request and latency budgets and the
                    minimum expected-value floor

Uncertainty (draft 5.3): a URL is UNCERTAIN when its URL-only calibrated
probability lies inside [low, high]; outside the band the local decision
is returned. Stopping: confident / budget exhausted / no eligible group /
score below the expected-value floor.

Reliability profile (draft 5.4): availability, p95/p99 from RECORDED
durations; value = mean |dp| and dF1 on validation; stability = train-vs-
validation F1-gain consistency (two-split measure; per-instance value and
richer splits are future work). score = value*availability*stability /
(p95_seconds + eps), exponents configurable (Phase D ablation hooks).

Approximations (documented): the sigmoid calibrator was fitted on the
full-evidence condition and is applied to masked states as well; latency
accounts for acquisition time only; budget eligibility uses profile p95
while the actual recorded duration is charged (conservative rule).

SHAP is NOT involved in acquisition (draft 5.6).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss

from features.evidence_features import mask_groups
from model.finalize import expected_calibration_error
from model.train import compute_metrics

logger = logging.getLogger(__name__)

GROUPS = ("dns", "rdap")
CONDITIONS = ("none", "dns_only", "rdap_only", "full")
_STATE_MASKS = {"none": ("dns", "rdap"), "dns_only": ("rdap",),
                "rdap_only": ("dns",), "full": ()}


def load_records(log_dir) -> Dict[str, Dict[str, dict]]:
    """{group: {domain: record}} from the frozen logs; LAST record wins."""
    out: Dict[str, Dict[str, dict]] = {g: {} for g in GROUPS}
    for g in GROUPS:
        p = Path(log_dir) / f"{g}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "domain" in rec:
                out[g][rec["domain"]] = rec
    return out


def precompute_states(predict_fused, E: np.ndarray) -> Dict[str, np.ndarray]:
    """The four evidence-state probability vectors for a fixed URL set.

    predict_fused(Ec) must return p(class=1) for the fused matrix built
    from the URL block and the (possibly masked) evidence block Ec.
    """
    states: Dict[str, np.ndarray] = {}
    for cond in CONDITIONS:
        masks = _STATE_MASKS[cond]
        Ec = mask_groups(E, masks) if masks else np.array(E, copy=True)
        states[cond] = np.asarray(predict_fused(Ec), dtype=np.float64)
    return states


@dataclass(frozen=True)
class GroupProfile:
    group: str
    availability: float
    p95_ms: float
    p99_ms: float
    mean_abs_dp: float        # validation: mean |p(group) - p(none)|, domain-hosted
    f1_gain_val: float
    f1_gain_train: float
    stability: float          # 1 - |gain_val - gain_train| / max(|gains|, eps)
    score: float


def _f1_at(y, p, thr: float) -> float:
    m = compute_metrics(np.asarray(y), (np.asarray(p) >= thr).astype(int), p)
    return float(m["f1"])


def build_profiles(states_train, y_train, hosted_train,
                   states_val, y_val, hosted_val,
                   records: Dict[str, Dict[str, dict]], threshold: float,
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, GroupProfile]:
    """Reliability profiles from TRAIN+VALIDATION data only (frozen before
    any test replay - draft section 5.4). `records` must already be
    restricted to train+val domains by the caller."""
    w = weights or {}
    wv = float(w.get("value", 1.0)); wa = float(w.get("availability", 1.0))
    ws = float(w.get("stability", 1.0)); wl = float(w.get("latency", 1.0))
    hv = np.asarray(hosted_val, dtype=bool)
    ht = np.asarray(hosted_train, dtype=bool)
    profiles: Dict[str, GroupProfile] = {}
    for g in GROUPS:
        cond = "dns_only" if g == "dns" else "rdap_only"
        recs = list(records.get(g, {}).values())
        n = len(recs)
        availability = (sum(1 for r in recs if r.get("status") == "ok") / n) if n else 0.0
        durs = sorted(float(r.get("duration_ms") or 0.0) for r in recs)
        p95 = durs[max(0, int(0.95 * len(durs)) - 1)] if durs else 0.0
        p99 = durs[max(0, int(0.99 * len(durs)) - 1)] if durs else 0.0

        dp = np.abs(states_val[cond] - states_val["none"])[hv]
        mean_abs_dp = float(dp.mean()) if dp.size else 0.0
        gain_v = (_f1_at(y_val[hv], states_val[cond][hv], threshold)
                  - _f1_at(y_val[hv], states_val["none"][hv], threshold))
        gain_t = (_f1_at(y_train[ht], states_train[cond][ht], threshold)
                  - _f1_at(y_train[ht], states_train["none"][ht], threshold))
        stability = max(0.0, 1.0 - abs(gain_v - gain_t)
                        / max(abs(gain_v), abs(gain_t), 1e-6))

        score = (((mean_abs_dp + 1e-9) ** wv)
                 * (max(availability, 1e-9) ** wa)
                 * (max(stability, 1e-9) ** ws)
                 / (((p95 / 1000.0) + 1e-6) ** wl))
        profiles[g] = GroupProfile(g, availability, p95, p99, mean_abs_dp,
                                   gain_v, gain_t, stability, score)
        logger.info("profile %s: avail=%.3f p95=%.0fms |dp|=%.4f gain(val)=%+.4f "
                    "gain(train)=%+.4f stab=%.3f score=%.4f",
                    g, availability, p95, mean_abs_dp, gain_v, gain_t,
                    stability, score)
    return profiles


@dataclass
class DecisionRecord:
    url: str
    label: int
    domain: Optional[str]
    escalated: bool
    verdict: str
    final_p: float
    final_condition: str
    requests: List[Dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0


def _state_name(acquired: set) -> str:
    d, r = "dns" in acquired, "rdap" in acquired
    return "full" if d and r else "dns_only" if d else "rdap_only" if r else "none"


def _confident(p: float, band: Sequence[float]) -> bool:
    return p < band[0] or p > band[1]


def _acquire(group: str, domain: Optional[str], records, profiles) -> Dict[str, Any]:
    """Replay one acquisition: the recorded outcome and cost are the truth.
    A domain absent from the logs (not expected; coverage was 100%) is
    charged the profile p95 as a conservative fallback with status error."""
    rec = records.get(group, {}).get(domain)
    if rec is not None:
        return {"group": group, "status": rec.get("status"),
                "duration_ms": float(rec.get("duration_ms") or 0.0)}
    return {"group": group, "status": "error",
            "duration_ms": float(profiles[group].p95_ms)}


def replay_policy(policy: str, urls, labels, domains, states, records, profiles,
                  threshold: float, band: Sequence[float],
                  max_requests: int = 2, latency_budget_ms: Optional[float] = None,
                  min_group_score: float = 0.0) -> List[DecisionRecord]:
    """Execute one policy over a split by replay. Pure; no network."""
    if policy not in ("B1", "B2", "B3", "P1"):
        raise ValueError(f"unknown policy {policy!r}")
    order = sorted(GROUPS, key=lambda g: -profiles[g].score)  # stable: dns first on ties
    out: List[DecisionRecord] = []
    for i, url in enumerate(urls):
        domain = domains[i]
        hosted = domain is not None
        p0 = float(states["none"][i])
        acquired: set = set()
        requests: List[Dict[str, Any]] = []
        charged = 0.0

        if policy == "B1":
            want: List[str] = []
        elif policy == "B2":
            want = list(order) if hosted else []
        else:  # B3, P1: uncertainty-gated
            want = list(order) if (hosted and not _confident(p0, band)) else []

        if policy in ("B2", "B3"):
            for g in want:
                q = _acquire(g, domain, records, profiles)
                requests.append(q); charged += q["duration_ms"]; acquired.add(g)
        elif policy == "P1":
            for g in want:
                if len(requests) >= max_requests:
                    break
                if g in acquired or profiles[g].score < min_group_score:
                    continue
                if (latency_budget_ms is not None
                        and (latency_budget_ms - charged) < profiles[g].p95_ms):
                    break  # conservative: a request that may not fit is not made
                q = _acquire(g, domain, records, profiles)
                requests.append(q); charged += q["duration_ms"]; acquired.add(g)
                if _confident(float(states[_state_name(acquired)][i]), band):
                    break

        p = float(states[_state_name(acquired)][i])
        out.append(DecisionRecord(url=str(url), label=int(labels[i]), domain=domain,
                                  escalated=len(requests) > 0,
                                  verdict="phishing" if p >= threshold else "safe",
                                  final_p=p, final_condition=_state_name(acquired),
                                  requests=requests, total_latency_ms=charged))
    return out


def summarize(recs: List[DecisionRecord], threshold: float) -> Dict[str, Any]:
    """Draft section 5.8 metrics: classification + calibration + operational."""
    y = np.asarray([r.label for r in recs], dtype=int)
    p = np.asarray([r.final_p for r in recs], dtype=float)
    m = compute_metrics(y, (p >= threshold).astype(int), p)
    out: Dict[str, Any] = {k: (round(float(v), 4) if isinstance(v, float) else v)
                           for k, v in m.items()}
    fp, tn = m["fp"], m["tn"]
    out["fpr"] = round(fp / (fp + tn), 4) if (fp + tn) else 0.0
    out["brier"] = round(float(brier_score_loss(y, p)), 4)
    out["ece_10bin"] = round(float(expected_calibration_error(y, p)), 4)

    n = len(recs)
    total_reqs = sum(len(r.requests) for r in recs)
    ok_reqs = sum(1 for r in recs for q in r.requests if q["status"] == "ok")
    lat = sorted(r.total_latency_ms for r in recs)
    pct = lambda q: (lat[max(0, int(q * len(lat)) - 1)] if lat else 0.0)
    out.update({
        "mean_requests": round(total_reqs / n, 4) if n else 0.0,
        "pct_escalated": round(sum(1 for r in recs if r.escalated) / n, 4) if n else 0.0,
        "acquisition_success_rate": (round(ok_reqs / total_reqs, 4)
                                     if total_reqs else None),
        "latency_p50_ms": round(pct(0.50), 1), "latency_p95_ms": round(pct(0.95), 1),
        "latency_p99_ms": round(pct(0.99), 1),
        "n": n,
    })
    return out


def select_band(states, labels, domains, records, profiles, threshold: float,
                grid: Optional[List[Tuple[float, float]]] = None,
                max_requests: int = 2,
                latency_budget_ms: Optional[float] = None
                ) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    """Coarse validation sweep: pick the band maximizing P1 validation F1
    (ties -> fewer mean requests). Frozen before any test replay."""
    grid = grid or [(lo, hi) for lo in (0.01, 0.02, 0.05)
                    for hi in (0.75, 0.85, 0.95)]
    best_key, best_band, best_summary = None, None, None
    for lo, hi in grid:
        recs = replay_policy("P1", labels=labels, urls=domains, domains=domains,
                             states=states, records=records, profiles=profiles,
                             threshold=threshold, band=(lo, hi),
                             max_requests=max_requests,
                             latency_budget_ms=latency_budget_ms)
        s = summarize(recs, threshold)
        key = (s["f1"], -s["mean_requests"])
        logger.info("band sweep (%.2f, %.2f): f1=%.4f mean_req=%.4f "
                    "escalated=%.3f", lo, hi, s["f1"], s["mean_requests"],
                    s["pct_escalated"])
        if best_key is None or key > best_key:
            best_key, best_band, best_summary = key, (lo, hi), s
    logger.info("band selected on validation: %s (f1=%.4f, mean_req=%.4f)",
                best_band, best_summary["f1"], best_summary["mean_requests"])
    return best_band, best_summary
