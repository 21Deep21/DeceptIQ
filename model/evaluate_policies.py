"""Phase C evaluation: profiles + band frozen on validation, then the
ONE-TIME policy comparison on the frozen test split (draft 5.7/5.8).

Flow:
  1. Load the v2 fusion bundle (model_v2/model.joblib) - no retraining.
  2. Precompute the four evidence-state probabilities for train/val/test
     (batched; replay is then exact table lookups).
  3. Build reliability profiles from TRAIN+VAL domains only (frozen).
  4. Select the uncertainty band on validation (frozen).
  5. Run B1/B2/B3/P1 on validation (reference) and TEST (one-time,
     guarded: re-runs refuse unless --force, which is recorded).
  6. Integrity check: test B1 must reproduce the Phase B test[none] row
     bit-for-bit (same states, same threshold).
  7. Write model_v2/policy_evaluation.json + policy_metrics.csv.

Usage:  python -m model.evaluate_policies [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from appconfig import get_config, setup_console_logging
from features.evidence_features import (
    evidence_log_fingerprint,
    evidence_matrix_for_domains,
    load_evidence_matrix,
)
from features.url_utils import is_ip_hostname
from model.acquisition import (
    GROUPS,
    build_profiles,
    load_records,
    precompute_states,
    replay_policy,
    select_band,
    summarize,
)
from model.split import dataset_fingerprint, get_or_create_splits

logger = logging.getLogger(__name__)

CSV_FIELDS = ["policy", "stage", "n", "accuracy", "precision", "recall", "f1",
              "roc_auc", "pr_auc", "fpr", "brier", "ece_10bin",
              "mean_requests", "pct_escalated", "acquisition_success_rate",
              "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]


def _states_for(bundle, df, matrix) -> Dict[str, np.ndarray]:
    domains = [None if is_ip_hostname(str(d)) else str(d)
               for d in df["registrable_domain"]]
    E = evidence_matrix_for_domains(domains, matrix)
    X_url = bundle["url_builder"].transform(df["url"].tolist())

    def predict_fused(Ec):
        X = hstack([X_url, csr_matrix(Ec)], format="csr")
        return bundle["calibrated_model"].predict_proba(X)[:, 1]

    return precompute_states(predict_fused, E)


def main(argv=None) -> int:
    setup_console_logging()
    ap = argparse.ArgumentParser(description="Phase C policy evaluation")
    ap.add_argument("--force", action="store_true",
                    help="re-run the one-time test replay (recorded)")
    args = ap.parse_args(argv)

    cfg = get_config()
    acq = dict(cfg.get("acquisition", {}))
    out_path = Path(cfg["model_v2"]["dir"]) / "policy_evaluation.json"
    if out_path.exists() and not args.force:
        try:
            if json.loads(out_path.read_text()).get("test_evaluated"):
                logger.error("policy test replay already evaluated at %s; "
                             "refusing (one-time discipline). Use --force.",
                             json.loads(out_path.read_text()).get("test_evaluated_at"))
                return 2
        except json.JSONDecodeError:
            logger.warning("policy_evaluation.json unreadable; continuing")

    bundle = joblib.load(cfg["model_v2"]["model_path"])
    threshold = float(bundle["threshold"])
    logger.info("loaded model_v2 bundle: %s @ threshold %.4f "
                "(evidence fingerprint %s...)", bundle["model_name"], threshold,
                bundle.get("evidence_log_fingerprint", "?")[:12])

    train_df, val_df, test_df, split_stats = get_or_create_splits(cfg)
    matrix = load_evidence_matrix(cfg["evidence"]["dir"])
    records = load_records(cfg["evidence"]["dir"])

    states = {"train": _states_for(bundle, train_df, matrix),
              "val": _states_for(bundle, val_df, matrix),
              "test": _states_for(bundle, test_df, matrix)}

    def ctx(df):
        domains = [None if is_ip_hostname(str(d)) else str(d)
                   for d in df["registrable_domain"]]
        return df["url"].tolist(), df["label"].to_numpy(dtype=int), domains

    urls_tr, y_tr, dom_tr = ctx(train_df)
    urls_va, y_va, dom_va = ctx(val_df)
    urls_te, y_te, dom_te = ctx(test_df)

    # ---- profiles from TRAIN+VAL domains only (frozen before test) ------
    tv_domains = set(dom_tr) | set(dom_va)
    tv_domains.discard(None)
    tv_records = {g: {d: r for d, r in records.get(g, {}).items() if d in tv_domains}
                  for g in GROUPS}
    profiles = build_profiles(
        states["train"], y_tr, [d is not None for d in dom_tr],
        states["val"], y_va, [d is not None for d in dom_va],
        tv_records, threshold, weights=acq.get("score_weights"))

    max_req = int(acq.get("max_requests", 2))
    lat_budget = acq.get("latency_budget_ms")
    lat_budget = float(lat_budget) if lat_budget is not None else None
    min_score = float(acq.get("min_group_score", 0.0))

    band, band_summary = select_band(
        states["val"], y_va, dom_va, records, profiles, threshold,
        max_requests=max_req, latency_budget_ms=lat_budget)
    logger.info("frozen band: %s | frozen budgets: max_requests=%s "
                "latency_ms=%s min_score=%s", band, max_req, lat_budget, min_score)

    # ---- run policies: validation (reference), test (one-time) ----------
    results: Dict[str, Dict[str, Any]] = {"validation": {}, "test": {}}
    rows: List[Dict[str, Any]] = []
    for stage, urls, y, doms, st in (("validation", urls_va, y_va, dom_va, states["val"]),
                                     ("test", urls_te, y_te, dom_te, states["test"])):
        logger.info("=== %s (n=%d) ===", stage.upper(), len(y))
        for pol in ("B1", "B2", "B3", "P1"):
            recs = replay_policy(pol, urls, y, doms, st, records, profiles,
                                 threshold, band, max_requests=max_req,
                                 latency_budget_ms=lat_budget,
                                 min_group_score=min_score)
            s = summarize(recs, threshold)
            results[stage][pol] = s
            rows.append({"policy": pol, "stage": stage, **s})
            logger.info("  %-3s f1=%.4f rec=%.4f prec=%.4f fpr=%.4f | "
                        "req=%.3f escal=%.3f succ=%s | p50=%.0f p95=%.0f p99=%.0f ms",
                        pol, s["f1"], s["recall"], s["precision"], s["fpr"],
                        s["mean_requests"], s["pct_escalated"],
                        s["acquisition_success_rate"], s["latency_p50_ms"],
                        s["latency_p95_ms"], s["latency_p99_ms"])

    # ---- integrity: test B1 must equal the Phase B test[none] row -------
    integrity = None
    meta_path = Path(cfg["model_v2"]["metadata_path"])
    if meta_path.exists():
        pb = json.loads(meta_path.read_text()).get("test_results", {}) \
                                             .get("none", {}).get("metrics", {})
        if pb:
            delta = abs(results["test"]["B1"]["f1"] - float(pb["f1"]))
            integrity = delta <= 1e-9
            (logger.info if integrity else logger.warning)(
                "integrity %s: test B1 f1 %.4f vs Phase B test[none] %.4f "
                "(delta %.2e)", "OK" if integrity else "MISMATCH",
                results["test"]["B1"]["f1"], float(pb["f1"]), delta)

    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "created_at": stamp,
        "band": list(band),
        "band_selection": "validation P1 F1 maximization over grid "
                          "(0.01/0.02/0.05 x 0.75/0.85/0.95), ties -> fewer requests",
        "band_validation_summary": band_summary,
        "profiles": {g: p.__dict__ for g, p in profiles.items()},
        "budgets": {"max_requests": max_req, "latency_budget_ms": lat_budget,
                    "min_group_score": min_score},
        "results": results,
        "integrity_b1_matches_phaseB_test_none": integrity,
        "provenance": {
            "evidence_log_fingerprint": evidence_log_fingerprint(cfg["evidence"]["dir"]),
            "dataset_fingerprint": dataset_fingerprint(train_df),
            "model_v2_created_at": bundle.get("created_at"),
            "threshold": threshold,
            "split_rows": split_stats["rows"],
        },
        "approximations": [
            "sigmoid calibrator fitted on the full-evidence condition is "
            "applied to masked states as well",
            "latency = simulated acquisition time only (recorded durations)",
            "group-level value profiles; per-instance value is future work",
            "budget eligibility uses profile p95; the recorded duration is charged",
        ],
        "test_evaluated": True, "test_evaluated_at": stamp,
    }, indent=2, default=str), encoding="utf-8")

    csv_path = Path(cfg["model_v2"]["dir"]) / "policy_metrics.csv"
    pd.DataFrame(rows)[CSV_FIELDS].to_csv(csv_path, index=False)
    logger.info("wrote %s and %s", out_path, csv_path)

    t, v = results["test"], results["validation"]
    logger.info("=" * 70)
    logger.info("MAIN COMPARISON (test, one-time): does P1 cut requests/latency "
                "vs B3/B2 without material recall loss?")
    for pol in ("B1", "B2", "B3", "P1"):
        logger.info("  %-3s recall=%.4f f1=%.4f | mean_req=%.3f p95=%.0fms "
                    "escal=%.1f%%", pol, t[pol]["recall"], t[pol]["f1"],
                    t[pol]["mean_requests"], t[pol]["latency_p95_ms"],
                    100 * t[pol]["pct_escalated"])
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
