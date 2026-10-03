"""Phase B finalization: calibrate, threshold, ONE-TIME test, persist model_v2.

Mirrors the v1 finalize discipline for the fusion model:
  1. Rebuild the same deterministic training state as train_v2
     (seeded splits/fingerprint, seeded augmentation, RF).
  2. Sigmoid calibration on the validation split, FULL-EVIDENCE condition
     (the controller's escalation endpoint). FrozenEstimator pattern:
     the calibrator sees only held-out rows.
  3. Recall-first threshold on validation (same policy as v1, from config).
  4. ONE-TIME test evaluation under BOTH conditions (full = primary
     deployment condition; none = URL-only fallback), plus the
     domain-hosted subgroup where evidence applies. Guarded: re-runs
     refuse (exit 2) unless --force (recorded).
  5. Persist model_v2/{model.joblib, metadata.json, threshold.json,
     model_metrics.csv}. The bundle records the evidence-log fingerprint
     (bound to its exact evidence snapshot) and an explicit INPUT CONTRACT
     (see bundle['input_contract']).

Usage:  python -m model.finalize_v2 [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from appconfig import get_config, setup_console_logging
from features.evidence_features import (
    EVIDENCE_FEATURE_NAMES,
    evidence_log_fingerprint,
    evidence_matrix_for_domains,
    load_evidence_matrix,
    mask_groups,
)
from features.feature_extraction import FEATURE_NAMES
from features.url_utils import is_ip_hostname
from model.finalize import expected_calibration_error, select_threshold
from model.split import dataset_fingerprint, get_or_create_splits
from model.train import compute_metrics
from model.train_v2 import CONDITIONS, _CONDITION_MASKS, _domain_hosted_mask, augment_training, fusion_matrix

logger = logging.getLogger(__name__)

SELECTION_RATIONALE = (
    "Fusion model (v2.0 Phase B): RandomForest (the documented v1 selection; "
    "no new model comparison - the research contribution is the evidence-"
    "aware feature space, not classifier choice) trained on URL features "
    "(26 dense + char 3-5-gram TF-IDF) + 14 evidence features (DNS/RDAP "
    "derived + ok/not_found/failed missingness indicators), with 4-condition "
    "group-masking augmentation so one predictor handles any acquisition "
    "outcome. TEMPORAL CAVEAT (measured, documented): evidence was acquired "
    "after the feed snapshot, so not_found partly reflects phishing-"
    "infrastructure takedown lag; full-evidence results are an upper bound "
    "for runtime value. Threshold selected on the FULL-EVIDENCE validation "
    "condition; the masked fast path inherits it (Phase C may revisit with "
    "condition-specific thresholds)."
)


def _metrics_row(system: str, stage: str, m: Dict[str, Any], n: int) -> Dict[str, Any]:
    return {"system": system, "stage": stage, "n": n,
            **{k: m[k] for k in ("accuracy", "precision", "recall", "f1",
                                 "roc_auc", "pr_auc", "tn", "fp", "fn", "tp")}}


def main(argv=None) -> int:
    setup_console_logging()
    ap = argparse.ArgumentParser(description="Phase B: finalize fusion model v2")
    ap.add_argument("--force", action="store_true",
                    help="re-run the one-time test evaluation (recorded)")
    args = ap.parse_args(argv)

    cfg = get_config()
    seed = int(cfg["dataset"]["random_seed"])
    m2 = cfg["model_v2"]
    bundle_path = Path(m2["model_path"])
    metadata_path = Path(m2["metadata_path"])

    if metadata_path.exists() and not args.force:
        try:
            if json.loads(metadata_path.read_text()).get("test_evaluated"):
                logger.error("v2 final test already evaluated at %s; refusing "
                             "(one-time discipline). Use --force.",
                             json.loads(metadata_path.read_text()).get("test_evaluated_at"))
                return 2
        except json.JSONDecodeError:
            logger.warning("model_v2 metadata unreadable; continuing")

    t0 = time.time()
    train_df, val_df, test_df, split_stats = get_or_create_splits(cfg)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    matrix = load_evidence_matrix(cfg["evidence"]["dir"])
    ev_fp = evidence_log_fingerprint(cfg["evidence"]["dir"])
    E_train = evidence_matrix_for_domains(train_df["registrable_domain"], matrix)
    E_val = evidence_matrix_for_domains(val_df["registrable_domain"], matrix)
    E_test = evidence_matrix_for_domains(test_df["registrable_domain"], matrix)

    from features.feature_builder import URLFeatureBuilder
    builder = URLFeatureBuilder().fit(train_df["url"].tolist())
    Xurl_train = builder.transform(train_df["url"].tolist())
    Xurl_val = builder.transform(val_df["url"].tolist())
    Xurl_test = builder.transform(test_df["url"].tolist())

    logger.info("training fusion RF (deterministic, same state as train_v2)...")
    Xaug, yaug, _c, _o = augment_training(Xurl_train, E_train, y_train, seed=seed)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(Xaug, yaug)

    X_val_full = fusion_matrix(builder, val_df["url"].tolist(), E_val)
    calibrated = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid")
    calibrated.fit(X_val_full, y_val)
    val_proba = np.asarray(calibrated.predict_proba(X_val_full))[:, 1]

    policy = cfg["model"]["threshold_policy"]
    thr = select_threshold(y_val, val_proba,
                           float(policy.get("target_recall", 0.97)),
                           float(policy.get("min_precision", 0.90)))
    logger.info("threshold selection (full-evidence validation): "
                "threshold=%.4f recall=%.4f precision=%.4f fallback=%s "
                "(n_qualifying=%s)", thr["threshold"], thr["recall"],
                thr["precision"], thr["fallback"],
                thr.get("n_qualifying_thresholds"))
    if thr["fallback"]:
        logger.warning("threshold policy NOT fully satisfiable: %s",
                       thr["fallback_reason"])

    # ---- ONE-TIME test evaluation: both conditions + domain-hosted subgroup
    logger.info("ONE-TIME v2 test evaluation (n=%d): full and none conditions",
                len(y_test))
    results: Dict[str, Any] = {}
    rows = []
    sub = _domain_hosted_mask(test_df["registrable_domain"])
    for cond in ("full", "none"):
        X = fusion_matrix(builder, test_df["url"].tolist(), E_test,
                          _CONDITION_MASKS[cond])
        proba = np.asarray(calibrated.predict_proba(X))[:, 1]
        pred = (proba >= thr["threshold"]).astype(int)
        m = compute_metrics(y_test, pred, proba)
        sub_m = compute_metrics(y_test[sub], pred[sub], proba[sub]) if sub.any() else None
        brier = brier_score_loss(y_test, proba)
        ece = expected_calibration_error(y_test, proba)
        results[cond] = {"metrics": {k: (round(float(v), 4) if isinstance(v, float) else v)
                                     for k, v in m.items()},
                         "subgroup_domain_hosted": (
                             {k: (round(float(v), 4) if isinstance(v, float) else v)
                              for k, v in sub_m.items()} if sub_m else None),
                         "brier": round(float(brier), 4),
                         "ece_10bin": round(float(ece), 4)}
        rows.append(_metrics_row(f"rf_v2_fusion[{cond}]", "test", m, len(y_test)))
        logger.info("  test[%s]: acc=%.4f prec=%.4f rec=%.4f f1=%.4f | "
                    "brier=%.4f ece=%.4f | domain-hosted f1=%s",
                    cond, m["accuracy"], m["precision"], m["recall"], m["f1"],
                    brier, ece,
                    f"{sub_m['f1']:.4f}" if sub_m else "n/a")

    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "format_version": 2,
        "model_name": "random_forest_v2_fusion",
        "calibrated_model": calibrated,
        "rf_model": rf,
        "url_builder": builder,
        "threshold": float(thr["threshold"]),
        "dense_feature_names": list(FEATURE_NAMES),
        "evidence_feature_names": list(EVIDENCE_FEATURE_NAMES),
        "evidence_log_fingerprint": ev_fp,
        "dataset_fingerprint": dataset_fingerprint(train_df),
        "input_contract": (
            "calibrated_model.predict_proba(X) where X = hstack(["
            "url_builder.transform(urls), csr_matrix(E)]) and E is (n, 14) in "
            "EVIDENCE_FEATURE_NAMES order; mask via features.evidence_features."
            "mask_groups. See model/train_v2.fusion_matrix."),
        "random_seed": seed, "created_at": stamp,
    }, bundle_path)
    logger.info("persisted %s (%d bytes)", bundle_path, bundle_path.stat().st_size)

    out_csv = Path(m2["metrics_path"])
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    Path(m2["threshold_path"]).write_text(json.dumps(
        {k: v for k, v in thr.items() if k != "pr_curve_sample"},
        indent=2, default=str), encoding="utf-8")

    metadata = {
        "model_name": "random_forest_v2_fusion", "created_at": stamp,
        "selection_rationale": SELECTION_RATIONALE,
        "evidence_log_fingerprint": ev_fp,
        "dataset_fingerprint": dataset_fingerprint(train_df),
        "split_stats": split_stats,
        "threshold": {k: thr.get(k) for k in ("threshold", "recall", "precision",
                                              "fallback", "fallback_reason",
                                              "n_qualifying_thresholds")},
        "threshold_selected_on": "validation, full-evidence condition",
        "test_results": results,
        "n_test_rows": int(len(y_test)),
        "temporal_caveat": ("evidence acquired after the feed snapshot; "
                            "not_found partly reflects takedown lag; full-"
                            "evidence results are an upper bound"),
        "test_evaluated": True, "test_evaluated_at": stamp,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str),
                             encoding="utf-8")
    logger.info("wrote %s, %s, %s | total %.1fs", metadata_path,
                m2["threshold_path"], out_csv, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
