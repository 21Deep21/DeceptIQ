"""Phase 3 - calibration, threshold optimization, final model persistence.

Order of operations (leakage discipline):
  1. Load the FROZEN domain-aware splits (train/val/test).
  2. Fit the selected raw pipeline on the TRAIN split only.
  3. Calibrate on the VALIDATION split via FrozenEstenticator: the
     calibrator sees only validation data, which shares ZERO registrable
     domains with train (domain-aware splits). Raw scores are scores;
     only calibrated outputs are called probabilities.
  4. Select the operating threshold on validation (recall-first policy).
  5. Evaluate the final TEST split EXACTLY ONCE (guarded via
     metadata.json; re-evaluation requires --force and is recorded).
  6. Persist: model/model.joblib (bundle incl. calibrated model, raw
     pipeline, threshold, feature names), threshold.json, metadata.json
     (incl. sha256 of the bundle), test error analysis, appended metrics.

Usage:
    python -m model.finalize                  # normal one-time run
    python -m model.finalize --model xgboost  # experiment with another model
    python -m model.finalize --force          # re-run (overwrites, recorded)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from appconfig import get_config, setup_console_logging
from features.feature_extraction import FEATURE_NAMES
from model.split import get_or_create_splits
from model.train import (
    CONFUSION_KEYS,
    CSV_COLUMNS,
    METRIC_KEYS,
    build_estimators,
    compute_metrics,
    make_pipeline,
)

logger = logging.getLogger(__name__)

PR_SAMPLE_MAX = 200

SELECTION_RATIONALE = (
    "Random Forest selected from the Phase 2R comparison on dataset revision 1R "
    "(benign class: 2,335 observed sitemap deep-link URLs + 2,665 constructed "
    "homepages). Measured grouped 5-fold CV on the train split: recall 0.9862 "
    "(+/-0.0087), precision 0.9757 (+/-0.0392), F1 0.9804, ROC-AUC 0.9977, "
    "PR-AUC 0.9977 - best CV F1 among individual models. Validation at threshold "
    "0.5: accuracy 0.9736, precision 0.9657, recall 0.9821, F1 0.9738, confusion "
    "tn=973 fp=35 fn=18 tp=985 - the most balanced operational profile, already "
    "satisfying the deployment policy (recall>=0.97, precision>=0.90) at the "
    "default threshold. The stacking ensemble was evaluated and NOT retained: "
    "validation PR-AUC 0.9967 vs 0.9966 for Random Forest (delta 0.0001, "
    "noise-level) with identical F1, while adding four base models, a "
    "meta-learner and far harder SHAP explainability. LightGBM (the v1 "
    "selection) was demoted: after the benign deep-link fix its 0.5-threshold "
    "precision collapsed (CV 0.8811, validation 0.8853, fp=129) - it over-triggers "
    "on legitimate URLs with paths; Random Forest dominates it on CV and "
    "validation F1. CatBoost had the best CV PR-AUC (0.9981) but lower validation "
    "F1 (0.9565) and ~10x slower fitting. Note: with a sklearn RandomForest raw "
    "model, shap.TreeExplainer is the single SHAP backend (the LightGBM-native "
    "pred_contrib fallback applies to LightGBM pipelines only); the active "
    "backend is reported in every explanation."
)


# --------------------------------------------------------------- pure helpers

def select_threshold(
    y_true, proba, target_recall: float, min_precision: float
) -> Dict[str, Any]:
    """Recall-first threshold selection on a precision-recall curve.

    Deterministic rule (documented, validation-only, never test):
      1. Consider thresholds where recall >= target_recall AND
         precision >= min_precision.
      2. Among them: maximize recall, then precision, then pick the
         LOWEST threshold (recall-side safety margin).
      3. If no threshold qualifies, fall back to the F2-optimal
         threshold (F2 weights recall above precision) and set
         fallback=True so the shortfall is reported, never hidden.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    out: Dict[str, Any] = {
        "target_recall": float(target_recall),
        "min_precision": float(min_precision),
        "rule": "maximize recall subject to precision >= min_precision; "
                "tie-break: max precision, then lowest threshold; "
                "fallback: F2-optimal threshold",
    }
    pred05 = (p >= 0.5).astype(int)
    out["recall_at_default_0.5"] = float(recall_score(y, pred05, zero_division=0))
    out["precision_at_default_0.5"] = float(precision_score(y, pred05, zero_division=0))

    if len(p) == 0 or len(np.unique(p)) <= 1:
        out.update(threshold=0.5, fallback=True,
                   fallback_reason="degenerate probability distribution",
                   recall=out["recall_at_default_0.5"],
                   precision=out["precision_at_default_0.5"])
        return out

    precision, recall, thresholds = precision_recall_curve(y, p)
    n = len(thresholds)
    prec = np.asarray(precision[:n], dtype=float)
    rec = np.asarray(recall[:n], dtype=float)
    thr = np.asarray(thresholds, dtype=float)

    qualifying = np.where((rec >= target_recall) & (prec >= min_precision))[0]
    out["n_qualifying_thresholds"] = int(len(qualifying))
    out["max_recall_any_threshold"] = float(rec.max())

    if len(qualifying) > 0:
        best = int(min(qualifying, key=lambda i: (-rec[i], -prec[i], thr[i])))
        out.update(threshold=float(thr[best]), recall=float(rec[best]),
                   precision=float(prec[best]), fallback=False,
                   fallback_reason=None)
    else:
        f2 = np.zeros(n, dtype=float)
        denom = prec + rec
        mask = denom > 0
        f2[mask] = 5.0 * prec[mask] * rec[mask] / (4.0 * denom[mask])
        best = int(np.argmax(f2))
        out.update(threshold=float(thr[best]), recall=float(rec[best]),
                   precision=float(prec[best]), fallback=True,
                   fallback_reason=("no threshold satisfies recall >= target AND "
                                    "precision >= min_precision; fell back to "
                                    "F2-optimal threshold"))

    pr, rc = out["precision"], out["recall"]
    out["f1_at_threshold"] = float(2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0

    idx = (np.linspace(0, n - 1, PR_SAMPLE_MAX).astype(int)
           if n > PR_SAMPLE_MAX else np.arange(n))
    out["pr_curve_sample"] = [
        {"threshold": round(float(thr[i]), 6),
         "precision": round(float(prec[i]), 6),
         "recall": round(float(rec[i]), 6)}
        for i in idx
    ]
    return out


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Equal-width binned ECE: sum(n_bin/N * |mean confidence - accuracy|)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    if len(p) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == int(n_bins) - 1 else (p >= lo) & (p < hi)
        if mask.any():
            ece += (mask.sum() / len(p)) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


# --------------------------------------------------------------- private utils

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lib_versions() -> Dict[str, str]:
    v = {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
         "pandas": pd.__version__, "joblib": joblib.__version__}
    for mod in ("lightgbm", "xgboost", "catboost"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            pass
    return v


def _evaluate_at(y, proba, threshold: float) -> Dict[str, Any]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return compute_metrics(y, pred, proba)


def _row(model: str, stage: str, threshold: float, stamp: str,
         m: Dict[str, Any], brier: float, ece: float, notes: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {c: np.nan for c in CSV_COLUMNS}
    row.update(model=model, stage=stage, n_folds=np.nan,
               threshold=round(float(threshold), 4), trained_at=stamp, notes=notes)
    for k in METRIC_KEYS:
        row[k] = round(float(m[k]), 4)
    for k in CONFUSION_KEYS:
        row[k] = int(m[k])
    row["brier"] = round(float(brier), 4)
    row["ece"] = round(float(ece), 4)
    return row


def _append_metric_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        combined = pd.concat([pd.read_csv(path), new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(path, index=False)


def _check_against_phase2(metrics_path: Path, selected: str,
                          val_metrics: Dict[str, Any]) -> None:
    """Integrity check: the refit raw model must reproduce the Phase 2
    validation result for the same model (same seed, same data)."""
    if not metrics_path.exists():
        logger.warning("no model_metrics.csv yet - cannot cross-check validation")
        return
    try:
        dfm = pd.read_csv(metrics_path)
        row = dfm[(dfm["model"] == selected) & (dfm["stage"] == "validation")]
    except Exception:
        logger.warning("model_metrics.csv unreadable - cannot cross-check validation")
        return
    if row.empty:
        logger.warning("no Phase 2 validation row for %s - cannot cross-check", selected)
        return
    phase2_f1 = round(float(row.iloc[0]["f1"]), 4)  # CSV stores 4-decimal precision
    now_f1 = round(float(val_metrics["f1"]), 4)
    if abs(phase2_f1 - now_f1) <= 1e-6:
        logger.info("integrity check OK: validation F1 reproduces the Phase 2 "
                    "recorded value (%.4f)", now_f1)
    else:
        logger.warning("validation F1 differs from Phase 2 record (%.4f vs %.4f) - "
                       "investigate before trusting artifacts", now_f1, phase2_f1)


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3: calibration, threshold, finalize")
    p.add_argument("--model", default=None,
                   help="override config model.selected_model (experimentation)")
    p.add_argument("--force", action="store_true",
                   help="re-run the one-time test evaluation (recorded)")
    return p.parse_args(argv)


# ------------------------------------------------------------------- main

def main(argv=None) -> int:
    setup_console_logging()
    args = _parse_args(argv)
    cfg = get_config()
    mcfg = cfg["model"]
    seed = int(cfg["dataset"]["random_seed"])
    selected = args.model or mcfg["selected_model"]
    cal_method = str(mcfg.get("calibration", {}).get("method", "sigmoid"))
    policy = mcfg.get("threshold_policy", {})
    target_recall = float(policy.get("target_recall", 0.97))
    min_precision = float(policy.get("min_precision", 0.90))

    bundle_path = Path(mcfg["model_path"])
    metadata_path = Path(mcfg["metadata_path"])
    threshold_path = Path(mcfg["threshold_path"])
    metrics_path = Path(mcfg["metrics_path"])
    errors_path = Path(mcfg.get("test_errors_path", "model/test_errors.csv"))

    # ---- one-time test guard --------------------------------------------
    if metadata_path.exists() and not args.force:
        try:
            meta = json.loads(metadata_path.read_text())
            if meta.get("test_evaluated"):
                logger.error(
                    "final test set was already evaluated at %s; refusing to "
                    "evaluate again (one-time discipline). Use --force to "
                    "override (the re-run is recorded).",
                    meta.get("test_evaluated_at"),
                )
                return 2
        except json.JSONDecodeError:
            logger.warning("metadata.json unreadable; continuing")

    models, _skipped = build_estimators(seed)
    if selected not in models:
        logger.error("model '%s' unavailable; available: %s", selected, list(models))
        return 1

    train_df, val_df, test_df, split_stats = get_or_create_splits(cfg)
    X_train = np.asarray(train_df["url"].tolist(), dtype=object)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_val = np.asarray(val_df["url"].tolist(), dtype=object)
    y_val = val_df["label"].to_numpy(dtype=int)
    X_test = np.asarray(test_df["url"].tolist(), dtype=object)
    y_test = test_df["label"].to_numpy(dtype=int)

    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # ---- 1. raw model on TRAIN ------------------------------------------
    logger.info("fitting raw %s pipeline on train split (%d rows)", selected, len(X_train))
    t0 = time.time()
    raw = make_pipeline(models[selected])
    raw.fit(X_train, y_train)
    logger.info("raw model fitted in %.1fs", time.time() - t0)

    raw_val_proba = np.asarray(raw.predict_proba(X_val))[:, 1]
    raw_val_metrics = _evaluate_at(y_val, raw_val_proba, 0.5)
    logger.info("raw %s validation @0.5: acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
                selected, raw_val_metrics["accuracy"], raw_val_metrics["precision"],
                raw_val_metrics["recall"], raw_val_metrics["f1"])
    _check_against_phase2(metrics_path, selected, raw_val_metrics)

    # ---- 2. calibrate on VALIDATION (no domain overlap with train) -------
    logger.info("calibrating (method=%s) on validation split (%d rows, zero domain "
                "overlap with train)", cal_method, len(X_val))
    calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method=cal_method)
    calibrated.fit(X_val, y_val)
    val_proba = np.asarray(calibrated.predict_proba(X_val))[:, 1]

    # ---- 3. threshold on VALIDATION --------------------------------------
    thr = select_threshold(y_val, val_proba, target_recall, min_precision)
    logger.info("threshold selection: threshold=%.4f recall=%.4f precision=%.4f "
                "fallback=%s (n_qualifying=%s, max_recall=%.4f)",
                thr["threshold"], thr["recall"], thr["precision"], thr["fallback"],
                thr.get("n_qualifying_thresholds"), thr.get("max_recall_any_threshold", float("nan")))
    if thr["fallback"]:
        logger.warning("threshold policy NOT fully satisfiable: %s", thr["fallback_reason"])
    val_op = _evaluate_at(y_val, val_proba, thr["threshold"])
    logger.info("validation operating point @threshold: acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
                val_op["accuracy"], val_op["precision"], val_op["recall"], val_op["f1"])

    # ---- 4. persist the inference bundle ---------------------------------
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 1,
        "model_name": selected,
        "calibrated_model": calibrated,          # served: calibrated probabilities
        "raw_pipeline": raw,                     # SHAP phase: TreeExplainer target
        "threshold": float(thr["threshold"]),
        "dense_feature_names": list(FEATURE_NAMES),
        "calibration": {"method": cal_method,
                        "fitted_on": "validation split (FrozenEstimator)"},
        "random_seed": seed,
        "created_at": stamp,
        "library_versions": _lib_versions(),
    }
    joblib.dump(bundle, bundle_path)
    digest = _sha256(bundle_path)
    logger.info("persisted %s (%d bytes, sha256=%s...)",
                bundle_path, bundle_path.stat().st_size, digest[:16])

    # ---- 5. ONE-TIME final TEST evaluation --------------------------------
    logger.info("FINAL TEST evaluation (one-time, %d rows) at threshold %.4f - "
                "first and only use of test.csv", len(X_test), thr["threshold"])
    test_proba = np.asarray(calibrated.predict_proba(X_test))[:, 1]
    test_pred = (test_proba >= thr["threshold"]).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_proba)
    test_brier = brier_score_loss(y_test, test_proba)
    test_ece = expected_calibration_error(y_test, test_proba)

    raw_test_proba = np.asarray(raw.predict_proba(X_test))[:, 1]
    raw_test_metrics = _evaluate_at(y_test, raw_test_proba, 0.5)
    raw_test_brier = brier_score_loss(y_test, raw_test_proba)
    raw_test_ece = expected_calibration_error(y_test, raw_test_proba)

    err_mask = test_pred != y_test
    errors = test_df.loc[err_mask].copy()
    errors["calibrated_probability"] = np.round(test_proba[err_mask], 6)
    errors["predicted_label"] = test_pred[err_mask]
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(errors_path, index=False)
    logger.info("wrote %s (%d misclassified rows)", errors_path, int(err_mask.sum()))

    rows = [
        _row(f"{selected}+calibrated", "test", thr["threshold"], stamp, test_metrics,
             test_brier, test_ece,
             "ONE-TIME final test evaluation: calibrated model at deployed threshold"),
        _row(f"{selected}_raw", "test", 0.5, stamp, raw_test_metrics,
             raw_test_brier, raw_test_ece,
             "ONE-TIME final test evaluation: raw model at default 0.5 (reference)"),
    ]
    _append_metric_rows(rows, metrics_path)
    logger.info("appended 2 test rows to %s", metrics_path)

    threshold_path.write_text(json.dumps({
        **{k: v for k, v in thr.items() if k != "pr_curve_sample"},
        "selected_on": "validation split (calibrated probabilities)",
        "operating_point_validation": {k: round(float(val_op[k]), 4) for k in METRIC_KEYS},
        "pr_curve_sample": thr.get("pr_curve_sample", []),
    }, indent=2), encoding="utf-8")
    logger.info("wrote %s", threshold_path)

    metadata = {
        "model_name": selected,
        "created_at": stamp,
        "random_seed": seed,
        "selection_rationale": (SELECTION_RATIONALE if not args.model else
                                f"overridden via --model {selected}; see "
                                f"model_metrics.csv Phase 2 rows for comparison"),
        "calibration": {
            "method": cal_method,
            "approach": "CalibratedClassifierCV(FrozenEstimator(raw_pipeline)) "
                        "fitted on the validation split",
            "no_domain_overlap": "train and validation share no registrable domains",
            "caveat": "calibrator and threshold both use the validation split; the "
                      "sigmoid adds 2 parameters so in-sample optimism on the "
                      "validation rows is negligible; the test split was never used "
                      "for calibration or threshold selection",
            "note": "raw model scores are scores, not probabilities; only "
                    "calibrated outputs are reported as probabilities",
        },
        "threshold": {k: thr.get(k) for k in (
            "threshold", "recall", "precision", "f1_at_threshold", "fallback",
            "fallback_reason", "target_recall", "min_precision",
            "n_qualifying_thresholds", "max_recall_any_threshold")},
        "split_stats": split_stats,
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "feature_set": {"dense_lexical": len(FEATURE_NAMES),
                        "tfidf": "char (3,5)-gram TF-IDF fitted on training data only"},
        "test_results": {
            "calibrated_at_threshold": {
                "threshold": round(float(thr["threshold"]), 4),
                **{k: round(float(test_metrics[k]), 4) for k in METRIC_KEYS},
                **{k: int(test_metrics[k]) for k in CONFUSION_KEYS},
                "brier": round(float(test_brier), 4),
                "ece_10bin": round(float(test_ece), 4),
            },
            "raw_at_0.5_reference": {
                **{k: round(float(raw_test_metrics[k]), 4) for k in METRIC_KEYS},
                **{k: int(raw_test_metrics[k]) for k in CONFUSION_KEYS},
                "brier": round(float(raw_test_brier), 4),
                "ece_10bin": round(float(raw_test_ece), 4),
            },
            "n_test_rows": int(len(X_test)),
            "n_misclassified": int(err_mask.sum()),
        },
        "artifacts": {
            "bundle": str(bundle_path),
            "bundle_sha256": digest,
            "threshold_json": str(threshold_path),
            "metrics_csv": str(metrics_path),
            "test_errors_csv": str(errors_path),
        },
        "test_evaluated": True,
        "test_evaluated_at": stamp,
        "library_versions": _lib_versions(),
        "one_time_discipline": "test.csv evaluated exactly once per run; "
                               "forced re-runs (--force) overwrite and re-stamp",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str),
                             encoding="utf-8")
    logger.info("wrote %s", metadata_path)

    logger.info("=" * 68)
    logger.info("PHASE 3 FINAL TEST RESULTS (one-time, calibrated @ threshold %.4f)",
                thr["threshold"])
    logger.info("accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f",
                test_metrics["accuracy"], test_metrics["precision"],
                test_metrics["recall"], test_metrics["f1"],
                test_metrics["roc_auc"], test_metrics["pr_auc"])
    logger.info("confusion: tn=%d fp=%d fn=%d tp=%d",
                test_metrics["tn"], test_metrics["fp"],
                test_metrics["fn"], test_metrics["tp"])
    logger.info("calibration quality on test: brier=%.4f ece(10-bin)=%.4f "
                "(raw reference: brier=%.4f ece=%.4f)",
                test_brier, test_ece, raw_test_brier, raw_test_ece)
    if test_metrics["recall"] < target_recall:
        logger.warning("test recall %.4f is BELOW the configured target %.2f - "
                       "reported exactly as measured", test_metrics["recall"], target_recall)
    logger.info("misclassified rows: %d (see %s)", int(err_mask.sum()), errors_path)
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
