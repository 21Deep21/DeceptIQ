"""Phase 2 - model training, domain-aware evaluation, and comparison.

Leakage discipline enforced in this file:
  * Splits are DOMAIN-aware; every registrable domain lives in exactly
    one of train/val/test (data/processed/splits/, frozen by manifest).
  * CV uses StratifiedGroupKFold on the TRAIN split with
    groups=registrable_domain.
  * The TF-IDF vectorizer (inside URLFeatureBuilder) is fitted only on
    the URLs of each fold's training part - never on the full dataset
    before splitting, never on validation.
  * The final TEST split is NOT evaluated in this phase; it is scored
    exactly once in Phase 3 after model selection + calibration +
    threshold optimization.
  * All comparisons here use the default 0.5 threshold.

Usage:
    python -m model.train                      # all available models + stacking
    python -m model.train --models logreg      # quick single-model smoke test
    python -m model.train --skip-stacking
    python -m model.train --force-split        # rebuild frozen splits
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from appconfig import get_config, setup_console_logging
from features.feature_builder import URLFeatureBuilder
from model.split import get_or_create_splits
from model.stacking import GroupedStackingClassifier

logger = logging.getLogger(__name__)

# Optional model libraries: a missing library is reported honestly, never fatal.
try:
    from xgboost import XGBClassifier
    XGB_IMPORT_ERROR: str = ""
except ImportError as exc:
    XGBClassifier = None
    XGB_IMPORT_ERROR = str(exc)
try:
    from lightgbm import LGBMClassifier
    LGBM_IMPORT_ERROR = ""
except ImportError as exc:
    LGBMClassifier = None
    LGBM_IMPORT_ERROR = str(exc)
try:
    from catboost import CatBoostClassifier
    CATBOOST_IMPORT_ERROR = ""
except ImportError as exc:
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = str(exc)

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
CONFUSION_KEYS = ("tn", "fp", "fn", "tp")
CSV_COLUMNS = (
    ["model", "stage", "n_folds", "threshold", "trained_at"]
    + list(METRIC_KEYS)
    + [f"{k}_std" for k in METRIC_KEYS]
    + list(CONFUSION_KEYS)
    + ["notes"]
)


def build_estimators(seed: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Available models + honest skip report for unavailable libraries."""
    models: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, str] = {}

    models["logreg"] = {
        "estimator": LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1000, random_state=seed
        ),
        "scale_dense": True,
        "family": "linear",
    }
    models["random_forest"] = {
        "estimator": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "scale_dense": False,
        "family": "tree",
    }
    if XGBClassifier is not None:
        models["xgboost"] = {
            "estimator": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                tree_method="hist", eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            ),
            "scale_dense": False,
            "family": "tree",
        }
    else:
        skipped["xgboost"] = f"xgboost not importable: {XGB_IMPORT_ERROR}"
    if LGBMClassifier is not None:
        models["lightgbm"] = {
            "estimator": LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.1,
                random_state=seed, n_jobs=-1, verbose=-1,
            ),
            "scale_dense": False,
            "family": "tree",
        }
    else:
        skipped["lightgbm"] = f"lightgbm not importable: {LGBM_IMPORT_ERROR}"
    if CatBoostClassifier is not None:
        models["catboost"] = {
            "estimator": CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.1,
                random_seed=seed, verbose=0, allow_writing_files=False,
            ),
            "scale_dense": False,
            "family": "tree",
        }
    else:
        skipped["catboost"] = f"catboost not importable: {CATBOOST_IMPORT_ERROR}"
    return models, skipped


def make_pipeline(spec: Dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("features", URLFeatureBuilder(scale_dense=bool(spec["scale_dense"]))),
            ("model", clone(spec["estimator"])),
        ]
    )


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if both_classes else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate(model, X, y, threshold: float = 0.5) -> Dict[str, Any]:
    proba = np.asarray(model.predict_proba(X))[:, 1]
    pred = (proba >= threshold).astype(int)
    return compute_metrics(y, pred, proba)


def grouped_cv(
    make_fresh: Callable[[], Any], X, y, groups, seed: int, n_folds: int
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """StratifiedGroupKFold CV over the TRAIN split; fresh pipeline per fold.

    Threshold is fixed at 0.5 here (Phase 3 optimizes it on validation).
    """
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: List[Dict[str, Any]] = []
    for fold, (tr, te) in enumerate(skf.split(X, y, groups), start=1):
        t0 = time.time()
        model = make_fresh()
        model.fit(X[tr], y[tr])
        m = evaluate(model, X[te], y[te])
        m["fold"] = fold
        m["fit_seconds"] = round(time.time() - t0, 1)
        folds.append(m)
        logger.info(
            "fold %d/%d: acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc=%.4f pr=%.4f (%.1fs)",
            fold, n_folds, m["accuracy"], m["precision"], m["recall"],
            m["f1"], m["roc_auc"], m["pr_auc"], m["fit_seconds"],
        )
    summary: Dict[str, float] = {"n_folds": float(n_folds)}
    for k in METRIC_KEYS:
        vals = [f[k] for f in folds if not np.isnan(f[k])]
        summary[k] = float(np.mean(vals)) if vals else float("nan")
        summary[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    return summary, folds


def _fill_metrics(row: Dict[str, Any], source: Dict[str, Any], with_std: bool) -> None:
    for k in METRIC_KEYS:
        row[k] = round(float(source[k]), 4)
        row[f"{k}_std"] = round(float(source[f"{k}_std"]), 4) if with_std else np.nan


def _base_row(model: str, stage: str, stamp: str, notes: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model": model, "stage": stage, "n_folds": np.nan, "threshold": 0.5,
        "trained_at": stamp, "notes": notes,
    }
    for k in METRIC_KEYS:
        row[k] = np.nan
        row[f"{k}_std"] = np.nan
    for k in CONFUSION_KEYS:
        row[k] = np.nan
    return row


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: model training and comparison")
    p.add_argument("--models", default=None,
                   help="comma-separated subset, e.g. logreg,random_forest (default: all)")
    p.add_argument("--skip-stacking", action="store_true")
    p.add_argument("--force-split", action="store_true", help="rebuild frozen splits")
    return p.parse_args(argv)


def main(argv=None) -> int:
    setup_console_logging()
    t_start = time.time()
    args = _parse_args(argv)
    cfg = get_config()
    seed = int(cfg["dataset"]["random_seed"])
    n_folds = int(cfg["model"].get("cv_folds", 5))

    train_df, val_df, test_df, split_stats = get_or_create_splits(cfg, force=args.force_split)
    logger.info(
        "splits: train=%d val=%d test=%d rows | domains %d/%d/%d | dual-label domains=%d",
        split_stats["rows"]["train"], split_stats["rows"]["val"], split_stats["rows"]["test"],
        split_stats["domains"]["train"], split_stats["domains"]["val"], split_stats["domains"]["test"],
        split_stats["dual_label_domains"],
    )
    logger.info("labels per split: train=%s val=%s test=%s",
                split_stats["by_label"]["train"], split_stats["by_label"]["val"],
                split_stats["by_label"]["test"])
    logger.info("FINAL TEST SET IS FROZEN: %d rows - NOT evaluated in this phase", len(test_df))

    X_train = np.asarray(train_df["url"].tolist(), dtype=object)
    y_train = train_df["label"].to_numpy(dtype=int)
    groups_train = train_df["registrable_domain"].to_numpy()
    X_val = np.asarray(val_df["url"].tolist(), dtype=object)
    y_val = val_df["label"].to_numpy(dtype=int)

    models, skipped = build_estimators(seed)
    for name, reason in skipped.items():
        logger.warning("SKIPPED %s: %s", name, reason)
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in models]
        if unknown:
            raise SystemExit(f"unknown models: {unknown}; available: {list(models)}")
        models = {k: v for k, v in models.items() if k in wanted}
    if not models:
        logger.error("no models available to train")
        return 1

    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows: List[Dict[str, Any]] = []

    for name, spec in models.items():
        logger.info("=" * 64)
        logger.info("model: %s - grouped %d-fold CV on train split", name, n_folds)
        cv_summary, _folds = grouped_cv(
            lambda spec=spec: make_pipeline(spec), X_train, y_train, groups_train, seed, n_folds
        )
        logger.info(
            "%s CV mean: acc=%.4f+-%.4f prec=%.4f+-%.4f rec=%.4f+-%.4f f1=%.4f+-%.4f roc=%.4f+-%.4f pr=%.4f+-%.4f",
            name,
            cv_summary["accuracy"], cv_summary["accuracy_std"],
            cv_summary["precision"], cv_summary["precision_std"],
            cv_summary["recall"], cv_summary["recall_std"],
            cv_summary["f1"], cv_summary["f1_std"],
            cv_summary["roc_auc"], cv_summary["roc_auc_std"],
            cv_summary["pr_auc"], cv_summary["pr_auc_std"],
        )
        t0 = time.time()
        final_model = make_pipeline(spec)
        final_model.fit(X_train, y_train)
        val_metrics = evaluate(final_model, X_val, y_val)
        logger.info(
            "%s VALIDATION: acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc=%.4f pr=%.4f | tn=%d fp=%d fn=%d tp=%d (%.1fs)",
            name, val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"],
            val_metrics["f1"], val_metrics["roc_auc"], val_metrics["pr_auc"],
            val_metrics["tn"], val_metrics["fp"], val_metrics["fn"], val_metrics["tp"],
            time.time() - t0,
        )

        cv_row = _base_row(name, "cv", stamp, "mean+std over grouped folds (train split only)")
        cv_row["n_folds"] = n_folds
        _fill_metrics(cv_row, cv_summary, with_std=True)
        rows.append(cv_row)

        val_row = _base_row(name, "validation", stamp,
                            "fitted on train, evaluated once on validation (threshold 0.5)")
        _fill_metrics(val_row, val_metrics, with_std=False)
        for k in CONFUSION_KEYS:
            val_row[k] = val_metrics[k]
        rows.append(val_row)

    stacking_done = False
    if not args.skip_stacking:
        tree_names = [n for n, s in models.items() if s.get("family") == "tree"]
        if len(tree_names) < 2:
            logger.warning("stacking skipped: fewer than 2 tree base models available")
        else:
            logger.info("=" * 64)
            logger.info("stacking ensemble: bases=%s meta=logistic regression (grouped OOF)",
                        tree_names)
            stack = GroupedStackingClassifier(
                base_estimators=[(n, make_pipeline(models[n])) for n in tree_names],
                meta_estimator=LogisticRegression(
                    C=1.0, solver="liblinear", max_iter=1000, random_state=seed
                ),
                n_folds=n_folds,
                seed=seed,
            )
            t0 = time.time()
            stack.fit(X_train, y_train, groups_train)
            stack_val = evaluate(stack, X_val, y_val)
            logger.info(
                "stacking VALIDATION: acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc=%.4f pr=%.4f | tn=%d fp=%d fn=%d tp=%d (%.1fs)",
                stack_val["accuracy"], stack_val["precision"], stack_val["recall"],
                stack_val["f1"], stack_val["roc_auc"], stack_val["pr_auc"],
                stack_val["tn"], stack_val["fp"], stack_val["fn"], stack_val["tp"],
                time.time() - t0,
            )
            srow = _base_row(
                "stacking", "validation", stamp,
                "grouped stacking: meta=logreg on out-of-fold base probabilities; "
                "no outer CV (computational cost, documented); threshold 0.5",
            )
            _fill_metrics(srow, stack_val, with_std=False)
            for k in CONFUSION_KEYS:
                srow[k] = stack_val[k]
            rows.append(srow)
            stacking_done = True

    metrics_path = Path(cfg["model"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=CSV_COLUMNS)
    out.to_csv(metrics_path, index=False)
    logger.info("wrote %s", metrics_path)

    notes = {
        "created_at": stamp,
        "phase": "2 - model comparison",
        "random_seed": seed,
        "n_folds": n_folds,
        "threshold_note": "All Phase 2 comparisons use threshold 0.5; threshold optimization is Phase 3 (validation set).",
        "test_set_note": "test.csv frozen and untouched this phase; evaluated once in Phase 3.",
        "split_stats": split_stats,
        "models_trained": list(models.keys()) + (["stacking"] if stacking_done else []),
        "models_skipped": skipped,
        "hyperparameters": {
            name: {k: str(v) for k, v in spec["estimator"].get_params().items()}
            for name, spec in models.items()
        },
        "feature_set": {
            "dense_lexical": 26,
            "tfidf": "char n-gram (3,5) TF-IDF fitted per fold / per training split inside URLFeatureBuilder",
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    notes_path = Path(cfg["model"].get("notes_path", "model/phase2_notes.json"))
    notes_path.write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", notes_path)

    logger.info("=" * 64)
    logger.info("PHASE 2 COMPLETE - validation ranking by PR-AUC (threshold 0.5):")
    val_rows = out[out["stage"] == "validation"].sort_values("pr_auc", ascending=False)
    for _, r in val_rows.iterrows():
        logger.info(
            "  %-14s pr_auc=%.4f roc_auc=%.4f f1=%.4f recall=%.4f precision=%.4f",
            r["model"], r["pr_auc"], r["roc_auc"], r["f1"], r["recall"], r["precision"],
        )
    print()
    print(out.to_string(index=False, na_rep=""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
