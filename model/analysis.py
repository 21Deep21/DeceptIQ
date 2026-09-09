"""Post-training analysis tooling (v1.1, improvements #6 and #7).

ABLATION: trains the deployed model configuration on three feature
subsets (combined / lexical-only / ngram-only) using the FROZEN train
split and evaluates on the VALIDATION split, at threshold 0.5 (same
protocol as the Phase 2 comparison). The one-time test split is NOT
touched - its evaluation was consumed in Phase 3R and is never re-used.
Results -> model/ablation_results.csv.

ERROR ANALYSIS: categorises the recorded final-test misclassifications
(model/test_errors.csv) by direction, source, probability and URL shape.
Measured data only - no new evaluation. Results -> model/error_analysis.json.

Usage:
    python -m model.analysis              # both
    python -m model.analysis --ablation
    python -m model.analysis --errors
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from appconfig import get_config, setup_console_logging
from features.feature_builder import URLFeatureBuilder
from model.split import get_or_create_splits
from model.train import build_estimators, compute_metrics

logger = logging.getLogger("app.analysis")

_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


# ------------------------------------------------------------ ablation (#6)

def ablation_rows(X_train: Sequence[str], y_train, X_val: Sequence[str], y_val,
                  seed: int = 42, estimator: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Train one estimator per feature_mode; evaluate on the given
    validation data at threshold 0.5. Pure and seeded - testable on any
    corpus. The real run passes the frozen splits and the deployed RF
    configuration."""
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    rows: List[Dict[str, Any]] = []
    y_tr = np.asarray(y_train, dtype=int)
    y_va = np.asarray(y_val, dtype=int)
    for mode in ("combined", "lexical", "ngram"):
        t0 = time.time()
        pipe = Pipeline([
            ("features", URLFeatureBuilder(feature_mode=mode)),
            ("model", clone(estimator)),
        ])
        pipe.fit(list(X_train), y_tr)
        proba = np.asarray(pipe.predict_proba(list(X_val)))[:, 1]
        m = compute_metrics(y_va, (proba >= 0.5).astype(int), proba)
        rows.append({
            "feature_mode": mode,
            "n_features": len(pipe.named_steps["features"].feature_names_),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "pr_auc": round(m["pr_auc"], 4),
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "fit_seconds": round(time.time() - t0, 1),
        })
    return rows


def run_ablation(cfg: Dict[str, Any]) -> Path:
    train_df, val_df, _test_df, _ = get_or_create_splits(cfg)
    logger.info("ablation on frozen splits: train=%d val=%d | TEST SPLIT NOT EVALUATED",
                len(train_df), len(val_df))
    X_train = np.asarray(train_df["url"].tolist(), dtype=object)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_val = np.asarray(val_df["url"].tolist(), dtype=object)
    y_val = val_df["label"].to_numpy(dtype=int)
    seed = int(cfg["dataset"]["random_seed"])
    models, _ = build_estimators(seed)
    rows = ablation_rows(X_train, y_train, X_val, y_val,
                         seed=seed, estimator=models["random_forest"]["estimator"])

    # integrity: the combined row must reproduce the Phase 2R validation record
    metrics_path = Path(cfg["model"]["metrics_path"])
    if metrics_path.exists():
        dfm = pd.read_csv(metrics_path)
        rec = dfm[(dfm["model"] == "random_forest") & (dfm["stage"] == "validation")]
        if not rec.empty and rows:
            delta = abs(float(rec.iloc[0]["f1"]) - rows[0]["f1"])
            if delta <= 1e-3:
                logger.info("integrity OK: combined-row validation F1 %.4f reproduces "
                            "the recorded Phase 2R value", rows[0]["f1"])
            else:
                logger.warning("combined-row F1 %.4f differs from recorded %.4f - "
                               "the dataset/splits have changed since that record",
                               rows[0]["f1"], float(rec.iloc[0]["f1"]))

    out = metrics_path.parent / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("wrote %s", out)
    for r in rows:
        logger.info("  %-9s features=%5d acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc=%.4f pr=%.4f (%.1fs)",
                    r["feature_mode"], r["n_features"], r["accuracy"], r["precision"],
                    r["recall"], r["f1"], r["roc_auc"], r["pr_auc"], r["fit_seconds"])
    return out


# ------------------------------------------------------ error analysis (#7)

def _bucket(sub: pd.DataFrame, worst: str) -> Dict[str, Any]:
    n = len(sub)
    if n == 0:
        return {"count": 0, "by_source": {}, "probability": None,
                "url_shape": None, "most_confident_5": []}
    hosts = sub["url"].map(lambda u: urlsplit(str(u)).hostname or "")
    paths = sub["url"].map(lambda u: urlsplit(str(u)).path)
    queries = sub["url"].map(lambda u: urlsplit(str(u)).query)
    probs = sub["calibrated_probability"].astype(float)
    order = sub.assign(p=probs).sort_values("p")
    top = order if worst == "lowest" else order.iloc[::-1]
    return {
        "count": int(n),
        "by_source": {str(k): int(v) for k, v in sub["source"].value_counts().items()},
        "probability": {
            "min": round(float(probs.min()), 4),
            "median": round(float(probs.median()), 4),
            "mean": round(float(probs.mean()), 4),
            "max": round(float(probs.max()), 4),
        },
        "url_shape": {
            "https_share": round(float(sub["url"].str.startswith("https://").mean()), 4),
            "ip_host_share": round(float(hosts.map(
                lambda h: bool(_IP_RE.match(str(h))) or str(h).startswith("[")).mean()), 4),
            "path_gt1_share": round(float((paths.str.len() > 1).mean()), 4),
            "has_query_share": round(float((queries.str.len() > 0).mean()), 4),
            "mean_path_length": round(float(paths.str.len().mean()), 1),
        },
        "most_confident_5": [
            {"url": str(r["url"]), "calibrated_probability": round(float(r["p"]), 4)}
            for _, r in top.head(5).iterrows()
        ],
    }


def analyze_errors(df: pd.DataFrame) -> Dict[str, Any]:
    """Categorise recorded test errors: FPs (worst = highest p) and FNs
    (worst = lowest p). Measured data only - never a new evaluation."""
    required = {"url", "label", "source", "calibrated_probability", "predicted_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"error file missing columns: {sorted(missing)}")
    return {
        "total_errors": int(len(df)),
        "note": "measured from the one-time final-test evaluation "
                "(model/test_errors.csv); no new evaluation performed",
        "false_positives": _bucket(df[df["label"] == 0], "highest"),
        "false_negatives": _bucket(df[df["label"] == 1], "lowest"),
    }


def run_error_analysis(cfg: Dict[str, Any]) -> Optional[Path]:
    src = Path(cfg["model"]["test_errors_path"])
    if not src.exists():
        logger.warning("no test_errors.csv at %s - skipping error analysis", src)
        return None
    result = analyze_errors(pd.read_csv(src))
    out = Path(cfg["model"]["metrics_path"]).parent / "error_analysis.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)
    fp, fn = result["false_positives"], result["false_negatives"]
    logger.info("errors: %d total | FP %d (sources: %s) | FN %d (sources: %s)",
                result["total_errors"], fp["count"], fp.get("by_source"),
                fn["count"], fn.get("by_source"))
    if fp.get("url_shape"):
        logger.info("FP url_shape: %s", fp["url_shape"])
    return out


def main(argv=None) -> int:
    setup_console_logging()
    p = argparse.ArgumentParser(
        description="Ablation + error analysis (validation split / recorded artifacts only)")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--errors", action="store_true")
    args = p.parse_args(argv)
    cfg = get_config()
    if args.ablation or not args.errors:
        run_ablation(cfg)
    if args.errors or not args.ablation:
        run_error_analysis(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
