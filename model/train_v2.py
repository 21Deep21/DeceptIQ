"""Phase B - fusion model training (URL + evidence features).

Trains ONE model (RandomForest, the documented v1 selection) on the fused
feature space: URL features (26 dense + char TF-IDF) + 14 evidence
features (DNS/RDAP derived + missingness indicators).

GROUP-MASKING AUGMENTATION (draft section 5.5): every training URL is
presented under all four evidence-availability conditions (full, DNS-only,
RDAP-only, none), producing a single predictor robust to any runtime
acquisition outcome. Labels are preserved per copy; augmentation happens
AFTER domain-aware split assignment, so all copies of a URL stay in its
split (no new leakage surface).

Reports the validation CONDITIONS TABLE (draft sections 5.7/5.8): URL-only
(masked), +DNS, +RDAP, full evidence, plus the deployed v1 bundle as the
production URL-only baseline (B1) on identical validation rows. Also
reports the DOMAIN-HOSTED subgroup, where evidence actually applies
(IP-literal hosts carry no domain evidence by design).

TEMPORAL CAVEAT (recorded in outputs): evidence was acquired after the
feed snapshot; malicious domains die quickly, so rdap_not_found partly
reflects takedown lag and may overstate its runtime predictive value.
Phase D stability analysis quantifies this; prospective collection is the
draft's fix.

Usage:  python -m model.train_v2
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier

from appconfig import get_config, setup_console_logging
from features.evidence_features import (
    EVIDENCE_FEATURE_NAMES,
    N_EVIDENCE_FEATURES,
    evidence_log_fingerprint,
    evidence_matrix_for_domains,
    load_evidence_matrix,
    mask_groups,
)
from features.feature_builder import URLFeatureBuilder
from features.url_utils import is_ip_hostname
from model.split import dataset_fingerprint, get_or_create_splits
from model.train import compute_metrics

logger = logging.getLogger(__name__)

CONDITIONS: Tuple[str, ...] = ("full", "dns_only", "rdap_only", "none")
_CONDITION_MASKS = {"full": (), "dns_only": ("rdap",),
                    "rdap_only": ("dns",), "none": ("dns", "rdap")}


def fusion_matrix(url_builder: URLFeatureBuilder, urls, E: np.ndarray,
                  mask: Tuple[str, ...] = ()) -> csr_matrix:
    """[URL block | evidence block]; mask zero-fills named groups."""
    Xurl = url_builder.transform(list(urls))
    Ec = mask_groups(E, mask) if mask else E
    return hstack([Xurl, csr_matrix(np.asarray(Ec, dtype=np.float64))],
                  format="csr")


def augment_training(Xurl: csr_matrix, E: np.ndarray, y: np.ndarray,
                     seed: int = 42) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """4-condition group-masking augmentation.

    Returns (X, y, condition_ids, original_indices). All copies of a row
    carry the same label; the shuffle is seeded (deterministic).
    """
    blocks, conds, origs, ys = [], [], [], []
    n = len(y)
    for cond in CONDITIONS:
        Ec = mask_groups(E, _CONDITION_MASKS[cond]) if _CONDITION_MASKS[cond] else np.array(E, copy=True)
        blocks.append(hstack([Xurl, csr_matrix(Ec)], format="csr"))
        conds.extend([cond] * n)
        origs.extend(range(n))
        ys.append(y)
    X = sparse.vstack(blocks, format="csr")
    yy = np.concatenate(ys)
    cc = np.asarray(conds)
    oo = np.asarray(origs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yy))
    return X[idx], yy[idx], cc[idx], oo[idx]


def _domain_hosted_mask(domains) -> np.ndarray:
    return np.asarray([not is_ip_hostname(str(d)) for d in domains], dtype=bool)


def _row(name: str, m: Dict[str, Any], sub_m: Optional[Dict[str, Any]] = None) -> str:
    line = (f"  {name:<26} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} f1={m['f1']:.4f} roc={m['roc_auc']:.4f} "
            f"pr={m['pr_auc']:.4f}")
    if sub_m is not None:
        line += (f"   | domain-hosted: prec={sub_m['precision']:.4f} "
                 f"rec={sub_m['recall']:.4f} f1={sub_m['f1']:.4f}")
    return line


def main() -> int:
    setup_console_logging()
    t0 = time.time()
    cfg = get_config()
    seed = int(cfg["dataset"]["random_seed"])

    train_df, val_df, _test_df, split_stats = get_or_create_splits(cfg)
    logger.info("splits: train=%d val=%d test=%d (domain overlap %s) | "
                "dataset fingerprint %s...",
                split_stats["rows"]["train"], split_stats["rows"]["val"],
                split_stats["rows"]["test"], split_stats["domain_overlap"],
                dataset_fingerprint(train_df)[:12])
    logger.info("dataset sources: %s",
                train_df["source"].value_counts().to_dict())

    matrix = load_evidence_matrix(cfg["evidence"]["dir"])
    fp = evidence_log_fingerprint(cfg["evidence"]["dir"])
    logger.info("evidence log fingerprint: %s...", fp[:16])

    all_domains = set(train_df["registrable_domain"].astype(str)) | \
                  set(val_df["registrable_domain"].astype(str)) | \
                  set(_test_df["registrable_domain"].astype(str))
    uncovered = [d for d in all_domains if d not in matrix and not is_ip_hostname(d)]
    logger.info("evidence coverage: %d/%d non-IP domains (%d uncovered -> "
                "'not acquired' zeros)", len(all_domains) - len(uncovered),
                len(all_domains), len(uncovered))

    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    E_train = evidence_matrix_for_domains(train_df["registrable_domain"], matrix)
    E_val = evidence_matrix_for_domains(val_df["registrable_domain"], matrix)

    builder = URLFeatureBuilder().fit(train_df["url"].tolist())
    Xurl_train = builder.transform(train_df["url"].tolist())
    Xurl_val = builder.transform(val_df["url"].tolist())

    logger.info("augmenting training data: %d rows -> %d (4 conditions)",
                len(y_train), 4 * len(y_train))
    Xaug, yaug, cond_ids, _orig = augment_training(Xurl_train, E_train, y_train, seed=seed)
    logger.info("training RandomForest on fused matrix %s ...", str(Xaug.shape))
    t1 = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(Xaug, yaug)
    logger.info("fusion RF fitted in %.1fs", time.time() - t1)

    sub = _domain_hosted_mask(val_df["registrable_domain"])
    logger.info("VALIDATION CONDITIONS TABLE (n=%d, domain-hosted n=%d):",
                len(y_val), int(sub.sum()))
    rows: List[Dict[str, Any]] = []
    for cond in CONDITIONS:
        X = fusion_matrix(builder, val_df["url"].tolist(), E_val, _CONDITION_MASKS[cond])
        proba = rf.predict_proba(X)[:, 1]
        m = compute_metrics(y_val, (proba >= 0.5).astype(int), proba)
        sub_m = compute_metrics(y_val[sub], ((proba >= 0.5).astype(int))[sub], proba[sub]) \
            if sub.any() else None
        logger.info(_row(cond, m, sub_m))
        rows.append({"system": f"rf_v2_fusion[{cond}]", "stage": "validation",
                     "n": len(y_val), **{k: m[k] for k in
                     ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                      "tn", "fp", "fn", "tp")}})

    # B1: deployed v1 production baseline on identical validation rows
    v1_path = Path(cfg["model"]["model_path"])
    if v1_path.exists():
        v1 = joblib.load(v1_path)
        proba = np.asarray(v1["calibrated_model"].predict_proba(val_df["url"].tolist()))[:, 1]
        m = compute_metrics(y_val, (proba >= 0.5).astype(int), proba)
        sub_m = compute_metrics(y_val[sub], ((proba >= 0.5).astype(int))[sub], proba[sub]) \
            if sub.any() else None
        logger.info(_row("B1 deployed v1 (URL-only)", m, sub_m))
        rows.append({"system": "deployed_v1[B1]", "stage": "validation",
                     "n": len(y_val), **{k: m[k] for k in
                     ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                      "tn", "fp", "fn", "tp")}})
    else:
        logger.warning("v1 bundle not found at %s - B1 baseline skipped", v1_path)

    logger.info("TEMPORAL CAVEAT: rdap_not_found partly reflects takedown lag "
                "(evidence collected after the feed snapshot); treat the "
                "full-evidence numbers as an upper bound for runtime value.")

    out_dir = Path(cfg["model_v2"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "model_metrics.csv", index=False)
    logger.info("wrote %s | total %.1fs", out_dir / "model_metrics.csv", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
