"""Phase E - per-condition probability calibration for the fusion model.

Fixes the documented Phase C approximation (D-C5a): the bundle's sigmoid
calibrator was fitted on the FULL-evidence validation condition and applied
to masked states. This fits one calibrator per evidence condition
(none/dns_only/rdap_only/full) on the validation split and measures, per
condition, Brier/ECE for shared vs per-condition on validation.

DISCIPLINE: this is a serving refinement verified on VALIDATION only. The
one-time test evaluations of Phases B/C/D used the shared calibrator and
are NOT recomputed. In-sample caveat (2 sigmoid params per condition,
validation rows) matches the shared calibrator's own caveat - documented.

Usage: python -m model.calibrate_conditions
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from appconfig import get_config, setup_console_logging
from features.evidence_features import (
    evidence_matrix_for_domains,
    load_evidence_matrix,
    mask_groups,
)
from features.url_utils import is_ip_hostname
from model.finalize import expected_calibration_error
from model.split import get_or_create_splits

logger = logging.getLogger(__name__)

_CONDITIONS = (("none", ("dns", "rdap")), ("dns_only", ("rdap",)),
               ("rdap_only", ("dns",)), ("full", ()))


def condition_matrices(bundle, df, matrix):
    domains = [None if is_ip_hostname(str(d)) else str(d)
               for d in df["registrable_domain"]]
    E = evidence_matrix_for_domains(domains, matrix)
    X_url = bundle["url_builder"].transform(df["url"].tolist())
    out = {}
    for cond, masks in _CONDITIONS:
        Ec = mask_groups(E, masks) if masks else E
        out[cond] = hstack([X_url, csr_matrix(Ec)], format="csr")
    return out


def main() -> int:
    setup_console_logging()
    cfg = get_config()
    bundle = joblib.load(cfg["model_v2"]["model_path"])
    _, val_df, _, _ = get_or_create_splits(cfg)
    matrix = load_evidence_matrix(cfg["evidence"]["dir"])
    y = val_df["label"].to_numpy(dtype=int)
    Xc = condition_matrices(bundle, val_df, matrix)
    shared = bundle["calibrated_model"]

    rows, cals = [], {}
    for cond, X in Xc.items():
        p_shared = np.asarray(shared.predict_proba(X))[:, 1]
        cal = CalibratedClassifierCV(FrozenEstimator(bundle["rf_model"]),
                                     method="sigmoid").fit(X, y)
        p_cond = np.asarray(cal.predict_proba(X))[:, 1]
        cals[cond] = cal
        rows.append({
            "condition": cond,
            "brier_shared": round(float(brier_score_loss(y, p_shared)), 4),
            "brier_per_condition": round(float(brier_score_loss(y, p_cond)), 4),
            "ece_shared": round(float(expected_calibration_error(y, p_shared)), 4),
            "ece_per_condition": round(float(expected_calibration_error(y, p_cond)), 4),
        })
        logger.info("condition %-9s brier %.4f -> %.4f | ece %.4f -> %.4f",
                    cond, rows[-1]["brier_shared"], rows[-1]["brier_per_condition"],
                    rows[-1]["ece_shared"], rows[-1]["ece_per_condition"])

    out_dir = Path(cfg["model_v2"]["dir"])
    joblib.dump(cals, out_dir / "condition_calibrators.joblib")
    (out_dir / "condition_calibration.json").write_text(json.dumps({
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "validation_table": rows,
        "note": ("Per-condition sigmoid calibrators fitted on the validation "
                 "split (in-sample for 2 sigmoid parameters each; same caveat "
                 "class as the shared calibrator). Deployed for the live "
                 "cascade. The one-time test evaluations (Phases B/C/D) used "
                 "the shared calibrator and are NOT recomputed."),
    }, indent=2), encoding="utf-8")
    improved = sum(1 for r in rows if r["brier_per_condition"] <= r["brier_shared"])
    logger.info("per-condition calibration <= shared Brier in %d/4 conditions; "
                "wrote condition_calibrators.joblib + condition_calibration.json",
                improved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
