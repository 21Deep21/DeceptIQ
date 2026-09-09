"""Domain-aware train/validation/test splitting (Phase 2).

DATA-LEAKAGE PREVENTION (core of the evaluation design):
a random URL-level split would place e.g. evil-example.com/login in
TRAIN and evil-example.com/account in TEST - the model has effectively
already seen that domain, inflating performance. Every registrable
domain is therefore assigned to EXACTLY ONE split and all of its URLs
follow it there.

A domain carrying both labels (a popular ranked domain that also
appears in a threat feed, e.g. a compromised host) is assigned the
MALICIOUS class for splitting purposes so all its rows stay together.

Splits are persisted under data/processed/splits/ with a manifest and
reused across runs: the TEST split is frozen and is evaluated exactly
once (Phase 3, after model selection, calibration and thresholding).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from appconfig import get_config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["url", "label", "source", "threat_type", "registrable_domain"]


def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["paths"]["processed_dataset_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path} - run Phase 1 (python -m dataset.download_data) first"
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")
    df["label"] = df["label"].astype(int)
    return df


def _split_stats(df, train_df, val_df, test_df, fracs, seed) -> Dict[str, Any]:
    def label_counts(d: pd.DataFrame) -> Dict[str, int]:
        return {str(k): int(v) for k, v in d["label"].value_counts().items()}

    mal = set(df.loc[df["label"] == 1, "registrable_domain"])
    ben = set(df.loc[df["label"] == 0, "registrable_domain"])
    d_tr = set(train_df["registrable_domain"])
    d_va = set(val_df["registrable_domain"])
    d_te = set(test_df["registrable_domain"])
    return {
        "seed": seed,
        "fracs": {"train": fracs[0], "val": fracs[1], "test": fracs[2]},
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "by_label": {
            "train": label_counts(train_df),
            "val": label_counts(val_df),
            "test": label_counts(test_df),
        },
        "domains": {"train": len(d_tr), "val": len(d_va), "test": len(d_te)},
        "dual_label_domains": len(mal & ben),
        "domain_overlap": {
            "train_val": len(d_tr & d_va),
            "train_test": len(d_tr & d_te),
            "val_test": len(d_va & d_te),
        },
    }


def domain_aware_split(
    df: pd.DataFrame, seed: int, fracs: Tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split by registrable domain (malicious label wins conflicts).

    Returns (train_df, val_df, test_df, stats). Raises RuntimeError if
    any domain would appear in more than one split (never expected).
    """
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {fracs}")

    domain_label: Dict[str, int] = {}
    for domain, label in zip(df["registrable_domain"], df["label"]):
        if int(label) == 1:
            domain_label[domain] = 1
    for domain, label in zip(df["registrable_domain"], df["label"]):
        domain_label.setdefault(domain, 0)

    rng = random.Random(seed)
    domain_to_split: Dict[str, int] = {}
    for label in (0, 1):
        domains = sorted(d for d, lab in domain_label.items() if lab == label)
        rng.shuffle(domains)
        n = len(domains)
        n_train = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        for d in domains[:n_train]:
            domain_to_split[d] = 0
        for d in domains[n_train : n_train + n_val]:
            domain_to_split[d] = 1
        for d in domains[n_train + n_val :]:
            domain_to_split[d] = 2

    assigned = df["registrable_domain"].map(domain_to_split)
    if assigned.isna().any():
        raise RuntimeError("internal error: unmapped registrable domain during splitting")
    assigned = assigned.astype(int)

    train_df = df[assigned == 0].reset_index(drop=True)
    val_df = df[assigned == 1].reset_index(drop=True)
    test_df = df[assigned == 2].reset_index(drop=True)

    stats = _split_stats(df, train_df, val_df, test_df, fracs, seed)
    if any(stats["domain_overlap"].values()):
        raise RuntimeError(f"domain leakage across splits: {stats['domain_overlap']}")
    return train_df, val_df, test_df, stats


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable content fingerprint (sha256 over label|url).

    Row-count alone cannot detect composition changes: v1.1's daily
    accumulation merged new URLs while the total stayed 10,000 rows.
    The fingerprint forces a split rebuild whenever the CONTENT changes,
    so stale splits are never silently reused for a retrain.
    """
    import hashlib
    h = hashlib.sha256()
    for url, label in zip(df["url"].tolist(), df["label"].tolist()):
        h.update(f"{label}|{url}".encode("utf-8"))
    return h.hexdigest()


def get_or_create_splits(
    cfg: Dict[str, Any], force: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load frozen splits if they match the current dataset+config, else build."""
    df = load_dataset(cfg)
    mcfg = cfg["model"]["split"]
    fracs = (float(mcfg["train_frac"]), float(mcfg["val_frac"]), float(mcfg["test_frac"]))
    seed = int(cfg["dataset"]["random_seed"])
    split_dir = Path(cfg["paths"]["processed_data_dir"]) / "splits"
    manifest_path = split_dir / "manifest.json"
    expected_fracs = {"train": fracs[0], "val": fracs[1], "test": fracs[2]}

    if manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("dataset_rows") == len(df)
                and manifest.get("seed") == seed
                and manifest.get("fracs") == expected_fracs
                and manifest.get("dataset_sha256") == dataset_fingerprint(df)
            ):
                train_df = pd.read_csv(split_dir / "train.csv")
                val_df = pd.read_csv(split_dir / "val.csv")
                test_df = pd.read_csv(split_dir / "test.csv")
                if (len(train_df), len(val_df), len(test_df)) == (
                    manifest["rows"]["train"],
                    manifest["rows"]["val"],
                    manifest["rows"]["test"],
                ):
                    logger.info("reusing frozen splits from %s", split_dir)
                    stats = _split_stats(df, train_df, val_df, test_df, fracs, seed)
                    return train_df, val_df, test_df, stats
            logger.warning("split manifest does not match current dataset/config - rebuilding")
        except Exception as exc:  # corrupted/missing split files -> rebuild
            logger.warning("could not reuse existing splits (%s) - rebuilding", exc)

    train_df, val_df, test_df, stats = domain_aware_split(df, seed, fracs)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.to_csv(split_dir / f"{name}.csv", index=False)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_rows": len(df),
        "dataset_sha256": dataset_fingerprint(df),
        "seed": seed,
        "fracs": expected_fracs,
        "rows": stats["rows"],
        "by_label": stats["by_label"],
        "domains": stats["domains"],
        "dual_label_domains": stats["dual_label_domains"],
        "domain_overlap": stats["domain_overlap"],
        "note": (
            "Domains are assigned to exactly one split. test.csv is FROZEN: not used "
            "for feature fitting, model selection, hyperparameter or threshold "
            "decisions; evaluated once in Phase 3 after the final model is fixed."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote frozen splits + manifest to %s", split_dir)
    return train_df, val_df, test_df, stats
