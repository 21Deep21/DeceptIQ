"""Phase D - replay analyses of the acquisition policies (draft section 6).

Five analyses, all replay-based (no network, no retraining, frozen logs):
  D1 budget analysis    - sweep latency/request budgets -> cost-performance
                          curve (validation side; test quoted separately)
  D2 stability rework   - CV-fold gain-consistency stability (saturation
                          resistant), used to RERANK P1 (P1'): a pre-registered
                          policy-definition revision. The Phase C main
                          comparison is NOT recomputed.
  D3 failure analysis   - disable DNS/RDAP; inject timeout rates; measure
                          recall/F1/confidence shifts (validation side)
  D4 ablation           - remove score terms (value/availability/stability/
                          latency) -> request allocation + F1 response
  D5 figures            - policy comparison, budget curves, evidence profiles,
                          failure impact -> model_v2/analysis_figures/

Guarding: validation-side analyses re-run freely (that is the point of
replay). TEST-side new quantities (budget curve on test) are written to
policy_analysis.json under a one-time guard (--force re-runs, recorded).
Phase C's frozen test table is QUOTED, never recomputed.

Usage:  python -m model.analyze_policies [--force]
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedGroupKFold

from appconfig import get_config, setup_console_logging
from features.evidence_features import (
    evidence_log_fingerprint,
    evidence_matrix_for_domains,
    load_evidence_matrix,
)
from features.url_utils import is_ip_hostname
from model.acquisition import (
    GROUPS,
    GroupProfile,
    build_profiles,
    load_records,
    precompute_states,
    replay_policy,
    select_band,
    summarize,
)
from model.split import get_or_create_splits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- plumbing

def _ctx(df):
    domains = [None if is_ip_hostname(str(d)) else str(d)
               for d in df["registrable_domain"]]
    return df["url"].tolist(), df["label"].to_numpy(dtype=int), domains


def _states_for(bundle, df, matrix):
    domains = [None if is_ip_hostname(str(d)) else str(d)
               for d in df["registrable_domain"]]
    E = evidence_matrix_for_domains(domains, matrix)
    X_url = bundle["url_builder"].transform(df["url"].tolist())

    def predict_fused(Ec):
        X = hstack([X_url, csr_matrix(Ec)], format="csr")
        return bundle["calibrated_model"].predict_proba(X)[:, 1]

    return precompute_states(predict_fused, E)


def _f1(states, cond, y, mask, thr):
    p = np.asarray(states[cond])[mask]
    yy = np.asarray(y)[mask]
    m = ((p >= thr).astype(int))
    tp = int(((m == 1) & (yy == 1)).sum()); fp = int(((m == 1) & (yy == 0)).sum())
    fn = int(((m == 0) & (yy == 1)).sum())
    return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def _rescore(profiles: Dict[str, GroupProfile], drop: Tuple[str, ...]
             ) -> Dict[str, GroupProfile]:
    """Recompute scores with named terms neutralized (weights -> 0)."""
    w = {"value": 1.0, "availability": 1.0, "stability": 1.0, "latency": 1.0}
    for t in drop:
        w[t] = 0.0
    out = {}
    for g, p in profiles.items():
        score = (((max(p.mean_abs_dp, 1e-9)) ** w["value"])
                 * ((max(p.availability, 1e-9)) ** w["availability"])
                 * ((max(p.stability, 1e-9)) ** w["stability"])
                 * (((p.p95_ms / 1000.0) + 1e-6) ** (-w["latency"])))
        out[g] = GroupProfile(**{**p.__dict__, "score": score})
    return out


def _summarize_states(states, y, mask, thr) -> Dict[str, Any]:
    return {"f1_none": round(_f1(states, "none", y, mask, thr), 4),
            "f1_dns": round(_f1(states, "dns_only", y, mask, thr), 4),
            "f1_rdap": round(_f1(states, "rdap_only", y, mask, thr), 4),
            "f1_full": round(_f1(states, "full", y, mask, thr), 4)}


# ---------------------------------------------------------------- D1 budget

def budget_analysis(states_val, y_val, dom_val, records, profiles, thr, band):
    """Validation-side cost-performance curve over budgets."""
    rows = []
    for max_req in (1, 2):
        for lat in (1000.0, 2000.0, 4000.0, 8000.0, 16000.0, None):
            recs = replay_policy("P1", dom_val, y_val, dom_val, states_val,
                                 records, profiles, thr, band,
                                 max_requests=max_req, latency_budget_ms=lat)
            s = summarize(recs, thr)
            rows.append({"policy": "P1", "max_requests": max_req,
                         "latency_budget_ms": lat, "f1": s["f1"],
                         "recall": s["recall"], "precision": s["precision"],
                         "fpr": s["fpr"], "mean_requests": s["mean_requests"],
                         "pct_escalated": s["pct_escalated"],
                         "latency_p95_ms": s["latency_p95_ms"]})
            logger.info("budget(max_req=%s, lat=%s): f1=%.4f mean_req=%.3f "
                        "escal=%.3f p95=%.0f", max_req, lat, s["f1"],
                        s["mean_requests"], s["pct_escalated"],
                        s["latency_p95_ms"])
    df = pd.DataFrame(rows)
    logger.info("budget analysis: %d configurations (validation)", len(df))
    return df


# ------------------------------------------------- D2 stability rework/P1'

def cv_stability(states_train, y_train, dom_train, thr, seed=42, k=5):
    """5-fold (domain-grouped) gain consistency within train: for each fold,
    F1(gain of each group's condition vs none). Stability = agreement of the
    gain SIGN and magnitude across folds - saturation-resistant because
    held-out folds are not memorized. Returns {group: stability in [0,1]}."""
    # Group keys must be sortable by numpy: IP-literal rows carry None,
    # which np.unique cannot sort against strings. Sanitize to a sentinel;
    # those rows are excluded from the gain masks below regardless.
    domains = np.asarray([d if d is not None else "__ip_host__"
                          for d in dom_train], dtype=object)
    hosted = np.asarray([d is not None for d in dom_train], dtype=bool)
    y = np.asarray(y_train)
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    gains: Dict[str, List[float]] = {g: [] for g in GROUPS}
    for tr_idx, ho_idx in skf.split(domains, y, groups=domains):
        mask = np.zeros(len(y), dtype=bool); mask[ho_idx] = True
        mask &= hosted                                             # hosted only
        if mask.sum() < 20:
            continue
        base = _f1(states_train, "none", y, mask, thr)
        for g, cond in (("dns", "dns_only"), ("rdap", "rdap_only")):
            gains[g].append(_f1(states_train, cond, y, mask, thr) - base)
    stab: Dict[str, float] = {}
    for g, vals in gains.items():
        vals = [v for v in vals if v is not None]
        if not vals:
            stab[g] = 0.0; continue
        signs = [np.sign(v) for v in vals]
        mag = float(np.mean(vals))
        sign_agreement = abs(sum(signs)) / len(signs)          # 1.0 if uniform
        stab[g] = round(max(0.0, sign_agreement * min(1.0, abs(mag) * 20)), 4)
    return stab


# ---------------------------------------------------------------- D3 failure

def _disable(records, group):
    r = copy.deepcopy(records)
    r.pop(group, None)
    return r


def _inject_timeouts(records, group, rate, seed=7):
    """Deterministically relabel a fraction of ok records as timeout."""
    r = copy.deepcopy(records)
    doms = sorted(d for d, rec in r.get(group, {}).items()
                  if rec.get("status") == "ok")
    n_hit = int(round(rate * len(doms)))
    hit = set(doms[:n_hit])                                    # sorted -> reproducible
    for d in hit:
        r[group][d] = {**r[group][d], "status": "timeout"}
    return r


def failure_analysis(states_val, y_val, dom_val, records, profiles, thr, band,
                     policies=("B2", "B3", "P1")):
    rows = []
    scenarios = ([("baseline", records)]
                 + [(f"disable_{g}", _disable(records, g)) for g in GROUPS]
                 + [(f"timeouts_{g}_{int(rate*100)}pct",
                     _inject_timeouts(records, g, rate))
                    for g in GROUPS for rate in (0.2, 0.5)])
    for name, recs in scenarios:
        for pol in policies:
            out = replay_policy(pol, dom_val, y_val, dom_val, states_val,
                                recs, profiles, thr, band)
            s = summarize(out, thr)
            rows.append({"scenario": name, "policy": pol,
                         **{k: s[k] for k in ("f1", "recall", "precision",
                                              "fpr", "mean_requests",
                                              "acquisition_success_rate")}})
            logger.info("failure[%s / %s]: f1=%.4f rec=%.4f prec=%.4f "
                        "succ=%s", name, pol, s["f1"], s["recall"],
                        s["precision"], s["acquisition_success_rate"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D4 ablation

def ablation_analysis(states_val, y_val, dom_val, records, profiles, thr, band):
    rows = []
    variants = [("full_score", ()), ("value_only", ("availability", "stability", "latency")),
                ("no_availability", ("availability",)), ("no_stability", ("stability",)),
                ("no_latency", ("latency",))]
    for name, drop in variants:
        prof = _rescore(profiles, drop)
        out = replay_policy("P1", dom_val, y_val, dom_val, states_val,
                            records, prof, thr, band)
        s = summarize(out, thr)
        alloc = {g: sum(1 for r in out for q in r.requests if q["group"] == g)
                 for g in GROUPS}
        rows.append({"variant": name, "f1": s["f1"], "recall": s["recall"],
                     "precision": s["precision"],
                     "mean_requests": s["mean_requests"],
                     "req_dns": alloc["dns"], "req_rdap": alloc["rdap"]})
        logger.info("ablation[%s]: f1=%.4f mean_req=%.3f alloc dns=%d rdap=%d",
                    name, s["f1"], s["mean_requests"], alloc["dns"], alloc["rdap"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- D5 figures

def make_figures(out_dir: Path, policy_eval: Dict[str, Any],
                 budget_df: pd.DataFrame, failure_df: pd.DataFrame,
                 ablation_df: pd.DataFrame, profiles: Dict[str, GroupProfile]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    MAL, BEN, BLUE, GRAY = "#c44e52", "#55a868", "#4c72b0", "#6d6d6d"
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "font.family": "serif",
                         "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "legend.fontsize": 8})

    # Fig P1: test policy comparison (quoted from policy_evaluation.json)
    t = policy_eval["results"]["test"]
    pols = ["B1", "B2", "B3", "P1"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    x = np.arange(len(pols))
    axes[0].bar(x, [t[p]["f1"] for p in pols], color=[GRAY, MAL, BLUE, BEN])
    axes[0].set_xticks(x); axes[0].set_xticklabels(pols)
    axes[0].set_ylim(0.85, 0.87); axes[0].set_ylabel("test F1")
    axes[0].set_title("(a) Detection quality")
    axes[1].bar(x, [t[p]["mean_requests"] for p in pols],
                color=[GRAY, MAL, BLUE, BEN])
    axes[1].set_xticks(x); axes[1].set_xticklabels(pols)
    axes[1].set_ylabel("mean requests / URL"); axes[1].set_title("(b) Request cost")
    fig.savefig(out_dir / "figP1_policy_comparison.png"); plt.close(fig)

    # Fig P2: budget curves (validation)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for max_req, sub in budget_df.groupby("max_requests"):
        sub = sub.sort_values("latency_budget_ms",
                              key=lambda s: s.fillna(1e9))
        xs = ["none" if pd.isna(v) else f"{int(v/1000)}k"
              for v in sub["latency_budget_ms"]]
        ax.plot(range(len(sub)), sub["f1"], "o-", label=f"max_req={max_req}")
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(xs)
    ax.set_xlabel("latency budget"); ax.set_ylabel("validation F1 (P1)")
    ax.legend()
    fig.savefig(out_dir / "figP2_budget_curves.png"); plt.close(fig)

    # Fig P3: failure impact (validation)
    base = failure_df[failure_df.scenario == "baseline"].set_index("policy")
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    scen = [s for s in failure_df.scenario.unique() if s != "baseline"]
    w = 0.25
    for i, pol in enumerate(("B2", "B3", "P1")):
        vals = [failure_df[(failure_df.scenario == s) & (failure_df.policy == pol)]
                .f1.iloc[0] - base.f1.loc[pol] for s in scen]
        ax.bar(np.arange(len(scen)) + i * w, vals, w, label=pol)
    ax.set_xticks(np.arange(len(scen)) + w); ax.set_xticklabels(scen, fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("delta F1 vs baseline"); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.savefig(out_dir / "figP3_failure_impact.png"); plt.close(fig)

    # Fig P4: evidence group profiles (from Phase A measurements)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    gs = list(GROUPS)
    axes[0].bar(gs, [profiles[g].availability for g in gs], color=[BEN, MAL])
    axes[0].set_ylabel("availability"); axes[0].set_ylim(0, 1)
    axes[0].set_title("(a) Group availability")
    axes[1].bar(gs, [profiles[g].p95_ms / 1000 for g in gs], color=[BEN, MAL])
    axes[1].set_ylabel("p95 latency (s)"); axes[1].set_title("(b) Group tail latency")
    fig.savefig(out_dir / "figP4_group_profiles.png"); plt.close(fig)
    print(f"figures -> {out_dir}/figP1..P4")


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    setup_console_logging()
    ap = argparse.ArgumentParser(description="Phase D policy analyses")
    ap.add_argument("--force", action="store_true",
                    help="re-run the one-time test-side budget analysis (recorded)")
    args = ap.parse_args(argv)

    cfg = get_config()
    out_path = Path(cfg["model_v2"]["dir"]) / "policy_analysis.json"
    if out_path.exists() and not args.force:
        try:
            if json.loads(out_path.read_text()).get("test_budget_evaluated"):
                logger.error("test-side budget analysis already evaluated at %s; "
                             "refusing (one-time). Use --force.",
                             json.loads(out_path.read_text()).get("test_budget_evaluated_at"))
                return 2
        except json.JSONDecodeError:
            logger.warning("policy_analysis.json unreadable; continuing")

    # frozen context (identical to Phase C)
    bundle = joblib.load(cfg["model_v2"]["model_path"])
    thr = float(bundle["threshold"])
    policy_eval = json.loads((Path(cfg["model_v2"]["dir"])
                              / "policy_evaluation.json").read_text())
    band = tuple(policy_eval["band"])
    train_df, val_df, test_df, _ = get_or_create_splits(cfg)
    matrix = load_evidence_matrix(cfg["evidence"]["dir"])
    records = load_records(cfg["evidence"]["dir"])
    states = {"train": _states_for(bundle, train_df, matrix),
              "val": _states_for(bundle, val_df, matrix),
              "test": _states_for(bundle, test_df, matrix)}
    urls_tr, y_tr, dom_tr = _ctx(train_df)
    urls_va, y_va, dom_va = _ctx(val_df)
    urls_te, y_te, dom_te = _ctx(test_df)
    tv_dom = set(dom_tr) | set(dom_va); tv_dom.discard(None)
    tv_records = {g: {d: r for d, r in records[g].items() if d in tv_dom}
                  for g in GROUPS}
    profiles = build_profiles(states["train"], y_tr,
                              [d is not None for d in dom_tr],
                              states["val"], y_va,
                              [d is not None for d in dom_va],
                              tv_records, thr)

    hosted_va = np.asarray([d is not None for d in dom_va], dtype=bool)

    # D1: validation budget analysis
    logger.info("=== D1: budget analysis (validation) ===")
    budget_df = budget_analysis(states["val"], y_va, dom_va, records,
                                profiles, thr, band)

    # D2: stability rework + P1' reranking (test side, one-time)
    logger.info("=== D2: stability rework ===")
    cv_stab = cv_stability(states["train"], y_tr, dom_tr, thr)
    logger.info("cv stability: %s", cv_stab)
    prof2 = {}
    for g, p in profiles.items():
        newscore = (max(p.mean_abs_dp, 1e-9) * max(p.availability, 1e-9)
                    * max(cv_stab.get(g, 0.0), 1e-9) / ((p.p95_ms / 1000.0) + 1e-6))
        prof2[g] = GroupProfile(**{**p.__dict__,
                                   "stability": cv_stab.get(g, 0.0),
                                   "score": newscore})
    logger.info("reranked scores: %s", {g: round(prof2[g].score, 4) for g in GROUPS})
    p1p_val = summarize(replay_policy("P1", urls_va, y_va, dom_va, states["val"],
                                      records, prof2, thr, band), thr)
    p1p_test = summarize(replay_policy("P1", urls_te, y_te, dom_te, states["test"],
                                       records, prof2, thr, band), thr)
    logger.info("P1' validation: f1=%.4f mean_req=%.3f | P1' test (one-time): "
                "f1=%.4f mean_req=%.3f", p1p_val["f1"], p1p_val["mean_requests"],
                p1p_test["f1"], p1p_test["mean_requests"])

    # one-time test-side budget curve (P1 only, coarse)
    test_budget_rows = []
    for max_req in (1, 2):
        for lat in (2000.0, 6000.0, None):
            recs = replay_policy("P1", urls_te, y_te, dom_te, states["test"],
                                 records, profiles, thr, band,
                                 max_requests=max_req, latency_budget_ms=lat)
            s = summarize(recs, thr)
            test_budget_rows.append({"max_requests": max_req,
                                     "latency_budget_ms": lat,
                                     **{k: s[k] for k in ("f1", "recall",
                                                          "precision",
                                                          "mean_requests",
                                                          "latency_p95_ms")}})
            logger.info("test budget(max_req=%s lat=%s): f1=%.4f mean_req=%.3f",
                        max_req, lat, s["f1"], s["mean_requests"])

    # D3: failure analysis (validation)
    logger.info("=== D3: failure analysis (validation) ===")
    failure_df = failure_analysis(states["val"], y_va, dom_va, records,
                                  profiles, thr, band)

    # D4: ablation (validation)
    logger.info("=== D4: score-term ablation (validation) ===")
    ablation_df = ablation_analysis(states["val"], y_va, dom_va, records,
                                    profiles, thr, band)

    # D5: figures
    fig_dir = Path(cfg["model_v2"]["dir"]) / "analysis_figures"
    make_figures(fig_dir, policy_eval, budget_df, failure_df, ablation_df,
                 profiles)
    budget_df.to_csv(Path(cfg["model_v2"]["dir"]) / "budget_analysis.csv",
                     index=False)
    failure_df.to_csv(Path(cfg["model_v2"]["dir"]) / "failure_analysis.csv",
                      index=False)
    ablation_df.to_csv(Path(cfg["model_v2"]["dir"]) / "ablation_analysis.csv",
                       index=False)

    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out = {
        "created_at": stamp,
        "quoted_phase_c_test": policy_eval["results"]["test"],
        "d1_budget_validation": json.loads(budget_df.to_json(orient="records")),
        "d2_stability_rework": {
            "cv_stability": cv_stab,
            "reranked_scores": {g: prof2[g].score for g in GROUPS},
            "p1_prime_validation": p1p_val,
            "p1_prime_test_onetime": p1p_test,
            "note": "pre-registered policy-definition revision; the Phase C "
                    "main comparison is NOT recomputed",
        },
        "d1_test_budget_onetime": test_budget_rows,
        "d3_failure_validation": json.loads(failure_df.to_json(orient="records")),
        "d4_ablation_validation": json.loads(ablation_df.to_json(orient="records")),
        "provenance": {
            "evidence_log_fingerprint": evidence_log_fingerprint(cfg["evidence"]["dir"]),
            "model_v2_created_at": bundle.get("created_at"),
            "threshold": thr, "band": list(band),
        },
        "test_budget_evaluated": True, "test_budget_evaluated_at": stamp,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s (+ budget/failure/ablation CSVs + analysis_figures/)",
                out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
