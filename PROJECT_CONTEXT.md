# PROJECT CONTEXT — READ THIS FIRST
## Handoff / state document for AI assistants and collaborators

This file is the single source of truth for the project's state, measured
facts, and working rules. It exists so that ANY assistant, collaborator, or
future session can continue with identical context. If a fact is not in
this file, re-measure it — never invent it.

Environment: Fedora Workstation, Python 3.14.7, venv `.venv`, repo
https://github.com/21Deep21/DeceptIQ (private). Never commit
secrets; the VirusTotal key lives only in the git-ignored `.env`.

---

## 1. AGENT WORKING RULES (non-negotiable)

1. **Read before acting:** this file, `README.md`, `config.yaml`, and the
   relevant module before changing anything.
2. **Never fabricate numbers.** The ledger below (section 3) is the source
   of truth. If something must be re-measured, re-measure it and update
   the ledger in the same commit.
3. **Verify before claiming.** Nothing "works" until the user has run the
   verification commands and pasted ACTUAL OUTPUT.
4. **Stop-and-wait discipline.** Build in phases: explain → files →
   tests → verification commands → STOP for user output. If the user
   pastes commands WITHOUT output, ask for the output.
5. **Defensive security only.** The analyzed URL is NEVER visited.
6. **No secrets in the repo.** Keys via environment / `.env` only.
   Credential redaction is a tested invariant — keep it that way.
7. **Test gates:** `python -m compileall -q ...` then `python -m pytest
   tests -q` must pass before any claim of working state.
8. **One-time test discipline:** `model.finalize` refuses re-runs (exit 2).
   `--force` re-evaluates and is RECORDED. Never tune on test results.
9. **No casual retraining.** The deployed bundle is frozen; retraining is
   a deliberate full procedure (section 6).

## 2. CURRENT STATE (v2.0.0)

- **Deployed v1 stack (default /predict):** unchanged - RandomForest @ 0.0706,
  calibrated, serving since Phase 3R.
- **v2.0 research increment (Phases A-E) COMPLETE:** frozen evidence base
  (DNS 0.949 / RDAP 0.624 availability, 3,065 domains); fusion model
  (model_v2/, threshold 0.0392, 4-condition masking, one-time tested);
  policy replay B1/B2/B3/P1 (one-time; B3/P1 ~99.4% of B2 F1 at 25% of
  requests); Phase D analyses (sub-6s SLA floor; budget-as-regularizer;
  failure-immune classification; P1 most graceful degradation; ablation
  flat at two groups); Phase E live cascade (opt-in include_evidence on
  /predict, per-condition calibrators, fusion SHAP with evidence_total,
  console ACQUISITION TRACE panel).
- **Three one-time guards armed:** model_v2 test, policy test replay,
  test-side budget analysis. Re-runs require --force and are recorded.

---

## 3. MEASURED-FACTS LEDGER (all measured on this machine)

### Dataset revision 1R (the project's key arc)
- v1 benign class was 100% constructed homepages → "presence of a path"
  became near-decisive malicious evidence (SHAP: path_length +6.8…+11.5).
- Fix: harvested 16,264 real benign deep-link URLs from public sitemaps
  (97/169 curated domains; robots.txt + sitemap files ONLY, pages never
  visited). Benign path>1 share: **0.000 → 0.459**; mean path length
  **1.0 → 19.3** (malicious 7.5).
- Final training dataset: 10,000 rows = 5,000 malicious (4,943 URLhaus
  malware_distribution + 57 OpenPhish phishing) + 5,000 benign
  (2,665 Majestic homepages + 2,335 sitemap). 6,270 domains.
- Residual skews (documented): benign 96.9% https / 0% IP-host vs
  malicious 8.7% / 90.8%.

### Splits (frozen, domain-aware 60/20/20, seed 42)
train 6,056 rows / 3,762 domains · val 2,011 / 1,254 · test 1,933 / 1,254 ·
domain overlap 0/0/0 · dual-label domains 1 · manifest now carries a
content sha256 fingerprint (forces rebuild when dataset content changes).

### Model comparison (1R, grouped 5-fold CV / validation @0.5)
| model | CV recall | CV prec | CV F1 | val F1 | val FP/FN |
|---|---|---|---|---|---|
| logreg | 0.9758 | 0.9768 | 0.9760 | 0.9646 | 36/35 |
| **random_forest** | 0.9862 | 0.9757 | 0.9804 | 0.9738 | 35/18 |
| xgboost | 0.9946 | 0.8887 | 0.9385 | 0.9314 | 142/5 |
| lightgbm | 0.9970 | 0.8811 | 0.9354 | 0.9361 | 129/7 |
| catboost | 0.9899 | 0.9429 | 0.9657 | 0.9565 | 77/13 |
| stacking | — | — | — | 0.9738 | 34/19 |

Stacking meta coefficients: RF 5.68, CatBoost 4.61, XGBoost 0.31,
LightGBM -0.09. Stacking NOT retained (PR-AUC delta 0.0001 vs RF).
LightGBM (v1 winner) demoted: post-fix validation precision 0.8853 (fp=129).

### Final one-time test (deployed RF, calibrated @0.0706, 1,933 rows)
accuracy 0.8981 · recall 0.9882 · precision 0.8451 · F1 0.9111 ·
ROC-AUC 0.9922 · PR-AUC 0.9938 · tn=727 fp=185 fn=12 tp=1009 ·
Brier 0.0449 · ECE(10-bin) 0.0309 (raw: 0.0473/0.0582).
Validation→test generalization gap: benign FP rate 3.5% → 20.3% —
reported, NOT re-tuned (no test-based optimization).
Threshold policy: recall >= 0.97 AND precision >= 0.90, max recall → max
precision → lowest threshold; selected ON VALIDATION ONLY (51 qualifying).

### v1 (archived, for comparison only)
LightGBM @0.7833: test F1 0.9985, precision 1.0000, recall 0.9970, fp=0 —
measured an artificially easy task (benign = homepages only).

### Ablation (validation only; test untouched)
combined F1 0.9738 (5,026 features) · lexical-only 0.9599 (26) ·
ngram-only 0.9630 (5,000) — integrity-checked against the Phase 2R record.

### Error analysis (from test_errors.csv)
185 FP — ALL from sitemap benign; worst: matplotlib.org/3.x version
paths at p=0.931; FP median 0.5487. 12 FN (7 urlhaus, 5 openphish).

### SHAP
shap.TreeExplainer (exact TreeSHAP). Space is MODEL-DEPENDENT and reported
per explanation: LightGBM → log-odds; sklearn RF → probability (no
margin output). For the deployed RF the top evidence for IP-host URLs is
has_ip_hostname (v1's was path_length — the bias artifact).

---

## 4. ARCHITECTURE / FILE MAP
dataset/ (download_data, sitemap_sources, sitemap_harvest_only,
build_dataset, accumulate_feeds) → features/ (url_utils, entropy,
feature_extraction, feature_builder, vectorizer, ioc_extraction,
whois_features, dns_features) → model/ (split, train, stacking, finalize,
analysis + artifacts) → services/ (prediction_service, shap_service,
scan_store, alert_logger, virustotal_service) → app.py → templates/ +
static/ (SOC console). Full descriptions in README.md sections 4-5.

## 5. RUN / TEST / VERIFY (quick reference)

    source .venv/bin/activate
    python app.py                    # console at http://127.0.0.1:5000
    python -m pytest tests -q        # 173 collected, 166 passed, 7 skipped
    python -m model.analysis         # ablation + error analysis
    python -m dataset.accumulate_feeds --force   # daily snapshot
    python -m dataset.build_dataset  # offline rebuild + merge snapshots
    python -m model.train            # full retrain (splits rebuild)
    python -m model.finalize         # guard: refuses if already evaluated

Safe demo URLs: https://example.com/ (safe LOW 0.0026) ·
http://192.0.2.10:8080/login.php (phishing CRITICAL 0.9782) ·
https://user:pw123@example.com/l?password=hunter2 (phishing MEDIUM,
shows credential redaction).

## 6. PENDING / FUTURE WORK
- **Scheduled retrain (the "2R" procedure)** after 1-2+ weeks of daily
  accumulation: build_dataset → model.train (fingerprint forces split
  rebuild) → model.finalize --force (recorded) → restart + battery →
  honest before/after table. Cron: see accumulate_feeds docstring.
- Improvement tiers: query-rich benign corpus (Common Crawl URL lists),
  RF hyperparameter tuning, McNemar significance tests, gunicorn/Redis,
  drift monitoring, STIX export (README "Future Improvements").
- **Rotate the VirusTotal key after the demo** (it was exposed in chat).

## 7. FIXED PITFALLS — do not re-break these

1. **Pickle compatibility:** bundles saved before v1.1 lack the
   feature_mode attribute (pickle bypasses __init__). Read it via
   _effective_mode() / getattr default "combined". Regression test:
   test_v1_pickle_bundle_compatibility.
2. **sklearn-RF SHAP:** shap_values returns shape (n, m, 2) — slice the
   positive class; explanations are in PROBABILITY space; call with
   check_additivity=False (known RF float quirk) and rely on the
   per-URL reconstruction check instead.
3. **Split manifest:** row-count alone missed a same-size content change;
   now sha256-fingerprinted (dataset_fingerprint).
4. **sklearn 1.8+:** penalty= in LogisticRegression is deprecated —
   use the default + C.
5. **One-time test guard:** deliberate, do not bypass casually.
6. **Credential redaction:** userinfo passwords AND credential-like query
   values, end-to-end (API, logs, alerts, DB, VT lookup) — tested.
7. **python-whois failures and DNS failures are Unavailable, never
   phishing signals.**
8. **Heredoc paste-clipping** has corrupted files before — always follow
   big file writes with an integrity check / test run before proceeding.

## 8. COMMIT HISTORY (state lineage)
1c6c65d Phases 0-7 · 191bd01/d698779 Phase 1R · a34efef Phase 1R-3R ·
2213249 Phase 8 (README, v1.0.0) · 3bc5432 v1.1 fix (pickle-safe builder,
split fingerprint, both-class ablation test).

When any measured number changes after a re-run, UPDATE THIS LEDGER in the
same commit. This file must never disagree with the artifacts
(model/model_metrics.csv, model/metadata.json,
data/processed/dataset_stats.json).

---

## 9. V2.0 PITFALLS (append to section 7)

9. **numpy cannot sort object arrays mixing None and str** - sanitize group
   keys to a sentinel before StratifiedGroupKFold (fixed in
   analyze_policies.cv_stability).
10. **pandas stores None budgets as NaN** - check pd.isna() BEFORE int()
    formatting (fixed in figP2 labels).
11. **Commit after green, not before** - violated twice during Phase D
    (committed with failing tests / crashed run); harmless only because
    nothing one-time had been consumed. Restore the gate discipline.
12. **shap TreeExplainer needs a DENSE ndarray** in this build (np.isnan
    validation rejects sparse input) - bit us three times: v1 service,
    Phase 3R, fusion service. Always X.todense() first.
13. **Truncated displays lie** - a test fixture URL copied from a
    70-char-truncated log line was the wrong URL (p0 0.0012, not 0.42).
    Measure fixture values from the bundle, never copy from output.
