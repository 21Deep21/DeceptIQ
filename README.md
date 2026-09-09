# Explainable AI-Based Phishing URL Detection & Cybersecurity Analysis Platform

**MCA minor project — a defensive cybersecurity application.**

The platform classifies URLs as phishing/malicious or legitimate using
lexical and statistical URL features, explains every verdict with SHAP,
deconstructs URLs into indicators of compromise (IOCs), optionally
enriches them with WHOIS/DNS and a VirusTotal cross-check, keeps a scan
history with security alert logging, and presents everything in a
SOC-style investigation console.

**Safety posture: the analyzed URL is NEVER visited.** Only URL strings
and public metadata are analyzed. Threat-feed files are downloaded from
their official publishers; URLs inside them are parsed as strings only.

All results quoted below are **measured** on this project's runs; the
artifacts (`model/model_metrics.csv`, `model/metadata.json`,
`data/processed/dataset_stats.json`) contain the full records.

---

## 1. Project Overview

An end-to-end, reproducible pipeline: multi-source dataset construction →
offline feature engineering (26 lexical features + Shannon entropy +
character n-gram TF-IDF) → domain-aware model training and comparison of
five algorithms → probability calibration and recall-first threshold
selection → SHAP explainability → a Flask backend with SQLite history,
rate limiting, caching and alerting → a professional investigation
console. The project also includes a documented **dataset revision
(v1 → 1R)** in which a measured source bias was found and fixed by
re-running the entire pipeline.

## 2. Problem Statement

Phishing is a primary initial-access vector. URL-string analysis is a
valuable first-line triage because it needs no page fetch, works at
scale, and can run before any human or automated browser touches the
link. Two requirements make this hard in practice: (a) the label noise
and source bias inherent in public feeds, and (b) the need for
*explainable* decisions — a SOC analyst must see *why* a URL was flagged.
This project addresses both: domain-aware evaluation to prevent inflated
results, and per-URL SHAP evidence for every verdict.

## 3. Objectives

1. Build a balanced, multi-source dataset with honest source tracking.
2. Engineer URL features locally (no page fetching).
3. Prevent domain-level train/test leakage via grouped splitting.
4. Compare five ML algorithms with reproducible seeds.
5. Deploy only calibrated probabilities with an explicitly optimized threshold.
6. Explain every prediction with SHAP.
7. Extract IOCs and redact credentials defensively.
8. Serve a SOC-style console with history, batch scanning, rate limiting
   and CEF-style alerts, plus an optional VirusTotal cross-check.

## 4. Architecture

```
 Threat feeds (OpenPhish, URLhaus)      Legitimacy sources
   malware-distribution/phishing        (Majestic ranking, sitemap harvest)
            │                                     │
            └────────────┬────────────────────────┘
                         ▼
        dataset/  download → normalize → clean → dedup →
        per-label domain caps → seeded balancing (benign mix)
                         ▼
        data/processed/urls.csv  (10,000 rows, 6,270 domains)
                         ▼
        model/split.py   domain-aware 60/20/20 frozen splits
                         ▼
        features/  26 lexical + entropy + char(3–5)-gram TF-IDF
                   (TF-IDF fitted on TRAINING data only)
                         ▼
        model/train.py  5 models + stacking, StratifiedGroupKFold
                         ▼
        model/finalize.py  sigmoid calibration on validation,
                           recall-first threshold, ONE-TIME test eval
                         ▼
        model/model.joblib  (calibrated model + raw pipeline + threshold)
                         ▼
        services/  prediction · SHAP · IOC · WHOIS/DNS · VirusTotal
                         ▼
        app.py  Flask: GET / · POST /predict · POST /predict_batch ·
                GET /history · GET /health · GET /api/model-info
                         ▼
        SOC investigation console (templates/ + static/)
```

## 5. Folder Structure

```
phishing-url-analyzer/
├── app.py                     Flask application (routes, rate limiting)
├── appconfig.py               config.yaml loader, .env loader, logging setup
├── config.yaml                all configuration (no secrets)
├── requirements.txt           direct dependencies (pinned: requirements.lock.txt)
├── dataset/
│   ├── download_data.py       feed + sitemap acquisition and orchestration
│   ├── sitemap_sources.py     robots.txt/sitemap harvester (Phase 1R)
│   ├── sitemap_harvest_only.py standalone harvest entry point
│   └── build_dataset.py       cleaning, caps, balancing, persistence
├── features/
│   ├── url_utils.py           validation, normalization, registrable domains
│   ├── entropy.py             Shannon entropy
│   ├── feature_extraction.py  26 lexical features
│   ├── feature_builder.py     dense+TF-IDF matrix transformer (sklearn)
│   ├── vectorizer.py          TF-IDF construction (fitted in training only)
│   ├── ioc_extraction.py      URL deconstruction, flags, credential redaction
│   ├── whois_features.py      WHOIS enrichment (graceful)
│   └── dns_features.py        DNS enrichment (graceful)
├── model/
│   ├── split.py               domain-aware frozen splits
│   ├── train.py               Phase 2 training/evaluation orchestration
│   ├── stacking.py            group-aware stacking ensemble
│   ├── finalize.py            Phase 3: calibration, threshold, persistence
│   ├── model.joblib           DEPLOYED bundle (calibrated RF + threshold)
│   ├── threshold.json         threshold policy, PR-curve sample
│   ├── metadata.json          model card: rationale, test results, hashes
│   ├── model_metrics.csv      every measured CV/validation/test row
│   └── test_errors.csv        final-test misclassifications
├── model_v1/                  archived v1 (LightGBM) artifacts for comparison
├── services/
│   ├── prediction_service.py  orchestration + bounded LRU cache
│   ├── shap_service.py        dual-backend exact TreeSHAP
│   ├── scan_store.py          SQLite history
│   ├── alert_logger.py        CEF-inspired phishing alerts
│   └── virustotal_service.py  optional lookup-only VT v3 client
├── templates/index.html       SOC console page
├── static/css, static/js      console styling and logic (vanilla JS)
├── tests/                     156 collected pytest tests
├── data/raw, data/processed   cached feeds, urls.csv, splits, scans.db
└── logs/app.log               rotating application log
```

## 6. Installation (Fedora Workstation)

```bash
sudo dnf install python3       # usually preinstalled
mkdir -p ~/projects/phishing-url-analyzer && cd ~/projects/phishing-url-analyzer
# place the project files here (git clone or copy)
python3 -m venv .venv
```

## 7. Virtual Environment

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt      # or requirements.lock.txt (pinned)
```

Optional VirusTotal: create a git-ignored `.env` containing
`VT_API_KEY=<your key>` (never commit it; the app reads it at startup).

## 8. Execution

```bash
source .venv/bin/activate

# 1. Dataset (network; ~15-30 min incl. sitemap harvest)
python -m dataset.download_data
# offline rebuild from cached raw files:
python -m dataset.build_dataset

# 2. Train and compare all models (grouped CV; ~5 min)
python -m model.train

# 3. Calibrate, select threshold, ONE-TIME test evaluation, persist bundle
python -m model.finalize          # re-runs refuse (one-time discipline)

# 4. Run the console
python app.py                     # http://127.0.0.1:5000

# 5. Tests
python -m pytest tests -q         # 156 collected, 149 passed, 7 skipped
```

## 9. Dataset Sources (revision 1R, actual)

| Source | Role | Raw records | Final selected |
|---|---|---|---|
| OpenPhish community feed | phishing | 300 | 57 |
| URLhaus (abuse.ch) | malware distribution | 14,289 | 4,943 |
| Majestic Million | legitimate (fallback) | 1,000,000 domains | 2,665 constructed homepages |
| Public sitemaps of 169 curated reputable organizations | legitimate deep links | 16,264 URLs (97 domains OK, 72 yielded nothing — reported) | 2,335 |
| PhishTank | phishing | **unavailable** (registration closed) | 0 |
| Tranco | legitimate | **failed** (HTTP 404) | 0 (Majestic fallback) |

Final: **10,000 rows** = 5,000 malicious (label 1: 4,943 `malware_distribution` + 57 `phishing`, threat types kept distinct) + 5,000 benign (label 0). 6,270 unique registrable domains.

**Dataset revision v1 → 1R (the project's key finding):** v1's benign class
was 100% constructed homepages, so SHAP showed "presence of a path" had
become near-decisive malicious evidence (benign path>1 share was 0.000).
Revision 1R mixed real deep-link URLs harvested from public sitemaps into
the benign class (only robots.txt and sitemap files were fetched; page
URLs were never requested). Measured effect: benign path>1 share
**0.000 → 0.459**, benign mean path length **1.0 → 19.3** (malicious: 7.5).

## 10. Dataset Preprocessing (actual numbers)

Normalization → malformed removal (**31**, all bare-public-suffix hosts) →
exact-URL dedup (0) → cross-label conflict resolution (0, malicious wins) →
per-label domain caps (malicious 5 / benign 25; **19,035** dropped, mostly
URLhaus host spam) → seeded class balancing with the benign mix
(2,335 observed + 2,665 constructed, actual observed fraction 0.47) →
seeded shuffle → `urls.csv` + `dataset_stats.json`.

## 11. URL Normalization

One shared function (`features/url_utils.normalize_url`) used at dataset
construction AND inference (no training-serving skew): trim, reject
interior whitespace/control characters, 2048-char cap, http/https only,
lowercase scheme+host, IDNA/punycode encoding, hostname shape validation,
userinfo/port/path/query/fragment preserved (userinfo is deliberately kept
for the model — `@` obfuscation is signal).

## 12. Lexical Features (26, all offline)

Lengths (URL/host/path/query/fragment), counts (dots, hyphens,
underscores, digits, special chars, slashes), subdomain count, query
parameter count, IP-host flag, `@` flag, HTTPS flag, explicit/unusual
port, percent-encoding count, distinct suspicious-keyword hits,
URL-shortener flag, path segments, digit ratio, special-char ratio,
`url_entropy`, `hostname_entropy`. None is proof of phishing; the model
weighs them jointly.

## 13. Shannon Entropy

`H(X) = -Σ p(x) log2 p(x)` over the character distribution of the URL and
hostname. Higher entropy can indicate randomized/DGA-like strings.
**Limitation: high entropy is NOT proof of phishing** (CDNs, IDN domains).

## 14. WHOIS Features (scan-time enrichment, NOT model input)

Registrar, creation/expiration dates, domain age, days to expiry.
Deliberately excluded from training: bulk WHOIS is impractical (per-TLD
rate limits) and non-reproducible (values change). **WHOIS unavailability
is never a phishing signal** — privacy protection, rate limits, and
unsupported registries all produce `Unavailable`.

## 15. DNS Features (scan-time enrichment, NOT model input)

A-record count/TTL (host), MX presence, NS count and diversity
(registrable domain). Strict timeouts; failures are `Unavailable`, never
verdicts. An authoritative NoAnswer *is* information (recorded as 0/False).

## 16. Character N-gram TF-IDF

Char (3,5)-gram TF-IDF over the URL string captures typosquatting
(`paypa1`), brand impersonation, embedded keywords and obfuscation.
**Fitted only on training data** — inside the pipeline, per CV fold and
per final fit; never on the full dataset before splitting. 5,000 features
max, `min_df=3`, persisted with the model.

## 17. Domain-Level Data Leakage

Random URL-level splitting would place `evil-example.com/login` in train
and `evil-example.com/account` in test — same domain, inflated scores.
Prevention: every registrable domain is assigned to exactly one split;
hosts without a registrable domain are dropped at cleaning time (they
cannot be grouped).

## 18. Group-Aware Splitting (actual)

By registrable domain, 60/20/20, seed 42: train 6,056 rows / 3,762
domains, validation 2,011 / 1,254, test 1,933 / 1,254. **Domain overlap:
0/0/0** (verified programmatically). Dual-label domains (compromised
popular hosts): 1, kept whole in one split. The test split is FROZEN and
was evaluated exactly once (Phase 3R), guarded in code
(`metadata.json` + refusal with exit code 2).

## 19. StratifiedGroupKFold

5-fold CV on the training split only, `groups=registrable_domain`:
label balance per fold while never letting a domain cross folds. Reported
per model: mean ± std of accuracy, precision, recall, F1, ROC-AUC, PR-AUC
(`model/model_metrics.csv`, stage `cv`).

## 20. Model Comparison (1R dataset, all measured)

Grouped 5-fold CV on train / validation (threshold 0.5):

| Model | CV recall | CV precision | CV F1 | Val F1 | Val FP/FN |
|---|---|---|---|---|---|
| Logistic Regression | 0.9758±0.0102 | 0.9768 | 0.9760 | 0.9646 | 36/35 |
| **Random Forest** | **0.9862±0.0087** | 0.9757 | **0.9804** | **0.9738** | 35/18 |
| XGBoost | 0.9946±0.0042 | 0.8887 | 0.9385 | 0.9314 | 142/5 |
| LightGBM | 0.9970±0.0044 | 0.8811 | 0.9354 | 0.9361 | 129/7 |
| CatBoost | 0.9899±0.0071 | 0.9429 | 0.9657 | 0.9565 | 77/13 |
| Stacking | — | — | — | 0.9738 | 34/19 |

## 21. Ensemble Learning

A custom **group-aware stacking** ensemble (RF/XGBoost/LightGBM/CatBoost
bases; logistic-regression meta-learner trained on out-of-fold
probabilities so it never sees in-sample base scores) was trained and
evaluated. Meta coefficients: RF 5.68, CatBoost 4.61, XGBoost 0.31,
LightGBM **−0.09**. **Not retained**: validation PR-AUC 0.9967 vs Random
Forest 0.9966 (Δ0.0001, noise) with identical F1 — four extra models and
far harder SHAP explainability were not justified by measured benefit.

## 22. Calibration

Sigmoid (Platt) calibration via `CalibratedClassifierCV(
FrozenEstimator(train_fitted_pipeline))` fitted on the **validation**
split — train and validation share zero domains, so the calibrator sees
only held-out data. Raw model scores are *scores*; only calibrated
outputs are reported as probabilities. Measured on the untouched test
set: **Brier 0.0449, 10-bin ECE 0.0309** (raw reference: 0.0473/0.0582).

## 23. Threshold Optimization

Policy (validation only, never test): among thresholds with
recall ≥ 0.97 AND precision ≥ 0.90, maximize recall, then precision,
then lowest threshold; F2 fallback if unsatisfiable.
**Deployed threshold: 0.0706** (no fallback; 51 qualifying thresholds;
validation operating point recall 0.9940, precision 0.9064). The full
precision-recall curve sample is persisted in `model/threshold.json`.

## 24–30. Evaluation Metrics (final one-time test, 1,933 rows)

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | 0.8981 | overall correctness — never the sole criterion |
| **Recall** | **0.9882** | 1009/1021 real malicious URLs detected; misses are the dangerous failure |
| Precision | 0.8451 | of alerts, the fraction that were real (alert-fatigue cost) |
| F1 | 0.9111 | harmonic balance |
| ROC-AUC | 0.9922 | threshold-free ranking quality |
| PR-AUC | 0.9938 | positive-class detection quality |
| Confusion | tn=727 fp=185 fn=12 tp=1009 | see `model/test_errors.csv` (197 rows) |

v1 comparison (archived in `model_v1/`): test F1 0.9985, precision 1.0000,
recall 0.9970, fp=0. The v1 numbers measured an artificially easy task
(benign = homepages only); 1R's lower numbers measure the harder, more
realistic one — 185 false positives on legitimate deep-link URLs became
*measurable at all* only after the dataset fix.

## 31. SHAP Explainability

Exact TreeSHAP via `shap.TreeExplainer` (health-checked at startup;
LightGBM-native `pred_contrib` is the automatic fallback for LightGBM
pipelines — not applicable to the deployed sklearn RF, so the shap
package is the single backend and the active backend is always reported).
For every URL the response carries per-feature name, value, contribution
and direction, split into *increasing* / *reducing* phishing risk, plus
lexical and n-gram evidence totals. **Explanation space (model-dependent,
reported per explanation):** LightGBM → log-odds margin; sklearn
RandomForest → predicted probability (no margin output exists). SHAP
values are evidence, never probabilities; the verdict comes from the
calibrated probability + deployed threshold. A per-URL reconstruction
check (base + Σφ vs model output) is measured and reported.

## 32. IOC Extraction

Full deconstruction (scheme/userinfo/host/subdomain/registrable domain/
TLD/IP/port/path/query/fragment) plus 13 indicator flags (IP host,
unusual port, userinfo trick, percent/double encoding, suspicious
parameter names, long path, many subdomains/params/segments, shortener,
punycode, executable/script extensions). **All flags are indicators, not
verdicts.** Credentials (userinfo passwords and credential-like query
values) are redacted to `[REDACTED]` in every response, log, alert and
database row; `strip_url_credentials()` additionally protects the
optional third-party (VirusTotal) lookup.

## 33. Flask Architecture

Routes: `GET /` (console), `POST /predict`, `POST /predict_batch` (≤50,
per-URL error isolation), `GET /history` (limit/verdict/severity/search/
order, server-side filtering + client-side sorting), plus exempt
`GET /health` and `GET /api/model-info`. SQLite scan history
(`data/scans.db`, WAL, credential-free). Bounded thread-safe LRU cache
(512, keyed on normalized URL; WHOIS/DNS/VT never cached). flask-limiter
(30/min/IP default). CEF-inspired alert lines to `data/alerts.log` per
phishing verdict. Rotating `logs/app.log`. Optional VirusTotal lookup-only
cross-check (env-var key; never ground truth; never submitted for
scanning; locally throttled for the free tier). Vanilla-JS SOC console
with XSS-safe rendering, JSON/CSV export, batch mode.

## 34. Security Considerations

Never visits analyzed URLs; feed files only, page URLs are strings.
Secrets only via environment/git-ignored `.env`; never logged. Credential
redaction end-to-end (caught and fixed one real leak via tests).
Per-IP rate limiting; bounded request/caches/batch sizes; strict network
timeouts; graceful degradation of every enrichment. Frontend escapes all
dynamic values. Flask dev server + in-memory limiter storage are
documented as development-only posture.

## 35. Limitations

- **Dataset bias (measured residuals):** benign is 96.9% https / 0% IP-host;
  malicious 8.7% https / 90.8% IP-host — real-world-plausible but
  source-correlated signals. Malicious class is 98.9% malware-distribution
  URLs (only 57 true phishing URLs — PhishTank/registration unavailable).
- **Validation→test generalization gap:** benign FP rate 3.5% (val) vs
  20.3% (test); precision floor held on validation (0.9064) but not on
  test (0.8451). Reported as measured; no test-based re-tuning was done
  (that would be test-set optimization).
- URL-string-only: a lexically clean phishing page on a legitimately
  registered domain is not detectable without page-content analysis
  (out of scope; demonstrated by v1's false negatives).
- Concept drift; newly registered domains; shortened URLs; compromised
  legitimate domains; adversarial URL mutation; WHOIS/DNS availability
  and volatility; 72/169 sitemap domains were unfetchable (bot filtering).
- Low-probability phishing verdicts (e.g., p≈0.09 with threshold 0.0706)
  carry LOW severity — verdict and severity are intentionally independent.
- Development server, in-memory rate-limit storage, single-node SQLite:
  not enterprise-grade. This is not a replacement for commercial
  security products.

## 36. Future Improvements

Richer benign corpus (crawled/available-URL corpora); phishing-majority
malicious feeds; page-content and certificate features; periodic
retraining and drift monitoring; per-deployment threshold profiles
(precision-first vs recall-first); production WSGI + Redis-backed
limiter; WHOIS caching with TTL; multilingual/IDN handling; SHAP
explanation of the calibrated layer; active-learning loop from analyst
feedback.

---

## Test Suite

156 collected: **149 passed, 7 skipped** (network/JIT-gated: WHOIS, DNS,
enrichment, shap-JIT, live-VT). Covers URL validation/normalization,
entropy, lexical features, IOC extraction + redaction, splitting
(domain purity/determinism), feature builder leakage guards, threshold
rule, ECE, bundle round-trip, RF-SHAP regression, prediction service
(cache, batch), scan store, all Flask routes, VT client (mocked + one
gated live test), frontend assets/markers.

## Reproducibility

Seeds fixed (42); frozen splits with manifest; pinned lockfile
(`requirements.lock.txt`); bundle sha256 in `model/metadata.json`
(`17883fbb…`); one-time test evaluation enforced in code.
EOFcat > README.md << 'EOF'
# Explainable AI-Based Phishing URL Detection & Cybersecurity Analysis Platform

**MCA minor project — a defensive cybersecurity application.**

The platform classifies URLs as phishing/malicious or legitimate using
lexical and statistical URL features, explains every verdict with SHAP,
deconstructs URLs into indicators of compromise (IOCs), optionally
enriches them with WHOIS/DNS and a VirusTotal cross-check, keeps a scan
history with security alert logging, and presents everything in a
SOC-style investigation console.

**Safety posture: the analyzed URL is NEVER visited.** Only URL strings
and public metadata are analyzed. Threat-feed files are downloaded from
their official publishers; URLs inside them are parsed as strings only.

All results quoted below are **measured** on this project's runs; the
artifacts (`model/model_metrics.csv`, `model/metadata.json`,
`data/processed/dataset_stats.json`) contain the full records.

---

## 1. Project Overview

An end-to-end, reproducible pipeline: multi-source dataset construction →
offline feature engineering (26 lexical features + Shannon entropy +
character n-gram TF-IDF) → domain-aware model training and comparison of
five algorithms → probability calibration and recall-first threshold
selection → SHAP explainability → a Flask backend with SQLite history,
rate limiting, caching and alerting → a professional investigation
console. The project also includes a documented **dataset revision
(v1 → 1R)** in which a measured source bias was found and fixed by
re-running the entire pipeline.

## 2. Problem Statement

Phishing is a primary initial-access vector. URL-string analysis is a
valuable first-line triage because it needs no page fetch, works at
scale, and can run before any human or automated browser touches the
link. Two requirements make this hard in practice: (a) the label noise
and source bias inherent in public feeds, and (b) the need for
*explainable* decisions — a SOC analyst must see *why* a URL was flagged.
This project addresses both: domain-aware evaluation to prevent inflated
results, and per-URL SHAP evidence for every verdict.

## 3. Objectives

1. Build a balanced, multi-source dataset with honest source tracking.
2. Engineer URL features locally (no page fetching).
3. Prevent domain-level train/test leakage via grouped splitting.
4. Compare five ML algorithms with reproducible seeds.
5. Deploy only calibrated probabilities with an explicitly optimized threshold.
6. Explain every prediction with SHAP.
7. Extract IOCs and redact credentials defensively.
8. Serve a SOC-style console with history, batch scanning, rate limiting
   and CEF-style alerts, plus an optional VirusTotal cross-check.

## 4. Architecture

```
 Threat feeds (OpenPhish, URLhaus)      Legitimacy sources
   malware-distribution/phishing        (Majestic ranking, sitemap harvest)
            │                                     │
            └────────────┬────────────────────────┘
                         ▼
        dataset/  download → normalize → clean → dedup →
        per-label domain caps → seeded balancing (benign mix)
                         ▼
        data/processed/urls.csv  (10,000 rows, 6,270 domains)
                         ▼
        model/split.py   domain-aware 60/20/20 frozen splits
                         ▼
        features/  26 lexical + entropy + char(3–5)-gram TF-IDF
                   (TF-IDF fitted on TRAINING data only)
                         ▼
        model/train.py  5 models + stacking, StratifiedGroupKFold
                         ▼
        model/finalize.py  sigmoid calibration on validation,
                           recall-first threshold, ONE-TIME test eval
                         ▼
        model/model.joblib  (calibrated model + raw pipeline + threshold)
                         ▼
        services/  prediction · SHAP · IOC · WHOIS/DNS · VirusTotal
                         ▼
        app.py  Flask: GET / · POST /predict · POST /predict_batch ·
                GET /history · GET /health · GET /api/model-info
                         ▼
        SOC investigation console (templates/ + static/)
```

## 5. Folder Structure

```
phishing-url-analyzer/
├── app.py                     Flask application (routes, rate limiting)
├── appconfig.py               config.yaml loader, .env loader, logging setup
├── config.yaml                all configuration (no secrets)
├── requirements.txt           direct dependencies (pinned: requirements.lock.txt)
├── dataset/
│   ├── download_data.py       feed + sitemap acquisition and orchestration
│   ├── sitemap_sources.py     robots.txt/sitemap harvester (Phase 1R)
│   ├── sitemap_harvest_only.py standalone harvest entry point
│   └── build_dataset.py       cleaning, caps, balancing, persistence
├── features/
│   ├── url_utils.py           validation, normalization, registrable domains
│   ├── entropy.py             Shannon entropy
│   ├── feature_extraction.py  26 lexical features
│   ├── feature_builder.py     dense+TF-IDF matrix transformer (sklearn)
│   ├── vectorizer.py          TF-IDF construction (fitted in training only)
│   ├── ioc_extraction.py      URL deconstruction, flags, credential redaction
│   ├── whois_features.py      WHOIS enrichment (graceful)
│   └── dns_features.py        DNS enrichment (graceful)
├── model/
│   ├── split.py               domain-aware frozen splits
│   ├── train.py               Phase 2 training/evaluation orchestration
│   ├── stacking.py            group-aware stacking ensemble
│   ├── finalize.py            Phase 3: calibration, threshold, persistence
│   ├── model.joblib           DEPLOYED bundle (calibrated RF + threshold)
│   ├── threshold.json         threshold policy, PR-curve sample
│   ├── metadata.json          model card: rationale, test results, hashes
│   ├── model_metrics.csv      every measured CV/validation/test row
│   └── test_errors.csv        final-test misclassifications
├── model_v1/                  archived v1 (LightGBM) artifacts for comparison
├── services/
│   ├── prediction_service.py  orchestration + bounded LRU cache
│   ├── shap_service.py        dual-backend exact TreeSHAP
│   ├── scan_store.py          SQLite history
│   ├── alert_logger.py        CEF-inspired phishing alerts
│   └── virustotal_service.py  optional lookup-only VT v3 client
├── templates/index.html       SOC console page
├── static/css, static/js      console styling and logic (vanilla JS)
├── tests/                     156 collected pytest tests
├── data/raw, data/processed   cached feeds, urls.csv, splits, scans.db
└── logs/app.log               rotating application log
```

## 6. Installation (Fedora Workstation)

```bash
sudo dnf install python3       # usually preinstalled
mkdir -p ~/projects/phishing-url-analyzer && cd ~/projects/phishing-url-analyzer
# place the project files here (git clone or copy)
python3 -m venv .venv
```

## 7. Virtual Environment

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt      # or requirements.lock.txt (pinned)
```

Optional VirusTotal: create a git-ignored `.env` containing
`VT_API_KEY=<your key>` (never commit it; the app reads it at startup).

## 8. Execution

```bash
source .venv/bin/activate

# 1. Dataset (network; ~15-30 min incl. sitemap harvest)
python -m dataset.download_data
# offline rebuild from cached raw files:
python -m dataset.build_dataset

# 2. Train and compare all models (grouped CV; ~5 min)
python -m model.train

# 3. Calibrate, select threshold, ONE-TIME test evaluation, persist bundle
python -m model.finalize          # re-runs refuse (one-time discipline)

# 4. Run the console
python app.py                     # http://127.0.0.1:5000

# 5. Tests
python -m pytest tests -q         # 156 collected, 149 passed, 7 skipped
```

## 9. Dataset Sources (revision 1R, actual)

| Source | Role | Raw records | Final selected |
|---|---|---|---|
| OpenPhish community feed | phishing | 300 | 57 |
| URLhaus (abuse.ch) | malware distribution | 14,289 | 4,943 |
| Majestic Million | legitimate (fallback) | 1,000,000 domains | 2,665 constructed homepages |
| Public sitemaps of 169 curated reputable organizations | legitimate deep links | 16,264 URLs (97 domains OK, 72 yielded nothing — reported) | 2,335 |
| PhishTank | phishing | **unavailable** (registration closed) | 0 |
| Tranco | legitimate | **failed** (HTTP 404) | 0 (Majestic fallback) |

Final: **10,000 rows** = 5,000 malicious (label 1: 4,943 `malware_distribution` + 57 `phishing`, threat types kept distinct) + 5,000 benign (label 0). 6,270 unique registrable domains.

**Dataset revision v1 → 1R (the project's key finding):** v1's benign class
was 100% constructed homepages, so SHAP showed "presence of a path" had
become near-decisive malicious evidence (benign path>1 share was 0.000).
Revision 1R mixed real deep-link URLs harvested from public sitemaps into
the benign class (only robots.txt and sitemap files were fetched; page
URLs were never requested). Measured effect: benign path>1 share
**0.000 → 0.459**, benign mean path length **1.0 → 19.3** (malicious: 7.5).

## 10. Dataset Preprocessing (actual numbers)

Normalization → malformed removal (**31**, all bare-public-suffix hosts) →
exact-URL dedup (0) → cross-label conflict resolution (0, malicious wins) →
per-label domain caps (malicious 5 / benign 25; **19,035** dropped, mostly
URLhaus host spam) → seeded class balancing with the benign mix
(2,335 observed + 2,665 constructed, actual observed fraction 0.47) →
seeded shuffle → `urls.csv` + `dataset_stats.json`.

## 11. URL Normalization

One shared function (`features/url_utils.normalize_url`) used at dataset
construction AND inference (no training-serving skew): trim, reject
interior whitespace/control characters, 2048-char cap, http/https only,
lowercase scheme+host, IDNA/punycode encoding, hostname shape validation,
userinfo/port/path/query/fragment preserved (userinfo is deliberately kept
for the model — `@` obfuscation is signal).

## 12. Lexical Features (26, all offline)

Lengths (URL/host/path/query/fragment), counts (dots, hyphens,
underscores, digits, special chars, slashes), subdomain count, query
parameter count, IP-host flag, `@` flag, HTTPS flag, explicit/unusual
port, percent-encoding count, distinct suspicious-keyword hits,
URL-shortener flag, path segments, digit ratio, special-char ratio,
`url_entropy`, `hostname_entropy`. None is proof of phishing; the model
weighs them jointly.

## 13. Shannon Entropy

`H(X) = -Σ p(x) log2 p(x)` over the character distribution of the URL and
hostname. Higher entropy can indicate randomized/DGA-like strings.
**Limitation: high entropy is NOT proof of phishing** (CDNs, IDN domains).

## 14. WHOIS Features (scan-time enrichment, NOT model input)

Registrar, creation/expiration dates, domain age, days to expiry.
Deliberately excluded from training: bulk WHOIS is impractical (per-TLD
rate limits) and non-reproducible (values change). **WHOIS unavailability
is never a phishing signal** — privacy protection, rate limits, and
unsupported registries all produce `Unavailable`.

## 15. DNS Features (scan-time enrichment, NOT model input)

A-record count/TTL (host), MX presence, NS count and diversity
(registrable domain). Strict timeouts; failures are `Unavailable`, never
verdicts. An authoritative NoAnswer *is* information (recorded as 0/False).

## 16. Character N-gram TF-IDF

Char (3,5)-gram TF-IDF over the URL string captures typosquatting
(`paypa1`), brand impersonation, embedded keywords and obfuscation.
**Fitted only on training data** — inside the pipeline, per CV fold and
per final fit; never on the full dataset before splitting. 5,000 features
max, `min_df=3`, persisted with the model.

## 17. Domain-Level Data Leakage

Random URL-level splitting would place `evil-example.com/login` in train
and `evil-example.com/account` in test — same domain, inflated scores.
Prevention: every registrable domain is assigned to exactly one split;
hosts without a registrable domain are dropped at cleaning time (they
cannot be grouped).

## 18. Group-Aware Splitting (actual)

By registrable domain, 60/20/20, seed 42: train 6,056 rows / 3,762
domains, validation 2,011 / 1,254, test 1,933 / 1,254. **Domain overlap:
0/0/0** (verified programmatically). Dual-label domains (compromised
popular hosts): 1, kept whole in one split. The test split is FROZEN and
was evaluated exactly once (Phase 3R), guarded in code
(`metadata.json` + refusal with exit code 2).

## 19. StratifiedGroupKFold

5-fold CV on the training split only, `groups=registrable_domain`:
label balance per fold while never letting a domain cross folds. Reported
per model: mean ± std of accuracy, precision, recall, F1, ROC-AUC, PR-AUC
(`model/model_metrics.csv`, stage `cv`).

## 20. Model Comparison (1R dataset, all measured)

Grouped 5-fold CV on train / validation (threshold 0.5):

| Model | CV recall | CV precision | CV F1 | Val F1 | Val FP/FN |
|---|---|---|---|---|---|
| Logistic Regression | 0.9758±0.0102 | 0.9768 | 0.9760 | 0.9646 | 36/35 |
| **Random Forest** | **0.9862±0.0087** | 0.9757 | **0.9804** | **0.9738** | 35/18 |
| XGBoost | 0.9946±0.0042 | 0.8887 | 0.9385 | 0.9314 | 142/5 |
| LightGBM | 0.9970±0.0044 | 0.8811 | 0.9354 | 0.9361 | 129/7 |
| CatBoost | 0.9899±0.0071 | 0.9429 | 0.9657 | 0.9565 | 77/13 |
| Stacking | — | — | — | 0.9738 | 34/19 |

## 21. Ensemble Learning

A custom **group-aware stacking** ensemble (RF/XGBoost/LightGBM/CatBoost
bases; logistic-regression meta-learner trained on out-of-fold
probabilities so it never sees in-sample base scores) was trained and
evaluated. Meta coefficients: RF 5.68, CatBoost 4.61, XGBoost 0.31,
LightGBM **−0.09**. **Not retained**: validation PR-AUC 0.9967 vs Random
Forest 0.9966 (Δ0.0001, noise) with identical F1 — four extra models and
far harder SHAP explainability were not justified by measured benefit.

## 22. Calibration

Sigmoid (Platt) calibration via `CalibratedClassifierCV(
FrozenEstimator(train_fitted_pipeline))` fitted on the **validation**
split — train and validation share zero domains, so the calibrator sees
only held-out data. Raw model scores are *scores*; only calibrated
outputs are reported as probabilities. Measured on the untouched test
set: **Brier 0.0449, 10-bin ECE 0.0309** (raw reference: 0.0473/0.0582).

## 23. Threshold Optimization

Policy (validation only, never test): among thresholds with
recall ≥ 0.97 AND precision ≥ 0.90, maximize recall, then precision,
then lowest threshold; F2 fallback if unsatisfiable.
**Deployed threshold: 0.0706** (no fallback; 51 qualifying thresholds;
validation operating point recall 0.9940, precision 0.9064). The full
precision-recall curve sample is persisted in `model/threshold.json`.

## 24–30. Evaluation Metrics (final one-time test, 1,933 rows)

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | 0.8981 | overall correctness — never the sole criterion |
| **Recall** | **0.9882** | 1009/1021 real malicious URLs detected; misses are the dangerous failure |
| Precision | 0.8451 | of alerts, the fraction that were real (alert-fatigue cost) |
| F1 | 0.9111 | harmonic balance |
| ROC-AUC | 0.9922 | threshold-free ranking quality |
| PR-AUC | 0.9938 | positive-class detection quality |
| Confusion | tn=727 fp=185 fn=12 tp=1009 | see `model/test_errors.csv` (197 rows) |

v1 comparison (archived in `model_v1/`): test F1 0.9985, precision 1.0000,
recall 0.9970, fp=0. The v1 numbers measured an artificially easy task
(benign = homepages only); 1R's lower numbers measure the harder, more
realistic one — 185 false positives on legitimate deep-link URLs became
*measurable at all* only after the dataset fix.

## 31. SHAP Explainability

Exact TreeSHAP via `shap.TreeExplainer` (health-checked at startup;
LightGBM-native `pred_contrib` is the automatic fallback for LightGBM
pipelines — not applicable to the deployed sklearn RF, so the shap
package is the single backend and the active backend is always reported).
For every URL the response carries per-feature name, value, contribution
and direction, split into *increasing* / *reducing* phishing risk, plus
lexical and n-gram evidence totals. **Explanation space (model-dependent,
reported per explanation):** LightGBM → log-odds margin; sklearn
RandomForest → predicted probability (no margin output exists). SHAP
values are evidence, never probabilities; the verdict comes from the
calibrated probability + deployed threshold. A per-URL reconstruction
check (base + Σφ vs model output) is measured and reported.

## 32. IOC Extraction

Full deconstruction (scheme/userinfo/host/subdomain/registrable domain/
TLD/IP/port/path/query/fragment) plus 13 indicator flags (IP host,
unusual port, userinfo trick, percent/double encoding, suspicious
parameter names, long path, many subdomains/params/segments, shortener,
punycode, executable/script extensions). **All flags are indicators, not
verdicts.** Credentials (userinfo passwords and credential-like query
values) are redacted to `[REDACTED]` in every response, log, alert and
database row; `strip_url_credentials()` additionally protects the
optional third-party (VirusTotal) lookup.

## 33. Flask Architecture

Routes: `GET /` (console), `POST /predict`, `POST /predict_batch` (≤50,
per-URL error isolation), `GET /history` (limit/verdict/severity/search/
order, server-side filtering + client-side sorting), plus exempt
`GET /health` and `GET /api/model-info`. SQLite scan history
(`data/scans.db`, WAL, credential-free). Bounded thread-safe LRU cache
(512, keyed on normalized URL; WHOIS/DNS/VT never cached). flask-limiter
(30/min/IP default). CEF-inspired alert lines to `data/alerts.log` per
phishing verdict. Rotating `logs/app.log`. Optional VirusTotal lookup-only
cross-check (env-var key; never ground truth; never submitted for
scanning; locally throttled for the free tier). Vanilla-JS SOC console
with XSS-safe rendering, JSON/CSV export, batch mode.

## 34. Security Considerations

Never visits analyzed URLs; feed files only, page URLs are strings.
Secrets only via environment/git-ignored `.env`; never logged. Credential
redaction end-to-end (caught and fixed one real leak via tests).
Per-IP rate limiting; bounded request/caches/batch sizes; strict network
timeouts; graceful degradation of every enrichment. Frontend escapes all
dynamic values. Flask dev server + in-memory limiter storage are
documented as development-only posture.

## 35. Limitations

- **Dataset bias (measured residuals):** benign is 96.9% https / 0% IP-host;
  malicious 8.7% https / 90.8% IP-host — real-world-plausible but
  source-correlated signals. Malicious class is 98.9% malware-distribution
  URLs (only 57 true phishing URLs — PhishTank/registration unavailable).
- **Validation→test generalization gap:** benign FP rate 3.5% (val) vs
  20.3% (test); precision floor held on validation (0.9064) but not on
  test (0.8451). Reported as measured; no test-based re-tuning was done
  (that would be test-set optimization).
- URL-string-only: a lexically clean phishing page on a legitimately
  registered domain is not detectable without page-content analysis
  (out of scope; demonstrated by v1's false negatives).
- Concept drift; newly registered domains; shortened URLs; compromised
  legitimate domains; adversarial URL mutation; WHOIS/DNS availability
  and volatility; 72/169 sitemap domains were unfetchable (bot filtering).
- Low-probability phishing verdicts (e.g., p≈0.09 with threshold 0.0706)
  carry LOW severity — verdict and severity are intentionally independent.
- Development server, in-memory rate-limit storage, single-node SQLite:
  not enterprise-grade. This is not a replacement for commercial
  security products.

## 36. Future Improvements

Richer benign corpus (crawled/available-URL corpora); phishing-majority
malicious feeds; page-content and certificate features; periodic
retraining and drift monitoring; per-deployment threshold profiles
(precision-first vs recall-first); production WSGI + Redis-backed
limiter; WHOIS caching with TTL; multilingual/IDN handling; SHAP
explanation of the calibrated layer; active-learning loop from analyst
feedback.

---

## Test Suite

156 collected: **149 passed, 7 skipped** (network/JIT-gated: WHOIS, DNS,
enrichment, shap-JIT, live-VT). Covers URL validation/normalization,
entropy, lexical features, IOC extraction + redaction, splitting
(domain purity/determinism), feature builder leakage guards, threshold
rule, ECE, bundle round-trip, RF-SHAP regression, prediction service
(cache, batch), scan store, all Flask routes, VT client (mocked + one
gated live test), frontend assets/markers.

## Reproducibility

Seeds fixed (42); frozen splits with manifest; pinned lockfile
(`requirements.lock.txt`); bundle sha256 in `model/metadata.json`
(`17883fbb…`); one-time test evaluation enforced in code.

---

## Post-1.0 Increment (v1.1)

- **OpenPhish daily accumulation** (`python -m dataset.accumulate_feeds`, cron-ready):
  dated snapshots in `data/raw/openphish_YYYYMMDD.txt`, merged by
  `dataset.build_dataset` (exact duplicates removed at cleaning). Grows the
  true-phishing share for the next scheduled retrain. No retraining happened
  in v1.1 — the deployed model is unchanged.
- **Analysis tooling** (`python -m model.analysis`): feature-group ablation
  (combined / lexical-only / ngram-only; validation split only, frozen test
  untouched) → `model/ablation_results.csv`; final-test error categorisation
  → `model/error_analysis.json`.
- **UI v1.1**: traffic-light risk colours (green/yellow/red across the score
  number, score bar, severity badges, IOC warnings, deconstruction highlights),
  SHAP legend and explanation-space label.

## AI-assistant / collaborator handoff

Read `PROJECT_CONTEXT.md` first - it is the authoritative state + facts ledger + working rules for anyone (human or AI) continuing this project.
