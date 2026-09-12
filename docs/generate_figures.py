"""Generate all research-paper figures from MEASURED project artifacts.

Every figure is derived from the project's recorded results - nothing is
fabricated. Provenance is printed per figure; cite the source artifact in
each caption. Output: docs/figures/*.png at 300 dpi.

Run: python -m docs.generate_figures
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MAL, BEN, BLUE, GRAY, YELLOW = "#c44e52", "#55a868", "#4c72b0", "#6d6d6d", "#e3b341"
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
})

# ---- ledger constants: values MEASURED in earlier verified runs -------
# v1 dataset (superseded; archived measurements): benign path>1 share 0.000,
# benign mean path length 1.0, malicious 0.994 / 7.5.
V1_BEN_SHARE, V1_BEN_LEN = 0.000, 1.0
V1_MAL_SHARE, V1_MAL_LEN = 0.994, 7.5
# 1R TRAINING composition (ledger; the deployed model trained on these):
TRAIN_COMP = {"urlhaus": 4943, "openphish": 57, "majestic": 2665, "sitemap": 2335}


def _load():
    d = {}
    d["stats"] = json.loads((ROOT / "data/processed/dataset_stats.json").read_text())
    d["metrics"] = pd.read_csv(ROOT / "model/model_metrics.csv")
    d["thr"] = json.loads((ROOT / "model/threshold.json").read_text())
    d["meta"] = json.loads((ROOT / "model/metadata.json").read_text())
    d["abl"] = pd.read_csv(ROOT / "model/ablation_results.csv")
    d["errs"] = pd.read_csv(ROOT / "model/test_errors.csv")
    d["urls"] = pd.read_csv(ROOT / "data/processed/urls.csv")
    return d


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(-0.22, 1.02); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                    fc=color, ec="black", lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=7.2, color="white", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))

    box(0.02, 0.84, 0.44, 0.14, "Threat feeds\nOpenPhish · URLhaus", MAL)
    box(0.54, 0.84, 0.44, 0.14, "Legitimacy sources\nMajestic · sitemap harvest", BEN)
    steps = [
        "Cleaning · normalization · per-label domain caps\n(10,000 URLs · 6,270 registrable domains)",
        "Domain-aware split 60/20/20 (domain overlap 0/0/0)\nfrozen train / validation / test",
        "Features: 26 lexical + Shannon entropy\n+ char (3–5)-gram TF-IDF (train-only fit)",
        "Model comparison: 5 algorithms + group-aware stacking\nRandomForest selected",
        "Sigmoid calibration · threshold 0.0706\none-time held-out test evaluation",
        "DeceptIQ console: Flask · SHAP evidence · IOC\nhistory · alerts · VirusTotal cross-check",
    ]
    ys = [0.66, 0.53, 0.40, 0.27, 0.14, 0.00]
    for y, s in zip(ys, steps):
        box(0.16, y, 0.68, 0.12, s, BLUE)
    arrow(0.24, 0.84, 0.38, 0.78); arrow(0.76, 0.84, 0.62, 0.78)
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(0.50, a, 0.50, b + 0.12)
    fig.savefig(OUT / "fig1_architecture.png"); plt.close(fig)
    print("fig1_architecture.png  <- pipeline schematic (project design)")


def fig_dataset_composition():
    mal = [("URLhaus", TRAIN_COMP["urlhaus"], MAL), ("OpenPhish", TRAIN_COMP["openphish"], "#a63c40")]
    ben = [("Majestic homepages", TRAIN_COMP["majestic"], BEN), ("Sitemap deep links", TRAIN_COMP["sitemap"], "#3f7f52")]
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    left = 0
    for name, v, c in mal:
        ax.barh(1, v, left=left, color=c, edgecolor="white", height=0.5)
        ax.text(left + v / 2, 1, f"{name}\n{v:,}", ha="center", va="center",
                fontsize=7.5, color="white")
        left += v
    left = 0
    for name, v, c in ben:
        ax.barh(0, v, left=left, color=c, edgecolor="white", height=0.5)
        ax.text(left + v / 2, 0, f"{name}\n{v:,}", ha="center", va="center",
                fontsize=7.5, color="white")
        left += v
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign (0)", "Malicious (1)"])
    ax.set_xlabel("URLs (1R training dataset)"); ax.set_xlim(0, 5200)
    fig.savefig(OUT / "fig2_dataset_composition.png"); plt.close(fig)
    cur = d["urls"].source.value_counts().to_dict()
    print(f"fig2_dataset_composition.png  <- 1R TRAINING ledger {TRAIN_COMP} "
          f"(current on-disk staged dataset: {cur})")


def fig_bias_revision():
    u = d["urls"]
    paths = u.url.map(lambda s: urlsplit(str(s)).path)
    ben, mals = u.label == 0, u.label == 1
    r_ben = float((paths[ben].str.len() > 1).mean())
    r_mal = float((paths[mals].str.len() > 1).mean())
    l_ben = float(paths[ben].str.len().mean())
    l_mal = float(paths[mals].str.len().mean())
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    x = np.arange(2); w = 0.35
    axes[0].bar(x - w / 2, [V1_BEN_SHARE, V1_MAL_SHARE], w, label="v1", color=GRAY)
    axes[0].bar(x + w / 2, [r_ben, r_mal], w, label="1R", color=BLUE)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["Benign", "Malicious"])
    axes[0].set_ylabel("share of URLs with path length > 1")
    axes[0].set_title("(a) Path-presence share"); axes[0].legend(); axes[0].set_ylim(0, 1.1)
    axes[1].bar(x - w / 2, [V1_BEN_LEN, V1_MAL_LEN], w, label="v1", color=GRAY)
    axes[1].bar(x + w / 2, [l_ben, l_mal], w, label="1R", color=BLUE)
    axes[1].set_xticks(x); axes[1].set_xticklabels(["Benign", "Malicious"])
    axes[1].set_ylabel("mean path length (chars)")
    axes[1].set_title("(b) Mean path length"); axes[1].legend()
    fig.savefig(OUT / "fig3_bias_revision.png"); plt.close(fig)
    print(f"fig3_bias_revision.png  <- v1 ledger values vs 1R computed from urls.csv "
          f"(benign {V1_BEN_SHARE:.3f}->{r_ben:.3f}, mean len {V1_BEN_LEN:.1f}->{l_ben:.1f})")


def fig_model_comparison():
    cv = d["metrics"][d["metrics"].stage == "cv"].copy()
    cv = cv.set_index("model").loc[["logreg", "random_forest", "xgboost",
                                    "lightgbm", "catboost"]]
    x = np.arange(len(cv)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ax.bar(x - w, cv.recall, w, yerr=cv.recall_std, capsize=2.5, label="Recall", color=MAL)
    ax.bar(x, cv.precision, w, yerr=cv.precision_std, capsize=2.5, label="Precision", color=BLUE)
    ax.bar(x + w, cv.f1, w, label="F1", color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(["LogReg", "Random\nForest", "XGBoost",
                                          "LightGBM", "CatBoost"], fontsize=8)
    ax.set_ylim(0.85, 1.01); ax.set_ylabel("score (grouped 5-fold CV)")
    ax.legend(ncol=3, loc="lower right")
    fig.savefig(OUT / "fig4_model_comparison.png"); plt.close(fig)
    print("fig4_model_comparison.png  <- model/model_metrics.csv (stage=cv, mean±std)")


def fig_pr_curve():
    pts = d["thr"]["pr_curve_sample"]
    r = [p["recall"] for p in pts]; p_ = [p["precision"] for p in pts]
    op = d["thr"]["operating_point_validation"]
    t = d["thr"]["threshold"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(r, p_, "-", color=BLUE, lw=1.4, label="validation PR curve")
    ax.plot(op["recall"], op["precision"], "o", color=MAL, ms=6,
            label=f"deployed threshold {t:.4f}")
    ax.annotate(f"recall {op['recall']:.3f}\nprecision {op['precision']:.3f}",
                (op["recall"], op["precision"]), textcoords="offset points",
                xytext=(-95, -12), fontsize=7.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0.88, 1.005)
    ax.set_ylim(0.88, 1.005); ax.legend(loc="lower left")
    fig.savefig(OUT / "fig5_pr_curve.png"); plt.close(fig)
    print("fig5_pr_curve.png  <- model/threshold.json (pr_curve_sample + operating point)")


def fig_confusion():
    tr = d["meta"]["test_results"]["calibrated_at_threshold"]
    cm = np.array([[tr["tn"], tr["fp"]], [tr["fn"], tr["tp"]]])
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.25)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            tot = cm[i].sum()
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}\n({cm[i, j] / tot:.1%})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted safe", "Predicted malicious"])
    ax.set_yticklabels(["Actual\nbenign", "Actual\nmalicious"])
    ax.grid(False); ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    fig.savefig(OUT / "fig6_confusion.png"); plt.close(fig)
    print("fig6_confusion.png  <- model/metadata.json (one-time test, threshold 0.0706)")


def fig_ablation():
    a = d["abl"].set_index("feature_mode").loc[["combined", "lexical", "ngram"]]
    x = np.arange(len(a)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.bar(x - w, a.precision, w, label="Precision", color=BLUE)
    ax.bar(x, a.recall, w, label="Recall", color=MAL)
    ax.bar(x + w, a.f1, w, label="F1", color=GRAY)
    for xi, n in zip(x, a.n_features):
        ax.text(xi + w, 0.86, f"n={n:,}", ha="center", fontsize=7, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({n:,} features)" for m, n in
                        zip(a.index, a.n_features)], fontsize=8)
    ax.set_ylim(0.85, 1.005); ax.set_ylabel("score (validation)")
    ax.legend(ncol=3, loc="lower right")
    fig.savefig(OUT / "fig7_ablation.png"); plt.close(fig)
    print("fig7_ablation.png  <- model/ablation_results.csv")


def fig_errors():
    e = d["errs"]
    fp = e[e.label == 0].calibrated_probability
    fn = e[e.label == 1].calibrated_probability
    t = d["thr"]["threshold"]
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(fp, bins=bins, color=MAL, alpha=0.85, label=f"False positives (n={len(fp)})")
    ax.hist(fn, bins=bins, color=BEN, alpha=0.85, label=f"False negatives (n={len(fn)})")
    ax.axvline(t, color="black", ls="--", lw=1, label=f"threshold {t:.4f}")
    ax.set_xlabel("calibrated probability"); ax.set_ylabel("URLs")
    ax.legend()
    fig.savefig(OUT / "fig8_errors.png"); plt.close(fig)
    print("fig8_errors.png  <- model/test_errors.csv (197 misclassifications)")


def fig_shap_case():
    from services.shap_service import ShapExplainer
    bundle = joblib.load(ROOT / "model/model.joblib")
    ex = ShapExplainer(bundle["raw_pipeline"], backend="shap")
    exp = ex.explain(["http://192.0.2.10:8080/login.php"])[0]
    items = exp.increasing[:8] + exp.decreasing[:4]
    names = [f"{c.name} = {c.display_value}" for c in items]
    vals = [c.shap for c in items]
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    ax.barh(range(len(vals)), vals,
            color=[MAL if v > 0 else BEN for v in vals])
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP contribution (probability space)")
    p = float(bundle["calibrated_model"].predict_proba(
        ["http://192.0.2.10:8080/login.php"])[0, 1])
    ax.set_title(f"p(malicious) = {p:.4f} — threshold {bundle['threshold']:.4f}")
    fig.savefig(OUT / "fig9_shap_case.png"); plt.close(fig)
    print("fig9_shap_case.png  <- deployed bundle, live TreeSHAP computation")


if __name__ == "__main__":
    d = _load()
    for fn in (fig_architecture, fig_dataset_composition, fig_bias_revision,
               fig_model_comparison, fig_pr_curve, fig_confusion,
               fig_ablation, fig_errors, fig_shap_case):
        try:
            fn()
        except Exception as exc:
            print(f"FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\nFIGURES -> {OUT}")


def _render_table(path, title, headers, rows, col_widths, note=None):
    import matplotlib.table as mtable
    n_rows = len(rows) + 1
    fig, ax = plt.subplots(figsize=(6.3, 0.32 * n_rows + 0.35))
    ax.axis("off")
    cell_text = [[str(c) for c in r] for r in rows]
    tbl = ax.table(cellText=cell_text, colLabels=headers,
                   colWidths=col_widths, cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("black"); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#e8e8e8"); cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f6f6f6")
    tbl.scale(1, 1.5)
    ax.set_title(title, fontsize=9, pad=18)
    if note:
        fig.text(0.02, -0.06 / n_rows, note, fontsize=6.5, style="italic",
                 color="dimgray", wrap=True)
    fig.savefig(OUT / path, bbox_inches="tight"); plt.close(fig)
    print(f"{path}  <- generated table: {title}")


def tab_dataset():
    _render_table(
        "tab1_sources.png", "Table 1: Dataset sources (1R training set)",
        ["Source", "Role", "Raw records", "Selected"],
        [["URLhaus (abuse.ch)", "Malware distribution", "14,289", "4,943"],
         ["OpenPhish feed", "Phishing", "300", "57"],
         ["Majestic Million", "Legitimate homepages", "1,000,000 domains", "2,665"],
         ["Public sitemaps (169 orgs)", "Legitimate deep links", "16,264", "2,335"],
         ["PhishTank", "Phishing", "unavailable (closed registration)", "0"],
         ["Tranco", "Legitimate", "failed (HTTP 404)", "0 (Majestic fallback)"],
         ["TOTAL", "—", "—", "10,000 rows / 6,270 domains"]],
        [0.26, 0.24, 0.26, 0.24],
        note="Malicious 5,000 (label 1) + benign 5,000 (label 0); cleaning removed 31 malformed; "
             "19,035 dropped by per-label domain caps (malicious 5 / benign 25).",
    )


def tab_models():
    cv = d["metrics"][d["metrics"].stage == "cv"].set_index("model")
    va = d["metrics"][d["metrics"].stage == "validation"].set_index("model")
    order = ["logreg", "random_forest", "xgboost", "lightgbm", "catboost", "stacking"]
    disp = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
            "xgboost": "XGBoost", "lightgbm": "LightGBM",
            "catboost": "CatBoost", "stacking": "Stacking (not retained)"}
    rows = []
    for m in order:
        if m in cv.index:
            r, p_ = cv.loc[m, "recall"], cv.loc[m, "precision"]
            rs, ps = cv.loc[m, "recall_std"], cv.loc[m, "precision_std"]
            rows.append([disp[m], f"{r:.4f}±{rs:.4f}", f"{p_:.4f}±{ps:.4f}",
                         f"{cv.loc[m, 'f1']:.4f}", f"{va.loc[m, 'f1']:.4f}",
                         f"{int(va.loc[m, 'fp'])}/{int(va.loc[m, 'fn'])}"])
        else:  # stacking has no CV row (documented: no outer CV)
            rows.append([disp[m], "—", "—", "—", f"{va.loc[m, 'f1']:.4f}",
                         f"{int(va.loc[m, 'fp'])}/{int(va.loc[m, 'fn'])}"])
    _render_table(
        "tab2_models.png", "Table 2: Model comparison (grouped 5-fold CV on train / validation @0.5)",
        ["Model", "CV Recall", "CV Precision", "CV F1", "Val F1", "Val FP/FN"],
        rows, [0.24, 0.17, 0.17, 0.13, 0.13, 0.16],
        note="Random Forest selected (best individual CV F1, most balanced validation confusion). "
             "Stacking evaluated and not retained (validation PR-AUC delta 0.0001 vs RF).",
    )


def tab_test():
    tr = d["meta"]["test_results"]["calibrated_at_threshold"]
    n_err = d["meta"]["test_results"]["n_misclassified"]
    t = d["thr"]["threshold"]
    _render_table(
        "tab3_test.png",
        f"Table 3: Final one-time test evaluation ({d['meta']['test_results']['n_test_rows']:,} rows, threshold {t:.4f})",
        ["Metric", "Value", "Metric", "Value"],
        [["Accuracy", f"{tr['accuracy']:.4f}", "Recall", f"{tr['recall']:.4f}"],
         ["Precision", f"{tr['precision']:.4f}", "F1", f"{tr['f1']:.4f}"],
         ["ROC-AUC", f"{tr['roc_auc']:.4f}", "PR-AUC", f"{tr['pr_auc']:.4f}"],
         ["Brier score", f"{tr['brier']:.4f}", "ECE (10-bin)", f"{tr['ece_10bin']:.4f}"],
         ["TN", f"{tr['tn']:,}", "FP", f"{tr['fp']:,}"],
         ["FN", f"{tr['fn']:,}", "TP", f"{tr['tp']:,}"],
         ["Errors (total)", f"{n_err}", "Threshold", f"{t:.4f}"]],
        [0.27, 0.23, 0.27, 0.23],
        note="One-time evaluation on the frozen held-out test split; calibrated model at the "
             "deployed threshold; no test-based tuning was performed.",
    )


if __name__ == "__main__":
    if "d" not in dir() or d is None:
        d = _load()
    for fn in (tab_dataset, tab_models, tab_test):
        try:
            fn()
        except Exception as exc:
            print(f"FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"TABLES -> {OUT}")

# =====================================================================
# Appendix B design diagrams (D2-D6) - drawn like all other figures.
# D1 = fig1_architecture.png (generated above).
# =====================================================================

from matplotlib.patches import Ellipse  # noqa: E402


def _dfig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    return fig, ax


def _dbox(ax, x, y, w, h, text, fc="white", ec="black", fs=6.5, tc="black",
          lw=0.7, ls="-", bold=False, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, linespacing=1.25, zorder=z + 1,
            weight="bold" if bold else "normal")


def _dline(ax, x1, y1, x2, y2, lw=0.55, ls="-", color="black"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls, zorder=1)


def _darrow(ax, x1, y1, x2, y2, lw=0.8, ls="-", color="black", rad=0.0,
            head="-|>", z=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=z,
                arrowprops=dict(arrowstyle=head, color=color, lw=lw,
                                linestyle=ls, shrinkA=1.5, shrinkB=1.5,
                                connectionstyle=f"arc3,rad={rad}"))


def _dtext(ax, x, y, text, fs=5.4, style="normal", color="black", ha="center",
           weight="normal", z=3):
    ax.text(x, y, text, fontsize=fs, style=style, color=color, ha=ha,
            va="center", zorder=z, weight=weight, linespacing=1.2)


def _dell(ax, cx, cy, w, h, text, fc="white", fs=6.0, lw=0.7, double=False):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec="black", lw=lw, zorder=2))
    if double:  # multi-valued attribute (ER notation)
        ax.add_patch(Ellipse((cx, cy), w * 0.84, h * 0.68, fc="none",
                             ec="black", lw=lw * 0.6, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3,
            linespacing=1.15)


def _dactor(ax, x, y, label):
    ax.add_patch(Ellipse((x, y + 0.052), 0.020, 0.026, fill=False,
                         ec="black", lw=0.7, zorder=2))
    ax.plot([x, x], [y + 0.039, y + 0.012], color="black", lw=0.7, zorder=2)
    ax.plot([x - 0.022, x + 0.022], [y + 0.030, y + 0.030],
            color="black", lw=0.7, zorder=2)
    ax.plot([x, x - 0.016], [y + 0.012, y - 0.012], color="black", lw=0.7, zorder=2)
    ax.plot([x, x + 0.016], [y + 0.012, y - 0.012], color="black", lw=0.7, zorder=2)
    ax.text(x, y - 0.024, label, ha="center", va="top", fontsize=6.5,
            zorder=3, linespacing=1.15)


def fig_usecase():
    fig, ax = _dfig(6.3, 4.4)
    ax.add_patch(FancyBboxPatch((0.23, 0.035), 0.54, 0.94,
                                boxstyle="square,pad=0", fc="#f4f6f8",
                                ec="black", lw=1.0, zorder=1))
    _dtext(ax, 0.50, 0.955, "DeceptIQ", fs=8, weight="bold")
    left = [(0.36, 0.845, "Analyze URL\n(predict)", 0.115),
            (0.36, 0.665, "View SHAP\nevidence", 0.115),
            (0.36, 0.485, "Extract IOCs", 0.115),
            (0.36, 0.300, "WHOIS/DNS\nenrichment\n«opt-in»", 0.135),
            (0.36, 0.115, "VirusTotal\ncross-check\n«opt-in»", 0.135)]
    right = [(0.64, 0.845, "Batch scan\n(≤ 50 URLs)", 0.115),
             (0.64, 0.665, "Search /\nfilter history", 0.115),
             (0.64, 0.485, "Export report\n(JSON / CSV)", 0.115),
             (0.64, 0.300, "Configure\nsystem", 0.115),
             (0.64, 0.115, "Retrain pipeline\n(offline)", 0.115)]
    for cx, cy, _, _ in left + right[:3]:        # analyst associations
        _dline(ax, 0.115, 0.53, cx, cy)
    for cx, cy, _, _ in [right[0], right[1], left[0]]:   # API consumer
        _dline(ax, 0.885, 0.74, cx, cy)
    for cx, cy, _, _ in right[3:]:               # administrator
        _dline(ax, 0.885, 0.26, cx, cy)
    for cx, cy, t, h in left + right:
        _dell(ax, cx, cy, 0.185, h, t, fs=5.5)
    # «include» relations from Analyze URL
    _darrow(ax, 0.36, 0.7875, 0.36, 0.7225, lw=0.6, ls=(0, (3, 2)))
    _dtext(ax, 0.435, 0.755, "«include»", fs=4.8, ha="left")
    _darrow(ax, 0.36, 0.6075, 0.36, 0.5425, lw=0.6, ls=(0, (3, 2)))
    _dtext(ax, 0.435, 0.575, "«include»", fs=4.8, ha="left")
    _dactor(ax, 0.095, 0.53, "Security\nAnalyst")
    _dactor(ax, 0.895, 0.74, "API\nConsumer")
    _dactor(ax, 0.895, 0.26, "Administrator")
    _dtext(ax, 0.50, 0.012,
           "All analysis is local - the analyzed URL is never visited; "
           "enrichment and VirusTotal are opt-in.",
           fs=5.0, style="italic", color="dimgray")
    fig.savefig(OUT / "figD2_usecase.png"); plt.close(fig)
    print("figD2_usecase.png  <- use-case diagram (SRS 2.5 / 2.4 actors and FRs)")


def fig_dfd0():
    fig, ax = _dfig(6.3, 4.3)

    def store(x, y, w, h, sid, name):
        ax.plot([x, x + w], [y + h, y + h], color="black", lw=0.8, zorder=3)
        ax.plot([x, x + w], [y, y], color="black", lw=0.8, zorder=3)
        ax.plot([x, x], [y, y + h], color="black", lw=0.8, zorder=3)
        ax.plot([x + 0.035, x + 0.035], [y, y + h], color="black", lw=0.8, zorder=3)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                    fc="white", ec="none", zorder=2))
        _dtext(ax, x + 0.017, y + h / 2, sid, fs=5.5, weight="bold")
        _dtext(ax, x + 0.05 + (w - 0.05) / 2, y + h / 2, name, fs=5.6)

    def proc(cx, cy, num, name):
        _dell(ax, cx, cy, 0.135, 0.125, "", fc="#dce6f2")
        _dtext(ax, cx, cy + 0.028, str(num), fs=6.2, weight="bold")
        _dtext(ax, cx, cy - 0.020, name, fs=4.9)

    # flows first (pass behind shapes)
    _darrow(ax, 0.13, 0.485, 0.1775, 0.485)
    _darrow(ax, 0.3125, 0.485, 0.3625, 0.485)
    _darrow(ax, 0.4975, 0.485, 0.5475, 0.485)
    _darrow(ax, 0.6825, 0.485, 0.7925, 0.485)
    _darrow(ax, 0.58, 0.435, 0.36, 0.20)
    _darrow(ax, 0.635, 0.435, 0.53, 0.20)
    _darrow(ax, 0.5725, 0.135, 0.6275, 0.135)
    _darrow(ax, 0.735, 0.175, 0.845, 0.43, ls=(0, (3, 2)))
    _darrow(ax, 0.75, 0.16, 0.88, 0.845, head="<|-|>", rad=-0.08)
    _darrow(ax, 0.155, 0.9325, 0.1775, 0.9325)
    _darrow(ax, 0.3125, 0.90, 0.375, 0.8275)
    _darrow(ax, 0.60, 0.79, 0.615, 0.5475)
    _darrow(ax, 0.86, 0.4225, 0.855, 0.325)
    _darrow(ax, 0.825, 0.425, 0.80, 0.072)
    _darrow(ax, 0.795, 0.445, 0.128, 0.47, rad=0.22)

    _dbox(ax, 0.015, 0.42, 0.115, 0.13, "ANALYST", fc="#e6e6e6", fs=6.5, bold=True)
    _dbox(ax, 0.845, 0.845, 0.145, 0.11, "VirusTotal\nWHOIS / DNS\n(external)",
          fc="#e6e6e6", fs=5.2)
    _dbox(ax, 0.015, 0.885, 0.14, 0.095, "Threat &\nlegitimacy feeds",
          fc="#e6e6e6", fs=5.2)
    proc(0.245, 0.485, 1, "Validate &\nnormalize URL")
    proc(0.43, 0.485, 2, "Feature\nextraction")
    proc(0.615, 0.485, 3, "Inference &\ncalibration")
    proc(0.86, 0.485, 7, "Respond,\npersist & alert")
    proc(0.315, 0.135, 4, "SHAP\nexplanation")
    proc(0.505, 0.135, 5, "IOC\nextraction")
    proc(0.695, 0.135, 6, "Enrichment\n(opt-in)")
    proc(0.245, 0.90, 8, "Offline\nretrain\npipeline")
    store(0.375, 0.79, 0.28, 0.075, "D3", "model.joblib - calibrated RF, threshold 0.0706")
    store(0.70, 0.26, 0.29, 0.06, "D1", "scans.db (SQLite, WAL)")
    store(0.70, 0.015, 0.29, 0.055, "D2", "alerts.log (CEF-style)")

    _dtext(ax, 0.153, 0.515, "URL", fs=4.8)
    _dtext(ax, 0.3375, 0.515, "normalized URL", fs=4.8)
    _dtext(ax, 0.5225, 0.515, "feature matrix", fs=4.8)
    _dtext(ax, 0.7375, 0.515, "result + evidence", fs=4.8)
    _dtext(ax, 0.46, 0.33, "raw output", fs=4.8)
    _dtext(ax, 0.60, 0.33, "URL", fs=4.8)
    _dtext(ax, 0.60, 0.163, "domain / IOCs", fs=4.8)
    _dtext(ax, 0.745, 0.365, "enrichment (opt-in)", fs=4.8)
    _dtext(ax, 0.93, 0.60, "opt-in\nlookups", fs=4.6)
    _dtext(ax, 0.166, 0.958, "feed files", fs=4.6)
    _dtext(ax, 0.36, 0.882, "persist bundle", fs=4.6)
    _dtext(ax, 0.645, 0.67, "reads", fs=4.8)
    _dtext(ax, 0.90, 0.375, "insert scan", fs=4.6)
    _dtext(ax, 0.86, 0.19, "alert (phishing only)", fs=4.6)
    _dtext(ax, 0.46, 0.275,
           "JSON response - verdict · probability · severity · SHAP · IOCs", fs=4.8)
    _dtext(ax, 0.42, 0.028,
           "The analyzed URL is never visited; only P6 (opt-in) and the "
           "offline feed path touch the network.", fs=4.8, style="italic",
           color="dimgray")
    fig.savefig(OUT / "figD3_dfd.png"); plt.close(fig)
    print("figD3_dfd.png  <- level-0 DFD (SRS 2.7 D3: processes 1-8, stores D1-D3)")


def fig_component():
    from matplotlib.patches import Polygon
    fig, ax = _dfig(6.3, 4.0)
    svc_x = [0.02, 0.19, 0.36, 0.53, 0.70]
    svc_w = 0.15
    centers = [x + svc_w / 2 for x in svc_x]
    # arrows first (they pass behind the boxes)
    for c in centers:
        _darrow(ax, c, 0.815, c, 0.725, lw=0.7)
    _darrow(ax, centers[0], 0.575, 0.125, 0.50)     # loads bundle
    _darrow(ax, centers[1], 0.575, 0.185, 0.50)     # raw pipeline (SHAP)
    _darrow(ax, 0.145, 0.575, 0.38, 0.50)           # features
    _darrow(ax, 0.16, 0.565, 0.60, 0.50)            # ioc_extraction
    _darrow(ax, 0.17, 0.555, 0.83, 0.50)            # whois/dns (opt-in)
    _darrow(ax, centers[3], 0.575, 0.53, 0.185)     # scan_store -> scans.db
    _darrow(ax, centers[4], 0.575, 0.80, 0.185)     # alert_logger -> alerts.log
    _darrow(ax, 0.85, 0.65, 0.875, 0.65, ls=(0, (3, 2)), lw=0.7)

    _dbox(ax, 0.03, 0.815, 0.94, 0.115,
          "app.py - Flask application (DeceptIQ console backend)\n"
          "GET / · POST /predict · POST /predict_batch · GET /history · GET /health · GET /api/model-info\n"
          "flask-limiter 30 req/min/IP · LRU cache · JSON errors 400 / 422 / 429 / 413",
          fc="#4c72b0", tc="white", fs=5.4)
    for x, t in zip(svc_x, ["prediction_\nservice", "shap_service\n(TreeSHAP)",
                            "scan_store\n(SQLite)", "alert_logger\n(CEF-style)",
                            "virustotal_\nservice (opt)"]):
        _dbox(ax, x, 0.575, svc_w, 0.15, t, fc="#dce6f2", fs=5.6)
    _dbox(ax, 0.02, 0.315, 0.21, 0.185,
          "model/\nmodel.joblib\n(calibrated RF\n+ raw pipeline\n+ threshold\n0.0706)",
          fc="#f2e6dc", fs=5.2)
    _dbox(ax, 0.27, 0.315, 0.23, 0.185,
          "features/\nurl_utils · entropy\nfeature_extraction\nfeature_builder\n(TF-IDF, train-only)",
          fs=5.2)
    _dbox(ax, 0.54, 0.315, 0.19, 0.185,
          "features/\nioc_extraction\n(credential\nredaction)", fs=5.2)
    _dbox(ax, 0.77, 0.315, 0.21, 0.185,
          "features/\nwhois_features\ndns_features\n(graceful)", fs=5.2)
    _dbox(ax, 0.42, 0.045, 0.22, 0.11, "data/scans.db\n(SQLite, WAL)", fs=5.6)
    _dbox(ax, 0.70, 0.045, 0.18, 0.11, "data/alerts.log\n(rotating)", fs=5.6)
    _dbox(ax, 0.02, 0.045, 0.30, 0.11,
          "config.yaml - all configuration (no secrets)", fs=5.4)
    _dbox(ax, 0.875, 0.585, 0.115, 0.135, "VirusTotal\nAPI v3\n(lookup\nonly)",
          fc="#e6e6e6", fs=4.8, ls=(0, (3, 2)))
    _dtext(ax, 0.8625, 0.672, "lookup", fs=4.6)
    _dtext(ax, 0.5, 0.012,
           "config.yaml configures all components · .env supplies VT_API_KEY to "
           "virustotal_service only (git-ignored, never logged)",
           fs=4.9, style="italic", color="dimgray")
    fig.savefig(OUT / "figD4_component.png"); plt.close(fig)
    print("figD4_component.png  <- component diagram (SRS 2.7 D4)")


def fig_er():
    from matplotlib.patches import Polygon
    fig, ax = _dfig(6.3, 3.6)
    attrs = [(0.14, 0.86, "id (PK)", False),
             (0.36, 0.86, "scanned_at", False),
             (0.58, 0.86, "url", False),
             (0.80, 0.86, "verdict", False),
             (0.90, 0.64, "probability", False),
             (0.90, 0.34, "severity", False),
             (0.14, 0.12, "threshold", False),
             (0.36, 0.12, "model_version", False),
             (0.58, 0.12, "shap_backend", False),
             (0.80, 0.12, "top_shap (JSON)", True),
             (0.10, 0.64, "ioc (JSON)", True)]
    for cx, cy, _, _ in attrs:
        _dline(ax, 0.50, 0.51, cx, cy)
    _dline(ax, 0.60, 0.51, 0.705, 0.51)
    _dline(ax, 0.815, 0.51, 0.88, 0.51)
    for cx, cy, t, multi in attrs:
        _dell(ax, cx, cy, 0.17 if multi else 0.19, 0.075, t, fs=5.5, double=multi)
    ax.add_patch(Polygon([(0.76, 0.552), (0.815, 0.51), (0.76, 0.468),
                          (0.705, 0.51)], closed=True, fc="white",
                         ec="black", lw=0.7, zorder=2))
    _dtext(ax, 0.76, 0.51, "produced\nby", fs=5.0)
    _dbox(ax, 0.88, 0.465, 0.11, 0.09, "MODEL\n(version)", fc="#dce6f2", fs=5.6)
    _dtext(ax, 0.655, 0.535, "1", fs=5.5)
    _dtext(ax, 0.845, 0.535, "1", fs=5.5)
    _dbox(ax, 0.40, 0.46, 0.20, 0.10, "SCAN", fc="#dce6f2", fs=8, bold=True)
    _dtext(ax, 0.50, 0.965,
           "Physical schema: one SQLite table 'scans' (WAL). Double ellipses = "
           "multi-valued JSON aggregates; one SCAN row per analyzed URL; "
           "credentials are never stored.", fs=4.9, style="italic", color="dimgray")
    fig.savefig(OUT / "figD5_er.png"); plt.close(fig)
    print("figD5_er.png  <- ER diagram (SRS 2.7 D5 / 3.3 schema)")


def fig_sequence():
    fig, ax = _dfig(6.3, 4.9)
    heads = ["Client\n(browser / curl)", "app.py\n(Flask route)", "prediction_\nservice",
             "calibrated\nmodel (RF)", "shap_service\n(TreeSHAP)", "ioc_\nextraction",
             "scan_store /\nalert_logger"]
    xs = [0.08, 0.215, 0.36, 0.505, 0.65, 0.795, 0.93]
    top, bot = 0.95, 0.03
    for x, h in zip(xs, heads):
        _dbox(ax, x - 0.065, top, 0.13, 0.07, h, fc="#4c72b0", tc="white", fs=4.9)
        ax.plot([x, x], [top, bot], color="gray", lw=0.55, ls=(0, (4, 3)), zorder=1)

    def msg(y, i, j, label, dashed=False):
        _darrow(ax, xs[i], y, xs[j], y, lw=0.75,
                ls=(0, (3, 2)) if dashed else "-")
        _dtext(ax, (xs[i] + xs[j]) / 2, y + 0.014, label, fs=4.9)

    msg(0.845, 0, 1, "POST /predict  {url}")
    msg(0.795, 1, 2, "predict(url)")
    msg(0.745, 2, 3, "predict_proba(X)")
    msg(0.695, 3, 2, "p = 0.9782 (calibrated)", dashed=True)
    _dtext(ax, 0.29, 0.662, "verdict := p ≥ 0.0706 · severity band",
           fs=4.6, style="italic", color="dimgray")
    msg(0.630, 2, 4, "explain(url)")
    msg(0.580, 4, 2, "SHAP contributions (space: probability)", dashed=True)
    msg(0.520, 2, 5, "extract_iocs(url)")
    msg(0.470, 5, 2, "components + indicator flags", dashed=True)
    ax.add_patch(FancyBboxPatch((0.14, 0.395), 0.84, 0.055,
                                boxstyle="square,pad=0", fc="none", ec="black",
                                lw=0.6, ls=(0, (3, 2)), zorder=2))
    _dtext(ax, 0.155, 0.435, "opt [include_enrichment | include_external]",
           fs=4.8, ha="left")
    msg(0.415, 2, 6, "WHOIS / DNS / VirusTotal lookups - strict timeouts, never cached",
        dashed=True)
    msg(0.355, 2, 6, "record(scan row)")
    msg(0.305, 6, 2, "scan_id · scanned_at", dashed=True)
    ax.add_patch(FancyBboxPatch((0.14, 0.225), 0.84, 0.055,
                                boxstyle="square,pad=0", fc="none", ec="black",
                                lw=0.6, ls=(0, (3, 2)), zorder=2))
    _dtext(ax, 0.155, 0.265, "alt [verdict == phishing]", fs=4.8, ha="left")
    msg(0.245, 2, 6, "alert_logger: CEF line → data/alerts.log", dashed=True)
    _dbox(ax, 0.14, 0.145, 0.84, 0.05,
          "The analyzed URL is never visited - every computation is local to "
          "the URL string.", fc="#fdf6e3", fs=5.0)
    msg(0.09, 1, 0, "200 OK · JSON result (verdict, p, severity, SHAP, IOCs)")
    fig.savefig(OUT / "figD6_sequence.png"); plt.close(fig)
    print("figD6_sequence.png  <- sequence diagram, POST /predict (SRS 2.7 D6)")


if __name__ == "__main__":
    if "d" not in dir() or d is None:
        d = _load()
    for fn in (fig_usecase, fig_dfd0, fig_component, fig_er, fig_sequence):
        try:
            fn()
        except Exception as exc:
            print(f"FAILED {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"DIAGRAMS -> {OUT}")
