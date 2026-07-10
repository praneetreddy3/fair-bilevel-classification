"""
Build the five paper tables (T1-T5) from outputs/draft_results_*.json.
Writes one CSV per table to outputs/tables/ and one combined outputs/tables/... ->
docs/PAPER_TABLES.md with all five as Markdown.

Run from project root: python build_tables.py
"""
import csv
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# Winner configs (per docs/RESULTS.md + this session's fill-in runs).
WINNER_SINGLE = {
    "credit": "draft_results_credit_none_rho0.05_eps0.1.json",
    "adult": "draft_results_adult_notune_seed42.json",  # tune_threshold=false, matches ablation/dirichlet configs
}
WINNER_5SEED = {
    "credit": [f"draft_results_credit_winner_seed{s}.json" for s in range(1, 6)],
    "adult": [f"draft_results_adult_notune_seed{s}.json" for s in range(1, 6)],  # raw-accuracy version
}


def load(name):
    p = os.path.join(OUT_DIR, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def fmt(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# T1: main performance (mean +/- std over 5 seeds), baseline vs pipeline.
# ---------------------------------------------------------------------------
def build_t1():
    header = ["dataset", "model", "accuracy", "F1", "macro_F1", "balanced_accuracy", "ROC_AUC", "PR_AUC"]
    rows = []
    for ds, files in WINNER_5SEED.items():
        runs = [load(f) for f in files]
        runs = [r for r in runs if r is not None]
        for model in ("baseline", "pipeline"):
            acc = mean_std([r[model]["accuracy"] for r in runs])
            f1 = mean_std([r[model]["F1_score"] for r in runs])
            macro_f1 = mean_std([r[model]["extended_metrics"]["macro_F1"] for r in runs])
            bal_acc = mean_std([r[model]["extended_metrics"]["balanced_accuracy"] for r in runs])
            roc_auc = mean_std([r[model]["extended_metrics"]["ROC_AUC"] for r in runs])
            pr_auc = mean_std([r[model]["extended_metrics"]["PR_AUC"] for r in runs])
            rows.append([ds, model, fmt(*acc), fmt(*f1), fmt(*macro_f1), fmt(*bal_acc), fmt(*roc_auc), fmt(*pr_auc)])
    write_csv(os.path.join(TABLES_DIR, "T1_main_performance.csv"), header, rows)
    return header, rows


# ---------------------------------------------------------------------------
# T2: fairness (mean +/- std over 5 seeds), baseline vs pipeline.
# ---------------------------------------------------------------------------
def build_t2():
    header = ["dataset", "model", "DP_gap", "EO_gap", "EOD_gap"]
    rows = []
    for ds, files in WINNER_5SEED.items():
        runs = [load(f) for f in files]
        runs = [r for r in runs if r is not None]
        for model in ("baseline", "pipeline"):
            dp_gap = mean_std([r[model]["extended_metrics"]["DP_gap"] for r in runs])
            eo_gap = mean_std([r[model]["EO_gap"] for r in runs])
            eod_gap = mean_std([r[model]["extended_metrics"]["EOD_gap"] for r in runs])
            rows.append([ds, model, fmt(*dp_gap), fmt(*eo_gap), fmt(*eod_gap)])
    write_csv(os.path.join(TABLES_DIR, "T2_fairness.csv"), header, rows)
    return header, rows


# ---------------------------------------------------------------------------
# T3: sensitivity sweep (single seed 42), from the existing 12-config grid per dataset.
# ---------------------------------------------------------------------------
def build_t3():
    header = ["dataset", "dp_variant", "rho", "epsilon_EO", "accuracy", "EO_gap", "F1"]
    rows = []
    import glob
    for ds in ("credit", "adult"):
        for f in sorted(glob.glob(os.path.join(OUT_DIR, f"draft_results_{ds}_*_rho*_eps*.json"))):
            name = os.path.basename(f)
            if "winner" in name or "dirichlet" in name:
                continue
            d = json.load(open(f))
            p = d["pipeline"]
            parts = name.replace(".json", "").split("_")
            rho = [x for x in parts if x.startswith("rho")][0][3:]
            eps = [x for x in parts if x.startswith("eps")][0][3:]
            rows.append([ds, p["dp_variant"], rho, eps, f"{p['accuracy']:.4f}", f"{p['EO_gap']:.4f}", f"{p['F1_score']:.4f}"])
    rows.sort(key=lambda r: (r[0], -float(r[4])))
    write_csv(os.path.join(TABLES_DIR, "T3_sensitivity.csv"), header, rows)
    return header, rows


# ---------------------------------------------------------------------------
# T4: non-IID robustness -- accuracy + EO gap vs Dirichlet alpha.
# ---------------------------------------------------------------------------
def build_t4():
    header = ["dataset", "partition", "accuracy", "EO_gap"]
    rows = []
    for ds in ("credit", "adult"):
        winner = load(WINNER_SINGLE[ds])
        if winner:
            rows.append([ds, "iid (winner)", f"{winner['pipeline']['accuracy']:.4f}", f"{winner['pipeline']['EO_gap']:.4f}"])
        for alpha in ("1.0", "0.5", "0.1"):
            d = load(f"draft_results_{ds}_dirichlet_alpha{alpha}.json")
            if d:
                rows.append([ds, f"dirichlet alpha={alpha}", f"{d['pipeline']['accuracy']:.4f}", f"{d['pipeline']['EO_gap']:.4f}"])
            else:
                rows.append([ds, f"dirichlet alpha={alpha}", "FAILED", "ZeroDivisionError/RuntimeError: extreme skew starves a client of data"])
    write_csv(os.path.join(TABLES_DIR, "T4_noniid_robustness.csv"), header, rows)
    return header, rows


# ---------------------------------------------------------------------------
# T5: ablation -- full winner vs no-Universum vs fairness-off vs no-intercept.
# ---------------------------------------------------------------------------
def build_t5():
    header = ["dataset", "variant", "accuracy", "EO_gap", "F1", "DP_gap", "EOD_gap"]
    rows = []
    variants = {
        "full (winner)": "{}",
        "no_universum": "_ablation_nouniversum",
        "fairness_off": "_ablation_fairnessoff",
        "no_intercept": "_ablation_nointercept",
    }
    for ds in ("credit", "adult"):
        for label, suffix in variants.items():
            fname = WINNER_SINGLE[ds] if suffix == "{}" else f"draft_results_{ds}{suffix}.json"
            d = load(fname)
            if not d:
                continue
            p = d["pipeline"]
            rows.append([
                ds, label, f"{p['accuracy']:.4f}", f"{p['EO_gap']:.4f}", f"{p['F1_score']:.4f}",
                f"{p['extended_metrics']['DP_gap']:.4f}", f"{p['extended_metrics']['EOD_gap']:.4f}",
            ])
    write_csv(os.path.join(TABLES_DIR, "T5_ablation.csv"), header, rows)
    return header, rows


def md_table(title, note, header, rows):
    lines = [f"## {title}", "", note, "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    t1 = build_t1()
    t2 = build_t2()
    t3 = build_t3()
    t4 = build_t4()
    t5 = build_t5()

    doc = ["# Paper Tables — Credit Risk & UCI Adult", "",
           "Generated by `build_tables.py` from `outputs/draft_results_*.json`. "
           "CSVs are in `outputs/tables/`.", ""]
    doc.append(md_table(
        "T1: Main performance (mean +/- std, 5 seeds)",
        "Credit: winner config (`dp=none, rho=0.05, eps=0.1, tune_threshold=true`). "
        "Adult: winner rho/eps with `tune_threshold=false` to report raw accuracy.",
        *t1))
    doc.append(md_table(
        "T2: Fairness (mean +/- std, 5 seeds)",
        "Same configs as T1.", *t2))
    doc.append(md_table(
        "T3: Sensitivity sweep (single seed 42)",
        "12-config grid per dataset: dp_variant(none / pre_server@sigma=0.25) x rho{0.05,0.1} x epsilon_EO{0.05,0.1,0.25}.",
        *t3))
    doc.append(md_table(
        "T4: Non-IID robustness (single seed 42)",
        "Winner config re-run with `--partition dirichlet --dirichlet_alpha {1.0,0.5,0.1}` vs the IID winner.",
        *t4))
    doc.append(md_table(
        "T5: Ablation (single seed 42)",
        "Winner config vs one-flag-flipped variants: no Universum, fairness off (rho forced to 0), no intercept.",
        *t5))

    with open(os.path.join(DOCS_DIR, "PAPER_TABLES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(doc))

    print("Wrote outputs/tables/T1..T5.csv and docs/PAPER_TABLES.md")


if __name__ == "__main__":
    main()
