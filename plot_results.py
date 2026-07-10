"""
Reusable results plotter.

Scans outputs/draft_results_*.json (whatever configs you've run) and regenerates:
  - outputs/results_pareto.png      accuracy vs EO gap, baseline as a star, best = top-left
  - outputs/results_bars.png        baseline vs pipeline accuracy & EO gap per config
  - outputs/results_convergence.png val accuracy / EO gap per round (if round_logs present)

Run after any re-run:   python plot_results.py
No hardcoded experiment pairs — it just plots whatever JSONs are in outputs/.
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _label(path: str) -> str:
    b = os.path.basename(path)
    b = b.replace("draft_results_", "").replace(".json", "")
    return b if b else "run"


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, "draft_results_*.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        base, pipe = d.get("baseline", {}), d.get("pipeline", {})
        runs.append({
            "label": _label(p),
            "dataset": d.get("dataset", "?"),
            "base_acc": base.get("accuracy"), "base_eo": base.get("EO_gap"),
            "pipe_acc": pipe.get("accuracy"), "pipe_eo": pipe.get("EO_gap"),
            "base_f1": base.get("F1_score"), "pipe_f1": pipe.get("F1_score"),
            "round_logs": d.get("round_logs", []),
        })
    return runs


def plot_pareto(runs):
    fig, ax = plt.subplots(figsize=(9, 6))
    seen_base = set()
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(runs):
        if r["pipe_acc"] is None:
            continue
        c = cmap(i % 10)
        ax.scatter(r["pipe_eo"] * 100, r["pipe_acc"] * 100, s=150, color=c,
                   edgecolor="black", zorder=3, label=f'{r["dataset"]}:{r["label"]}')
        ax.annotate(r["label"], (r["pipe_eo"] * 100, r["pipe_acc"] * 100),
                    fontsize=7, xytext=(6, 4), textcoords="offset points")
        key = (r["dataset"], round(r["base_acc"] or 0, 4))
        if r["base_acc"] is not None and key not in seen_base:
            seen_base.add(key)
            ax.scatter(r["base_eo"] * 100, r["base_acc"] * 100, s=320, marker="*",
                       color="black", zorder=4)
            ax.annotate(f'{r["dataset"]} baseline', (r["base_eo"] * 100, r["base_acc"] * 100),
                        fontsize=7, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("EO gap (%) — lower is fairer")
    ax.set_ylabel("Accuracy (%) — higher is better")
    ax.set_title("Accuracy–Fairness trade-off (best = top-left)", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "results_pareto.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_bars(runs):
    runs = [r for r in runs if r["pipe_acc"] is not None]
    if not runs:
        return
    labels = [f'{r["dataset"]}\n{r["label"]}' for r in runs]
    x = np.arange(len(runs)); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(10, 1.8 * len(runs)), 5))
    a1.bar(x - w / 2, [r["base_acc"] * 100 for r in runs], w, label="Baseline", color="#9aa0a6")
    a1.bar(x + w / 2, [r["pipe_acc"] * 100 for r in runs], w, label="Pipeline", color="#2a9d8f")
    a1.set_title("Accuracy (%)", fontweight="bold"); a1.set_xticks(x)
    a1.set_xticklabels(labels, fontsize=7); a1.legend(); a1.grid(axis="y", alpha=0.3)
    a2.bar(x - w / 2, [r["base_eo"] * 100 for r in runs], w, label="Baseline", color="#9aa0a6")
    a2.bar(x + w / 2, [r["pipe_eo"] * 100 for r in runs], w, label="Pipeline", color="#e76f51")
    a2.set_title("EO gap (%) — lower is fairer", fontweight="bold"); a2.set_xticks(x)
    a2.set_xticklabels(labels, fontsize=7); a2.legend(); a2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "results_bars.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_convergence(runs):
    runs = [r for r in runs if r["round_logs"]]
    if not runs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for r in runs:
        rounds = [x["round"] for x in r["round_logs"]]
        a1.plot(rounds, [x.get("val_accuracy", np.nan) * 100 for x in r["round_logs"]],
                marker="o", label=f'{r["dataset"]}:{r["label"]}')
        a2.plot(rounds, [x.get("val_EO_gap", np.nan) * 100 for x in r["round_logs"]],
                marker="o", label=f'{r["dataset"]}:{r["label"]}')
    a1.set_title("Validation accuracy per round", fontweight="bold")
    a1.set_xlabel("round"); a1.set_ylabel("accuracy (%)"); a1.grid(alpha=0.3); a1.legend(fontsize=7)
    a2.set_title("Validation EO gap per round", fontweight="bold")
    a2.set_xlabel("round"); a2.set_ylabel("EO gap (%)"); a2.grid(alpha=0.3); a2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "results_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    runs = load_runs()
    if not runs:
        print(f"No draft_results_*.json found in {OUT}. Run the pipeline first.")
    else:
        plot_pareto(runs); plot_bars(runs); plot_convergence(runs)
        print(f"Plotted {len(runs)} run(s). Figures saved in {OUT}/:")
        print("  results_pareto.png, results_bars.png, results_convergence.png")
