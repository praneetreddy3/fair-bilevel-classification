"""
Interactive results plotter -- hover over a line to see its run name,
instead of reading a giant static color legend.

Same data source as plot_results.py (outputs/draft_results_*.json).
Produces an HTML file you open in a browser (not a PNG -- hover tooltips
only work in an interactive format, they can't be baked into a static image
or into the paper's PDF).

Requires: pip install plotly

Run:    python plot_results_interactive.py
Output: outputs/results_convergence_interactive.html
"""
import os, glob, json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        runs.append({
            "label": _label(p),
            "dataset": d.get("dataset", "?"),
            "round_logs": d.get("round_logs", []),
        })
    return runs


def plot_convergence_interactive(runs):
    runs = [r for r in runs if r["round_logs"]]
    if not runs:
        print("No runs with round_logs found.")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Validation accuracy per round", "Validation EO gap per round"
    ))

    for r in runs:
        name = f'{r["dataset"]}:{r["label"]}'
        rounds = [x["round"] for x in r["round_logs"]]
        acc = [x.get("val_accuracy") * 100 if x.get("val_accuracy") is not None else None
               for x in r["round_logs"]]
        eo = [x.get("val_EO_gap") * 100 if x.get("val_EO_gap") is not None else None
              for x in r["round_logs"]]

        fig.add_trace(
            go.Scatter(
                x=rounds, y=acc, mode="lines+markers", name=name,
                hovertemplate=f"<b>{name}</b><br>round=%{{x}}<br>accuracy=%{{y:.1f}}%<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rounds, y=eo, mode="lines+markers", name=name,
                hovertemplate=f"<b>{name}</b><br>round=%{{x}}<br>EO gap=%{{y:.1f}}%<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="round", row=1, col=1)
    fig.update_xaxes(title_text="round", row=1, col=2)
    fig.update_yaxes(title_text="accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="EO gap (%) -- lower is fairer", row=1, col=2)
    fig.update_layout(
        title_text="Convergence (hover over any line to see which run it is)",
        height=550, width=1200,
    )

    out_path = os.path.join(OUT, "results_convergence_interactive.html")
    fig.write_html(out_path)
    print(f"Saved interactive plot to {out_path}")
    print("Open it in a browser and hover over any line to see its run name.")


if __name__ == "__main__":
    runs = load_runs()
    if not runs:
        print(f"No draft_results_*.json found in {OUT}. Run the pipeline first.")
    else:
        plot_convergence_interactive(runs)
