"""
Cross-check every number in the paper's Tables I-III against the result files in outputs/.

Usage (from repo root):
    python scripts/check_paper.py "path/to/Fair_bilevel_classfication.pdf"
    python scripts/check_paper.py paper.txt            # or a text export / the .tex source

For each table cell it recomputes mean ± std over the 5 seeds from the JSON result files and checks
that the exact string "m.mmm ± s.sss" occurs in the paper. Prints OK / MISSING per cell and a final
count. Needs only numpy (and pdftotext, or the pypdf/pdfplumber package, if you pass a PDF).
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


def paper_text(path):
    if not path.lower().endswith(".pdf"):
        return open(path, encoding="utf-8", errors="ignore").read()
    try:
        return subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True,
                              check=True).stdout
    except Exception:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            sys.exit("Install pdfplumber (pip install pdfplumber) or pass a .txt/.tex export.")


def runs(tag):
    files = sorted(glob.glob(os.path.join(OUT, f"draft_results_{tag}_seed[0-9].json")))
    return [json.load(open(f)) for f in files]


def ms(vals):
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


METRICS = {
    "Acc.": lambda m: m["accuracy"], "F1": lambda m: m["F1_score"],
    "Bal. Acc.": lambda m: m["extended_metrics"]["balanced_accuracy"],
    "PR-AUC": lambda m: m["extended_metrics"]["PR_AUC"],
    "DemP gap": lambda m: m["extended_metrics"]["DP_gap"], "EO gap": lambda m: m["EO_gap"],
    "EOD gap": lambda m: m["extended_metrics"]["EOD_gap"],
}


def cells():
    """Yield (table, row label, metric, expected string)."""
    for ds in ["credit", "adult", "law", "compas"]:
        r = runs(f"{ds}_final")
        if len(r) != 5:
            print(f"!! {ds}_final has {len(r)} seeds"); continue
        for who, key in (("Baseline", "baseline"), ("Ours", "pipeline")):
            for met in ["Acc.", "F1", "Bal. Acc.", "PR-AUC"]:
                yield "I", f"{ds} {who}", met, ms([METRICS[met](x[key]) for x in r])
            for met in ["DemP gap", "EO gap", "EOD gap"]:
                yield "II", f"{ds} {who}", met, ms([METRICS[met](x[key]) for x in r])
    # Table III: the "other" Universum setting
    for ds, tag, label in (("credit", "credit_nouniv", "Off"), ("adult", "adult_nouniv", "Off"),
                           ("compas", "compas_withuniv", "On")):
        r = runs(tag)
        if len(r) != 5:
            print(f"!! {tag} has {len(r)} seeds"); continue
        for met in ["Acc.", "F1", "EO gap", "DemP gap", "EOD gap"]:
            yield "III", f"{ds} Universum {label}", met, ms([METRICS[met](x["pipeline"]) for x in r])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    txt = paper_text(sys.argv[1])
    norm = re.sub(r"\s*(±|\\pm|\$\\pm\$)\s*", " ± ", txt.replace("$", ""))
    ok = bad = 0
    for tab, row, met, exp in cells():
        if exp in norm:
            ok += 1; print(f"OK       Table {tab:<3} {row:<26} {met:<10} {exp}")
        else:
            bad += 1; print(f"MISSING  Table {tab:<3} {row:<26} {met:<10} expected {exp}")
    print(f"\n{ok} cells match, {bad} missing/different.")


if __name__ == "__main__":
    main()
