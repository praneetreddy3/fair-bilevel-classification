"""
Load and preprocess the ProPublica COMPAS "Two Years" recidivism dataset for the
fair bilevel pipeline.

Source: the original ProPublica investigation dataset (Angwin et al. 2016,
"Machine Bias"), compas-scores-two-years.csv — 7,214 defendants scored by the
COMPAS risk tool in Broward County, FL, with 2-year re-arrest outcomes.
Target    : two_year_recid  (1 = re-arrested within 2 years, 0 = not)
Sensitive : race            (Caucasian = 1, African-American = 0)

This is a real-world criminal-justice risk-assessment dataset, distinct in
domain from Adult/Credit/Law (finance/education) — used here as the additional
real-life application requested for the submission.

Filtering follows the standard preprocessing used by ProPublica's own analysis
and nearly all downstream fairness benchmarks (e.g. IBM AIF360's
CompasDataset default_preprocessing):
  - |days_b_screening_arrest| <= 30   (charge date close to COMPAS screening)
  - is_recid != -1                    (recidivism status known)
  - c_charge_degree != 'O'            (ordinary traffic offenses excluded)
  - score_text != 'N/A'               (a COMPAS score was actually produced)
  - race in {African-American, Caucasian}  (the standard two-group comparison
    used in the original ProPublica study and essentially all follow-up work;
    the other race categories are too small for a stable per-group TPR here)

Drop the CSV into a folder named  CompasData/  inside the project root
(file: compas-scores-two-years.csv). If missing, falls back to extracting the
identical file bundled inside the `responsibly` PyPI package
(pip install responsibly) — no separate download needed either way.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPAS_DIR = os.path.join(PROJECT_ROOT, "CompasData")
COMPAS_CSV = os.path.join(COMPAS_DIR, "compas-scores-two-years.csv")

FEATURE_COLS = [
    "sex", "age", "age_cat", "priors_count",
    "c_charge_degree", "juv_fel_count", "juv_misd_count", "juv_other_count",
]


def _read_raw() -> pd.DataFrame:
    if os.path.isfile(COMPAS_CSV):
        return pd.read_csv(COMPAS_CSV)
    # Fallback: pull the same file bundled inside the `responsibly` package.
    try:
        import responsibly
        pkg_dir = os.path.dirname(responsibly.__file__)
        bundled = os.path.join(pkg_dir, "dataset", "compas", "compas-scores-two-years.csv")
        if os.path.isfile(bundled):
            return pd.read_csv(bundled)
        raise FileNotFoundError(bundled)
    except Exception as e:
        raise FileNotFoundError(
            f"COMPAS data not found at {COMPAS_CSV} and the `responsibly` fallback failed ({e}). "
            f"Put compas-scores-two-years.csv into {COMPAS_DIR}, or `pip install responsibly`."
        )


def _apply_standard_filters(df: pd.DataFrame) -> pd.DataFrame:
    """ProPublica / AIF360-standard quality filters (see module docstring)."""
    d_b = pd.to_numeric(df["days_b_screening_arrest"], errors="coerce")
    mask = (
        d_b.between(-30, 30)
        & (pd.to_numeric(df["is_recid"], errors="coerce") != -1)
        & (df["c_charge_degree"].astype(str) != "O")
        & (df["score_text"].astype(str) != "N/A")
        & (df["race"].isin(["African-American", "Caucasian"]))
    )
    return df.loc[mask].reset_index(drop=True)


def prepare_compas_for_draft(sensitive: str = "race", test_size: float = 0.2, seed: int = 42) -> tuple:
    """Return (X_train, A_train, Y_train), (X_test, A_test, Y_test) with A,Y in {0,1}."""
    if sensitive != "race":
        raise ValueError("COMPAS only supports sensitive='race' (Caucasian=1 / African-American=0)")

    df = _read_raw()
    df = _apply_standard_filters(df)
    if len(df) == 0:
        raise ValueError("COMPAS filtering left zero rows — check the source CSV.")

    # Label: re-arrested within two years.
    Y = (pd.to_numeric(df["two_year_recid"], errors="coerce") == 1).astype(np.float64).values

    # Sensitive attribute: race (Caucasian = 1, African-American = 0) — matches the
    # privileged=1 convention already used for Law's race attribute in this repo.
    A = (df["race"].astype(str) == "Caucasian").astype(np.float64).values

    Xdf = df[FEATURE_COLS].copy()
    for col in ("sex", "age_cat", "c_charge_degree"):
        Xdf[col] = Xdf[col].astype(str)
    cat_cols = ["sex", "age_cat", "c_charge_degree"]
    Xdf = pd.get_dummies(Xdf, columns=cat_cols, drop_first=True)
    for col in Xdf.columns:
        Xdf[col] = pd.to_numeric(Xdf[col], errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    X = Xdf.values.astype(np.float64)

    X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(
        X, A, Y, test_size=test_size, random_state=seed, stratify=Y
    )
    return (X_tr, A_tr, Y_tr), (X_te, A_te, Y_te)
