"""
Load and preprocess the Law School (law) dataset for the fair bilevel pipeline.
Source: FairSynData/rawdata/law.csv (the reference paper's own dataset).
Target : pass_bar  (1 = passed the bar, 0 = did not)
Sensitive: race  (White = 1, Non-White = 0)

Running our model on Law lets us compare directly against the reference (FairSynData)
on its own dataset.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAW_CSV = os.path.join(PROJECT_ROOT, "FairSynData", "rawdata", "law.csv")


def prepare_law_for_draft(sensitive: str = "race", test_size: float = 0.2, seed: int = 42) -> tuple:
    """Return (X_train, A_train, Y_train), (X_test, A_test, Y_test) with A,Y in {0,1}."""
    if not os.path.isfile(LAW_CSV):
        raise FileNotFoundError(f"Law data not found at {LAW_CSV}")
    df = pd.read_csv(LAW_CSV)

    # Label: passed the bar.
    Y = (pd.to_numeric(df["pass_bar"], errors="coerce") == 1).astype(np.float64).values

    # Sensitive attribute: race (White = 1, Non-White = 0). 'male' stays a feature.
    A = (df["race"].astype(str).str.strip().str.lower() == "white").astype(np.float64).values

    # Features = everything except the label and the sensitive column (A is passed separately).
    feat_cols = [c for c in df.columns if c not in ("race", "pass_bar")]
    Xdf = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    X = Xdf.values.astype(np.float64)

    X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(
        X, A, Y, test_size=test_size, random_state=seed, stratify=Y
    )
    return (X_tr, A_tr, Y_tr), (X_te, A_te, Y_te)
