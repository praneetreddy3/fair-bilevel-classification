"""
Load and preprocess the 'Default of Credit Card Clients' dataset for the fair bilevel pipeline.
Dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Target : default payment next month  (1 = default / bad, 0 = no default / good)
Sensitive: sex  (male = 1, female = 0)   [secondary option: age group]

Drop the downloaded file into a folder named  CreditData/  inside the project root.
Accepted file names (any one):
    default_of_credit_card_clients.xls / .xlsx / .csv
    default of credit card clients.xls / .xlsx / .csv
    UCI_Credit_Card.csv            (common Kaggle export)
Or, if `ucimlrepo` is installed and you have internet, it is fetched automatically (id=350).
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDIT_DIR = os.path.join(PROJECT_ROOT, "CreditData")


def _find_local_file() -> str | None:
    if not os.path.isdir(CREDIT_DIR):
        return None
    for ext in ("*.xls", "*.xlsx", "*.csv"):
        hits = glob.glob(os.path.join(CREDIT_DIR, ext))
        if hits:
            return sorted(hits)[0]
    return None


def _read_raw() -> pd.DataFrame:
    """Return a DataFrame with the real column names (LIMIT_BAL, SEX, ... , default...)."""
    path = _find_local_file()
    if path is not None:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
            # Some CSVs carry the descriptive first row; detect & fix.
            if "LIMIT_BAL" not in df.columns and df.shape[1] > 1:
                df = pd.read_csv(path, header=1)
        else:
            # UCI .xls has a descriptive row first -> real header is row index 1.
            df = pd.read_excel(path, header=1)
        return df
    # Fallback: ucimlrepo (needs internet)
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=350)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        return df
    except Exception as e:
        raise FileNotFoundError(
            f"No credit file found in {CREDIT_DIR} and ucimlrepo fetch failed ({e}). "
            f"Put the downloaded .xls/.csv into {CREDIT_DIR}."
        )


def _col(df: pd.DataFrame, *candidates: str) -> str:
    low = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    # loose contains-match (e.g. 'default payment next month')
    for c in df.columns:
        if any(cand.lower() in c.lower() for cand in candidates):
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def prepare_credit_for_draft(sensitive: str = "sex", test_size: float = 0.2, seed: int = 42) -> tuple:
    """Return (X_train, A_train, Y_train), (X_test, A_test, Y_test) with A,Y in {0,1}."""
    df = _read_raw()

    # Identify target, sensitive, id columns.
    target_col = _col(df, "default payment next month", "default.payment.next.month", "Y", "default")
    sex_col = _col(df, "SEX")
    age_col = _col(df, "AGE")
    id_candidates = [c for c in df.columns if c.lower() in ("id", "unnamed: 0")]

    # Label: 1 = default.
    Y = (pd.to_numeric(df[target_col], errors="coerce") == 1).astype(np.float64).values

    # Sensitive attribute.
    if sensitive == "sex":
        # UCI encoding: 1 = male, 2 = female  ->  male = 1, female = 0
        A = (pd.to_numeric(df[sex_col], errors="coerce") == 1).astype(np.float64).values
    else:  # 'race' not available -> use age group (>= median age = 1)
        age = pd.to_numeric(df[age_col], errors="coerce")
        A = (age >= age.median()).astype(np.float64).values

    # Features = everything except target, id, and the sensitive column (A is passed separately).
    drop = {target_col, sex_col, *id_candidates}
    feat_cols = [c for c in df.columns if c not in drop]
    Xdf = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    # Median imputation for numeric columns (project standard).
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    X = Xdf.values.astype(np.float64)

    # Stratified split; runner applies StandardScaler after the split.
    X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(
        X, A, Y, test_size=test_size, random_state=seed, stratify=Y
    )
    return (X_tr, A_tr, Y_tr), (X_te, A_te, Y_te)
