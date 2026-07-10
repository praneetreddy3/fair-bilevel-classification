"""
Load and preprocess the UCI Adult dataset for the fair bilevel pipeline.
Dataset: https://archive.ics.uci.edu/dataset/2/adult
Target: income >50K (1) vs <=50K (0)
Sensitive: sex (Female=0, Male=1) or race (Non-White=0, White=1)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Preprocessed CSVs (legacy FairSynData layout)
ADULT_DIR = os.path.join(PROJECT_ROOT, "FairSynData", "datasets", "adult")
# Raw UCI files: adult.data, adult.test (see https://archive.ics.uci.edu/dataset/2/adult)
UCI_ADULT_FOLDER = os.path.join(PROJECT_ROOT, "UCIAdultdataset")

# Column order matches adult.names (UCI distribution)
ADULT_UCI_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def _extract_ay(df: pd.DataFrame, sensitive: str) -> tuple:
    """Extract A and Y from dataframe."""
    if "income" in df.columns:
        y_raw = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
        Y = (y_raw == ">50K").astype(np.float64).values
    elif "target" in df.columns:
        y_raw = df["target"].astype(str).str.strip().str.replace(".", "", regex=False)
        Y = (y_raw == ">50K").astype(np.float64).values
    else:
        raise ValueError("No income/target column found")

    if sensitive == "sex":
        A = (df["sex"].astype(str).str.strip().str.lower() == "male").astype(np.float64).values
    elif sensitive == "race":
        A = (df["race"].astype(str).str.strip().str.lower() == "white").astype(np.float64).values
    else:
        raise ValueError("sensitive must be 'sex' or 'race'")
    return A, Y


def _encode_features(train_df: pd.DataFrame, test_df: pd.DataFrame, drop_cols: list) -> tuple:
    """
    Encode features so train and test have identical columns (avoid different one-hot dims).
    Concatenate, encode, then split to ensure same schema.
    """
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    for col in X_train.columns:
        if X_train[col].dtype == object:
            mode_val = X_train[col].replace("?", np.nan).mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else ""
            X_train[col] = X_train[col].replace("?", fill_val)
            X_test[col] = X_test[col].replace("?", fill_val)

    # Concatenate to get identical one-hot columns, then split
    n_train = len(X_train)
    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    combined = np.nan_to_num(combined.astype(np.float64).values, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = combined[:n_train]
    X_test = combined[n_train:]
    return X_train, X_test


def _preprocess_adult(df: pd.DataFrame, sensitive: str = "sex") -> tuple:
    """
    Preprocess Adult dataframe to (X, A, Y). Use for single-df case (e.g. CSV).
    """
    df = df.copy()
    A, Y = _extract_ay(df, sensitive)
    drop_cols = ["income", "target", "sex", "race"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace("?", np.nan)
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = np.nan_to_num(df.astype(np.float64).values, nan=0.0, posinf=0.0, neginf=0.0)
    return X, A, Y


def load_adult_from_uci_folder(folder: Optional[str] = None, sensitive: str = "sex"):
    """
    Load from raw UCI Adult files ``adult.data`` (train) and ``adult.test`` (test).
    Uses the standard UCI train/test split (not a random split).

    ``folder`` defaults to ``<project>/UCIAdultdataset``.
    """
    base = folder or UCI_ADULT_FOLDER
    train_path = os.path.join(base, "adult.data")
    test_path = os.path.join(base, "adult.test")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. "
            "Place UCI Adult adult.data and adult.test in UCIAdultdataset/."
        )

    train_df = pd.read_csv(
        train_path,
        header=None,
        names=ADULT_UCI_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )
    # adult.test often starts with a non-data line (e.g. "|1x3 Cross validator")
    test_df = pd.read_csv(
        test_path,
        header=None,
        names=ADULT_UCI_COLUMNS,
        na_values="?",
        skipinitialspace=True,
        skiprows=1,
    )
    # Normalize label: test file may use ">50K." with a trailing period
    for df in (train_df, test_df):
        df["income"] = (
            df["income"].astype(str).str.strip().str.rstrip(".").str.replace(" ", "", regex=False)
        )

    A_train, Y_train = _extract_ay(train_df, sensitive)
    A_test, Y_test = _extract_ay(test_df, sensitive)
    drop_cols = ["income", "target", "sex", "race"]
    X_train, X_test = _encode_features(train_df, test_df, drop_cols)

    return (X_train, A_train, Y_train), (X_test, A_test, Y_test)


def load_adult_from_ucimlrepo(sensitive: str = "sex", test_size: float = 0.2, seed: int = 42):
    """
    Load Adult via ucimlrepo. Requires: pip install ucimlrepo
    Returns (X_train, A_train, Y_train), (X_test, A_test, Y_test)
    """
    try:
        from ucimlrepo import fetch_ucirepo  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Install ucimlrepo: pip install ucimlrepo")

    adult = fetch_ucirepo(id=2)
    df = adult.data.features.copy()
    target_col = adult.data.targets.iloc[:, 0]
    df["income"] = target_col.astype(str).str.strip().str.replace(".", "", regex=False)

    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    A_train, Y_train = _extract_ay(train_df, sensitive)
    A_test, Y_test = _extract_ay(test_df, sensitive)
    drop_cols = ["income", "target", "sex", "race"]
    X_train, X_test = _encode_features(train_df, test_df, drop_cols)

    return (X_train, A_train, Y_train), (X_test, A_test, Y_test)


def load_adult_from_csv(sensitive: str = "sex"):
    """
    Load from preprocessed CSVs in FairSynData/datasets/adult/.
    Expects: adult_train.csv, adult_test.csv with columns including income, sex, race.
    """
    train_path = os.path.join(ADULT_DIR, "adult_train.csv")
    test_path = os.path.join(ADULT_DIR, "adult_test.csv")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Adult data not found. Create {ADULT_DIR}/adult_train.csv and adult_test.csv, "
            "or use load_adult_from_ucimlrepo() after: pip install ucimlrepo"
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train, A_train, Y_train = _preprocess_adult(train, sensitive)
    X_test, A_test, Y_test = _preprocess_adult(test, sensitive)
    return (X_train, A_train, Y_train), (X_test, A_test, Y_test)


def prepare_adult_for_draft(
    sensitive: str = "sex",
    use_ucimlrepo: bool = True,
    uci_folder: Optional[str] = None,
):
    """
    Main entry: returns data in format expected by run_draft.py.

    Priority when ``use_ucimlrepo`` is True:
    1. Local ``UCIAdultdataset/adult.data`` + ``adult.test`` (if present)
    2. Fetch via ``ucimlrepo`` (requires ``pip install ucimlrepo``)

    Set ``use_ucimlrepo=False`` to load only from FairSynData CSVs
    (``FairSynData/datasets/adult/adult_train.csv``).
    """
    if not use_ucimlrepo:
        return load_adult_from_csv(sensitive=sensitive)

    folder = uci_folder or UCI_ADULT_FOLDER
    data_path = os.path.join(folder, "adult.data")
    test_path = os.path.join(folder, "adult.test")
    if os.path.isfile(data_path) and os.path.isfile(test_path):
        return load_adult_from_uci_folder(folder=folder, sensitive=sensitive)
    return load_adult_from_ucimlrepo(sensitive=sensitive)
