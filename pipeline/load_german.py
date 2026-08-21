"""
Load and preprocess the UCI "Statlog (German Credit Data)" dataset for the fair
bilevel pipeline.
Dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Target : credit risk (1 = bad/high-risk, 0 = good/low-risk)
Sensitive: age (binary split at the median) -- the only supported value here. The
foreign-worker attribute used in an earlier single-seed run (see
docs/PROJECT_STATUS.md) splits the data ~96%/4%, too skewed to estimate a stable
group-level TPR; age gives a balanced ~50/50 split instead.

Drop the downloaded file into a folder named  GermanData/  inside the project root.
Accepted file name: german.data (raw UCI symbolic format: 20 whitespace-separated
attribute columns + a class column, no header).
Or, if `ucimlrepo` is installed and you have internet, it is fetched automatically
(id=144).
"""
from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GERMAN_DIR = os.path.join(PROJECT_ROOT, "GermanData")

# UCI attribute order (Statlog German Credit Data, symbolic version).
_ATTR_COLS = [f"Attribute{i}" for i in range(1, 21)]
_NUMERIC_ATTRS = {
    "Attribute2", "Attribute5", "Attribute8", "Attribute11",
    "Attribute13", "Attribute16", "Attribute18",
}
_CATEGORICAL_ATTRS = [c for c in _ATTR_COLS if c not in _NUMERIC_ATTRS]
_AGE_COL = "Attribute13"


def _find_local_file() -> str | None:
    if not os.path.isdir(GERMAN_DIR):
        return None
    for pattern in ("*.data", "*.csv"):
        hits = glob.glob(os.path.join(GERMAN_DIR, pattern))
        if hits:
            return sorted(hits)[0]
    return None


def _read_raw() -> pd.DataFrame:
    """Return a DataFrame with columns Attribute1..Attribute20 + class."""
    path = _find_local_file()
    if path is not None:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=_ATTR_COLS + ["class"])
        return df
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=144)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        return df
    except Exception as e:
        raise FileNotFoundError(
            f"No German Credit file found in {GERMAN_DIR} and ucimlrepo fetch failed "
            f"({e}). Put the downloaded german.data into {GERMAN_DIR}."
        )


def prepare_german_for_draft(sensitive: str = "age", test_size: float = 0.2, seed: int = 42) -> tuple:
    """Return (X_train, A_train, Y_train), (X_test, A_test, Y_test) with A,Y in {0,1}."""
    if sensitive != "age":
        raise ValueError(
            f"German Credit only supports sensitive='age' (got {sensitive!r}); other "
            "attributes (e.g. foreign worker) are too skewed for a stable EO-gap estimate."
        )

    df = _read_raw()

    # Label: 1 = bad/high credit risk (UCI encoding: 1 = good, 2 = bad).
    Y = (pd.to_numeric(df["class"], errors="coerce") == 2).astype(np.float64).values

    # Sensitive attribute: binary age, split at the median.
    age = pd.to_numeric(df[_AGE_COL], errors="coerce")
    A = (age >= age.median()).astype(np.float64).values

    # Features = every attribute column except age (passed separately as A).
    feat_cols = [c for c in _ATTR_COLS if c != _AGE_COL]
    Xdf = df[feat_cols].copy()

    cat_cols = [c for c in _CATEGORICAL_ATTRS if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    Xdf[num_cols] = Xdf[num_cols].apply(pd.to_numeric, errors="coerce")
    Xdf[num_cols] = Xdf[num_cols].fillna(Xdf[num_cols].median(numeric_only=True))

    Xdf = pd.get_dummies(Xdf, columns=cat_cols, drop_first=True)
    X = np.nan_to_num(Xdf.astype(np.float64).values, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(
        X, A, Y, test_size=test_size, random_state=seed, stratify=Y
    )
    return (X_tr, A_tr, Y_tr), (X_te, A_te, Y_te)
