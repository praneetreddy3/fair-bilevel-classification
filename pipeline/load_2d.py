"""
Load 2D example data for the draft pipeline and FairSynData.
Format: x1, x2, s, y (-1/1), client
"""
import os
import numpy as np
import pandas as pd


def load_2d_for_draft(project_root: str, csv_path: str = None, train_frac: float = 0.8, seed: int = 42):
    """
    Load 2D data for draft pipeline.
    Returns (X_train, A_train, Y_train), (X_test, A_test, Y_test) with Y in {0,1}.
    """
    if csv_path is None:
        csv_path = os.path.join(project_root, "outputs", "2d_data.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"2D data not found: {csv_path}. Run run_2d_example.py first."
        )
    df = pd.read_csv(csv_path)
    X = df[["x1", "x2"]].values.astype(np.float64)
    A = df["s"].values.astype(np.int64)
    y_raw = df["y"].values
    Y = (y_raw == 1).astype(np.float64)  # -1/1 -> 0/1

    n = len(Y)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, int(n * train_frac))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, A_train, Y_train = X[train_idx], A[train_idx], Y[train_idx]
    X_test, A_test, Y_test = X[test_idx], A[test_idx], Y[test_idx]
    return (X_train, A_train, Y_train), (X_test, A_test, Y_test)


def convert_2d_to_law_format(csv_path: str, out_train: str, out_test: str, train_frac: float = 0.8, seed: int = 42):
    """
    Convert 2D data to Law format (decile1b,...,tier, race, pass_bar) for FairSynData.
    Writes train and test CSVs.
    """
    cols = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa", "fulltime", "fam_inc", "male", "tier", "race", "pass_bar"]
    df = pd.read_csv(csv_path)
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, int(n * train_frac))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def to_law_df(indices):
        x1 = df["x1"].values[indices]
        x2 = df["x2"].values[indices]
        race = df["s"].values[indices].astype(np.float64)
        pass_bar = df["y"].values[indices].astype(np.float64)  # -1 or 1
        # 10 features: x1, x2, then 8 zeros
        zeros = np.zeros((len(indices), 8))
        X = np.column_stack([x1, x2, zeros])
        out = pd.DataFrame(X, columns=cols[:-2])
        out["race"] = race
        out["pass_bar"] = pass_bar
        return out

    train_df = to_law_df(train_idx)
    test_df = to_law_df(test_idx)
    os.makedirs(os.path.dirname(out_train) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_test) or ".", exist_ok=True)
    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)
    return out_train, out_test
