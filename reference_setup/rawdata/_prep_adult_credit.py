"""
One-off script: build FairSynData/rawdata/adult.csv and credit.csv from the
raw UCI sources already used by pipeline/load_adult.py and pipeline/load_credit.py.
Not part of FairSynData's method code -- purely data plumbing to register the
two datasets, mirroring the target/sensitive definitions used by our own pipeline
(income>50K, sex male=1/female=0 for Adult; default=1, sex male=1/female=0 for Credit).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UCI_ADULT = os.path.join(ROOT, "UCIAdultdataset")
CREDIT_DIR = os.path.join(ROOT, "CreditData")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]


def build_adult():
    train = pd.read_csv(os.path.join(UCI_ADULT, "adult.data"), header=None,
                         names=ADULT_COLS, na_values="?", skipinitialspace=True)
    test = pd.read_csv(os.path.join(UCI_ADULT, "adult.test"), header=None,
                        names=ADULT_COLS, na_values="?", skipinitialspace=True, skiprows=1)
    df = pd.concat([train, test], ignore_index=True)

    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".").str.replace(" ", "", regex=False)

    # drop race (not used as a feature -- matches pipeline/load_adult.py's drop_cols)
    df = df.drop(columns=["race"])

    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "native-country"]
    for c in cat_cols:
        mode_val = df[c].mode().iloc[0] if df[c].notna().any() else "Unknown"
        df[c] = df[c].fillna(mode_val)
        df[c] = df[c].astype("category").cat.codes.astype(int)

    # capital-gain/capital-loss are ~91%/95% zero -> zero IQR -> FairSynData's RobustScaler
    # (mycodes/datasetsPreprocess.py:scale_data, untouched) leaves them completely unscaled
    # (sklearn falls back to scale_=1 for zero-IQR columns), producing raw values up to 99999
    # next to every other feature's O(1) robust-scaled range. log1p compresses the skew so
    # the column gets a real nonzero IQR and is scaled like everything else.
    df["capital-gain"] = np.log1p(df["capital-gain"])
    df["capital-loss"] = np.log1p(df["capital-loss"])

    # sex/income kept as raw strings; FairSynData/mycodes/datasetsPreprocess.py maps them explicitly
    df["sex"] = df["sex"].astype(str).str.strip()

    # mycodes/trainAndPredict.py:predict_client_test_data reads the sensitive attribute and
    # target POSITIONALLY (iloc[:, -2] / iloc[:, -1]), not by column name -- Law/Dutch's raw
    # CSVs already end with [..., sensitive, target] by construction. Enforce the same physical
    # column order here (matches Adult.py's declared all_columns) so "sex" is truly second-to-
    # last and "income" truly last, not whatever happened to land there.
    feature_cols = [c for c in df.columns if c not in ("sex", "income")]
    df = df[feature_cols + ["sex", "income"]]

    out_path = os.path.join(OUT_DIR, "adult.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}")


def build_credit():
    xls_candidates = [f for f in os.listdir(CREDIT_DIR) if f.lower().endswith((".xls", ".xlsx"))]
    if not xls_candidates:
        raise FileNotFoundError(f"No .xls/.xlsx found in {CREDIT_DIR}")
    path = os.path.join(CREDIT_DIR, sorted(xls_candidates)[0])
    df = pd.read_excel(path, header=1)

    target_col = [c for c in df.columns if "default payment" in c.lower()][0]
    id_cols = [c for c in df.columns if c.lower() in ("id", "unnamed: 0")]
    df = df.drop(columns=id_cols)
    df = df.rename(columns={target_col: "default"})

    # BILL_AMT*/PAY_AMT* are heavy-tailed monetary columns (PAY_AMT2 raw max ~1.68M vs a
    # median of ~2000) -- one client's split hit a NaN gradient during the bilevel outer
    # update at K=20, the same class of scale issue as Adult's capital-gain/loss. BILL_AMT*
    # can be negative (credit balance), so use a signed log1p rather than plain log1p.
    money_cols = [c for c in df.columns if c.startswith("BILL_AMT") or c.startswith("PAY_AMT")]
    for c in money_cols:
        df[c] = np.sign(df[c]) * np.log1p(np.abs(df[c]))

    # SEX raw UCI encoding: 1=male, 2=female. Keep raw; mapped explicitly downstream.

    # Same positional-column fix as Adult (see comment there): predict_client_test_data reads
    # sensitive attr / target via iloc[:, -2] / iloc[:, -1], so enforce [..., SEX, default].
    feature_cols = [c for c in df.columns if c not in ("SEX", "default")]
    df = df[feature_cols + ["SEX", "default"]]

    out_path = os.path.join(OUT_DIR, "credit.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}")


if __name__ == "__main__":
    build_adult()
    build_credit()
