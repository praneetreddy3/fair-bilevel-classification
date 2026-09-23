# Changes to apply to a `FairSynData/` checkout

`FairSynData/` is the reference implementation this project builds on. It carries no
licence, so its source is **not redistributed here** — neither the original files nor
modified copies of them. This file instead specifies every change needed to add
`adult` and `credit` support, so the setup is fully reproducible from a clean checkout.

New files that are entirely this project's own work still live in this folder and can
be copied in directly:

| File here | Copy to |
|---|---|
| `mydatasets/Adult.py` | `FairSynData/mydatasets/Adult.py` |
| `mydatasets/Credit.py` | `FairSynData/mydatasets/Credit.py` |
| `rawdata/_prep_adult_credit.py` | `FairSynData/rawdata/_prep_adult_credit.py` |

The three files below are modifications to existing FairSynData source. Apply them by
hand — each change is small and fully specified.

---

## 1. `mycodes/myParams.py`

One value. In `AlgorithmParams.__init__`, change `self.K` from `3` to `20`:

```python
self.K = 20  # bilevel prob: max iter for updating outer problem
             # (raised from the smoke-test default of 3)
```

The default of 3 outer iterations is too few for the fairness-constrained synthetic
data to converge. See `docs/COMPARISON.md`.

---

## 2. `mydatasets/DatasetFactory.py`

Register the two new datasets alongside the existing ones.

Add to the imports at the top:

```python
from mydatasets.Adult import Adult
from mydatasets.Credit import Credit
```

Then extend the list returned by `get_raw_datasets`:

```python
def get_raw_datasets(isFL=False):
    return [
        Law(isFL), Dutch(isFL), Adult(isFL), Credit(isFL)
    ]
```

No other change.

---

## 3. `mycodes/datasetsPreprocess.py`

Four additions, each following the pattern already used for `law` and `dutch`.

**a. Dispatch in `preprocess_data`** — add two branches after the `dutch` one:

```python
    elif data_name == 'adult':
        return preprocess_adult(df)
    elif data_name == 'credit':
        return preprocess_credit(df)
```

**b. Two new preprocessing functions** — add after `preprocess_dutch`:

```python
def preprocess_adult(df):
    """
    Preprocess Adult: convert 'sex' to 0 (Female) and 1 (Male), and 'income' to
    -1 (<=50K) and 1 (>50K), matching the -1/1 label convention used elsewhere.
    """
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['income'] = df['income'].map({'<=50K': -1, '>50K': 1})
    return df


def preprocess_credit(df):
    """
    Preprocess Credit: convert 'SEX' to 0 (Female) and 1 (Male) (raw UCI encoding
    is 1=male, 2=female), and 'default' to -1 (no default) and 1 (default).
    """
    df['SEX'] = df['SEX'].map({1: 1, 2: 0})
    df['default'] = df['default'].map({0: -1, 1: 1})
    return df
```

**c. Sensitive attribute and target in `split_data_to_tensors`** — add two branches:

```python
    elif data_name == 'adult':
        sensitive_attributes = 'sex'
        target = 'income'
    elif data_name == 'credit':
        sensitive_attributes = 'SEX'
        target = 'default'
```

**d. Columns excluded from scaling in `scale_data`** — add two branches:

```python
    elif data_name == 'adult':
        columns_to_exclude = ["sex"]
    elif data_name == 'credit':
        columns_to_exclude = ["SEX"]
```

**e. Non-numeric guard in `detect_outliers`** — add these two lines as the first
statement inside the `for column in df.columns[:-1]:` loop:

```python
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
```

`detect_outliers` is diagnostic only — its output is logged and never used to filter
data — but it raises on the non-numeric columns present in Adult.

---

## Data preparation

`rawdata/_prep_adult_credit.py` builds `rawdata/adult.csv` and `rawdata/credit.csv`
from `UCIAdultdataset/` and `CreditData/`. It includes two fixes found during the
convergence investigation (`docs/COMPARISON.md`):

- `log1p` on heavy-tailed columns (`capital-gain`, `capital-loss`, `BILL_AMT*`, `PAY_AMT*`)
- explicit column reordering so the sensitive attribute is second-to-last and the
  target last — FairSynData's `trainAndPredict.py` reads them positionally via
  `iloc[:, -2]` / `iloc[:, -1]`

Neither is a change to the optimisation algorithm.
