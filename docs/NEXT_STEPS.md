# Next-Steps Roadmap

**Goal:** (1) bring our model's results up to the minimum requirements, (2) run a fair
head-to-head against the reference model (FairSynData) on the *same* datasets, (3) make every
result reproducible, and (4) have a repeatable way to check the code is correct.

## The comparison matrix we are filling

The end goal is this table, every cell filled with mean ± std over seeds:

| Dataset | Our model (draft) | Reference model (FairSynData) |
|---|---|---|
| UCI Adult | have (needs MLP + reseed) | **TODO** (add Adult to FairSynData) |
| Credit (Default of Credit Card) | have (needs MLP + reseed) | **TODO** (add Credit to FairSynData) |
| Law | **TODO** (run our model on Law) | have (reference's own dataset) |
| German Credit (Yanjia) | in progress (balanced attr) | optional |

So two integration jobs remain: **our model on Law**, and **the reference model on Adult + Credit**.

---

## Step 0 — Verification harness first (do this before other changes)

**Status: DONE.** `verify_step3.py` retired; `verify.py` at repo root implements all 6 checks,
self-contained (runs its own fresh subprocesses, no dependency on pre-existing `outputs/*.json`
so it stays valid as the model/loaders change). Run: `python verify.py`.

1. Compiles: `py_compile` on all modules. **PASS**
2. Smoke run finishes with no NaN; baseline + pipeline populated. **PASS**
3. Baseline ≈ `sklearn.LogisticRegression` on the same features (±2%). **PASS** (credit
   0.8077 vs 0.8078; adult 0.8524 vs 0.8524)
4. Regression: `--data adult` with **default flags** reproduces ≈ 0.846 acc / 0.012 EO. **PASS**
   (got acc=0.8463, EO=0.0119 — matches exactly)
5. Ablation sanity: `--no_universum` worsens EO. **PASS**
6. Determinism: same `--seed` → identical numbers. **FAIL — real finding, not a harness bug.**
   The **baseline** is bit-identical across repeated runs (confirmed 3x: exactly
   0.8463239358761747 every time). The **pipeline** (bilevel AL training) is *not*
   deterministic given the same seed: 3 identical-config adult runs gave pipeline accuracy
   0.670 / 0.700 / 0.689 and EO_gap 0.106 / 0.011 / 0.146. `bilevel_al.py` does call
   `torch.manual_seed(seed)` per client-round, but that doesn't pin down PyTorch CPU's
   multi-threaded floating-point reduction order; the outer feature-update loop (implicit CG,
   `J_outer` iterations) is iterative enough that tiny floating-point differences early on
   compound into meaningfully different final states after a few rounds. Not fixed here
   (core-method file, out of scope for Step 0) — **flagging for Step 1/4**: pin
   `torch.set_num_threads(1)` + `torch.use_deterministic_algorithms(True)` in `run_draft.py`,
   or accept and always report mean±std over seeds (which the sweep work already does) rather
   than relying on single-seed reproducibility.

Why first: every later change (MLP, new loaders) is then automatically checked.

## Step 1 — Correct results to minimum requirements (our model)

1. Add an **opt-in small MLP** (`--model mlp`, 1 hidden layer) to replace the linear scorer —
   the main lever to lift pipeline accuracy toward the baseline. Keep the method math intact.
2. Apply calibration per dataset: intercept ON for Credit, OFF for Adult; threshold tuning where
   it helps the minority class.
3. Re-run the main configs with **5–10 seeds**, report mean ± std.

Deliverable: pipeline accuracy within a small band of baseline while EO gap stays low.

## Step 2 — Run OUR model on the Law dataset

1. Add `pipeline/load_law.py` (mirror `load_credit.py`): read `FairSynData/rawdata/law.csv`,
   sensitive = **race** (White/Non-White), label = **pass_bar**.
2. Wire `--data law` into `run_draft.py`.
3. Run our model on Law → fills the "our model / Law" cell (apples-to-apples on the reference's
   own dataset).

## Step 3 — Run the REFERENCE model (FairSynData) on Adult + Credit

1. Add `mydatasets/Adult.py` and `mydatasets/Credit.py` — copy `mydatasets/Law.py` and set:
   columns, `cat_columns`, sensitive attribute (sex), target (income>50K / default).
2. Register both in `mydatasets/DatasetFactory.py`.
3. Put `adult.csv` and `credit.csv` into `FairSynData/rawdata/`.
4. Build client splits with `generate_datasets.py`, then run `main.py` in **NO_DB mode** with
   `syn_2_skip=True` (skip CTGAN) to get accuracy + EO without a database.
5. → fills the "reference model / Adult" and "reference model / Credit" cells.

Note: keep the *same* client split, sensitive attribute, and seeds as our runs so the comparison
is fair.

## Step 4 — Reproducibility

1. Pin versions in `requirements.txt` (numpy, torch, sklearn, pandas, xlrd, matplotlib).
2. Fix seeds everywhere; convert key run settings into a small config so runs are declarative.
3. One master script `scripts/run_all.sh` that regenerates **every** result JSON, table (T1–T5),
   and figure from scratch with a single command. Document it in the README.
4. Add a short "How to get the data" note (download links) since datasets are not in the repo.

## Step 5 — Finalize

- Run the full matrix (both models × all datasets), 5–10 seeds.
- Regenerate tables + figures via `build_tables.py` and `plot_results.py`.
- Update `docs/RESULTS.md` and `docs/PAPER_TABLES.md` with the head-to-head numbers.

---

## Recommended order

1. **Step 0** (verification harness) — safety net.
2. **Step 1** (MLP + calibration + reseed) — fixes the accuracy shortfall.
3. **Step 2** (our model on Law) — small, high value.
4. **Step 3** (reference on Adult + Credit) — the biggest integration job.
5. **Step 4–5** (reproducibility wrapper + final matrix + docs).

## Who does what

- I can **scaffold** the code: the MLP option, `load_law.py`, the FairSynData `Adult.py`/`Credit.py`
  classes, and the expanded `verify.py` — plus hand-off prompts for the VS Code agent.
- The **runs** happen on your machine / cluster (they need PyTorch + the data).
