# COMPAS — the 4th real-life application (for the professor meeting)

## What this dataset is and why it was added

**COMPAS** ("Correctional Offender Management Profiling for Alternative Sanctions") is a
criminal-risk-assessment tool used by US courts to score defendants' likelihood of
reoffending. In 2016, ProPublica ran an investigation ("Machine Bias") showing the tool
flagged Black defendants as high-risk at a much higher rate than white defendants, even
after controlling for actual reoffense outcomes — it is now the single most-cited real-world
example of algorithmic bias in the fairness-in-ML literature, and the closest thing that
field has to a "standard benchmark."

It was added as our project's fourth real-life application, alongside Credit Risk (main
application), UCI Adult, and Law School, specifically because it is:
- **Independent of the reference paper** — FairSynData (the paper we compare against) never
  used this dataset, so it demonstrates our method generalizes beyond the reference's own
  benchmarks, not just reproduces them.
- **A criminal-justice domain**, not finance/education — broadens the paper's claimed scope
  from "credit-risk-style tabular data" to a genuinely different application area.
- **A dataset with well-documented, real unfairness** — unlike Credit or Adult (where the
  calibrated baseline turns out to already be close to fair), COMPAS's baseline is
  *genuinely* biased, which is exactly the condition needed to show the fairness method
  actually doing something.

Source data: ProPublica's original `compas-scores-two-years.csv` (7,214 Broward County, FL
defendants, 2013–2014, with 2-year re-arrest outcomes) — the same file underlying essentially
every COMPAS fairness paper, obtained via the `responsibly` Python package (which bundles the
identical ProPublica file) rather than a re-derived or synthetic substitute.

## Task, features, and sensitive attribute

- **Label (what the model predicts):** `two_year_recid` — 1 if the person was re-arrested
  within two years, 0 if not.
- **Sensitive attribute (kept separate from the features, never used to predict):** `race`,
  encoded Caucasian = 1, African-American = 0. Filtered to just these two groups (5,278 of
  the 7,214 rows) — the standard restriction used by ProPublica's own analysis and virtually
  every downstream fairness paper, because the remaining race categories are too small to
  measure a stable per-group true-positive rate.
- **Input features (8, after standard filtering):** `sex`, `age`, `age_cat` (age bracket),
  `priors_count` (number of prior offenses), `c_charge_degree` (felony/misdemeanor),
  `juv_fel_count`, `juv_misd_count`, `juv_other_count` (juvenile offense counts). The three
  categorical ones (`sex`, `age_cat`, `c_charge_degree`) are one-hot encoded; the rest are
  numeric as-is.
- **Row filtering before training** (standard ProPublica/AIF360 preprocessing, not our
  invention): drop rows where the arrest date is more than 30 days from the COMPAS screening
  date, where the recidivism outcome is unknown, where the charge is a minor traffic offense,
  or where no COMPAS score was produced.

## How the model's settings were chosen (not guessed)

Run via `python scripts/compas_sweep.py`: a grid of 5 fairness-penalty strengths (`rho`) x
Universum on/off x 5 random seeds (50 total runs). The winning configuration is the one that
maximizes **validation-set** accuracy among configs whose **validation-set** EO gap is ≤ 0.1
— test-set numbers are never looked at during selection. This is the identical selection rule
already used for Credit/Adult/Law (`scripts/fair_comparison.py`), so COMPAS is held to the
same standard, not given a shortcut.

**Winner:** `rho = 0.01`, Universum **off**. (Turning Universum off winning here is a real,
reportable finding — see verdict below — not a workaround.)

## Results (5 seeds, mean ± std)

| Model | Accuracy | EO gap | DP gap | EOD gap |
|---|---|---|---|---|
| Baseline (no fairness step) | 0.640 ± 0.003 | 0.276 ± 0.013 | 0.248 ± 0.015 | 0.282 ± 0.011 |
| Our pipeline | 0.632 ± 0.015 | 0.166 ± 0.060 | 0.146 ± 0.049 | 0.178 ± 0.056 |

**Verdict:** this is the cleanest fairness result across all four datasets in the project.
The baseline is genuinely unfair (EO gap 0.276 — African-American defendants get flagged at a
much higher true-positive rate than Caucasian defendants, matching ProPublica's original
finding). The pipeline cuts that gap by **~40%** (EO gap 0.276 → 0.166), with DP gap down
~41% and EOD gap down ~37%, at a cost of well under one accuracy point (64.0% → 63.2%). On
Credit and Adult the calibrated baseline was already close to fair, so the method had little
room to help; COMPAS is where it clearly does.

The Universum mechanism (the pseudo-positive points designed to help the underrepresented
group) *lost* the validation selection here — it made both accuracy and fairness worse at
every `rho` tried. Combined with Adult (where Universum helps) and Credit (where it hurts),
COMPAS is a third, independent data point confirming Universum's effect is genuinely
dataset-dependent, not a mechanism that helps everywhere.

## Code files to show the professor

1. **`pipeline/load_compas.py`** — the new dataset loader. This is the actual new code: reads
   the raw CSV, applies the standard filters, builds the feature matrix, encodes the sensitive
   attribute. Walk through this if asked "what did you actually add."
2. **`scripts/compas_sweep.py`** — the validation-only settings search described above. Shows
   the methodology is legitimate (never touches test data) and consistent with how the other
   three datasets were tuned.
3. **`docs/RESULTS.md`** and **`docs/PROJECT_STATUS.md`** — the honest, full write-up of all
   four datasets' results, including this one, in the project's existing reporting style.
4. **`outputs/final_bars.png`** and **`outputs/final_pareto.png`** — the clean, paper-ready
   figures (one point per dataset, mean ± std) with COMPAS included as the 4th group/color.
5. **`outputs/tables/compas_sweep.csv`** — the full 10-config sweep grid, if asked to justify
   why `rho=0.01`/no-Universum specifically was picked.

The one-line change to `draft_model/run_draft.py` (adding `compas` as a `--data` option) isn't
worth walking through — it's three lines of wiring, not new logic.
