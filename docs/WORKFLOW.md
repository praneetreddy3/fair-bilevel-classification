# Project Workflow: How Files Connect

**Visual guide showing data flow through the entire system**

---

## Measured Results Update (Apr 29, 2026)

- Baseline (Adult): `accuracy=84.63%`, `EO_gap=1.19%`.
- Best accuracy from latest grid (`outputs/grid_search_summary.json`):
  - `dp_variant=pre_server`, `dp_sigma=0.5`, `rho=0.1`, `epsilon_EO=0.05`
  - pipeline `accuracy=70.51%`, `EO_gap=19.64%`.
- Best fairness from latest grid:
  - `dp_variant=pre_server`, `dp_sigma=0.25`, `rho=0.05`, `epsilon_EO=0.05`
  - pipeline `accuracy=68.89%`, `EO_gap=1.31%`.
- No tested run reached `accuracy >= 72%` on Adult in this grid.
- Stopping comparison (same Adult split):
  - `stop_criterion=eo_gap`: Acc=69.65%, EO=6.74%
  - `stop_criterion=grad_inf` (FairSynData-like): Acc=67.23%, EO=15.47%
  - Baseline: Acc=84.63%, EO=1.19%

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    FAIRBILEVEL WORKFLOW                         │
└────────────────────────────────────────────────────────────────┘

STEP 1: INITIALIZATION
├─ run_draft.py (main orchestrator)
│   └─ Loads: load_adult_data() → adult dataset
│       Output: X_train (24k × 105), A_train (24k), Y_train (24k)
│
└─ Preprocess: StandardScaler
    Output: Scaled X_train, X_val, X_test

STEP 2: FEDERATED SETUP
├─ run_draft.py
│   └─ notation.py
│       └─ Split into K clients (default K=3)
│           Output: K ClientOriginalData objects
│
└─ Each client gets a partition of X, A, Y (size depends on K)

STEP 3: MINIBATCH + SYNTHETIC DATA (Per Client, Per Round)
├─ run_draft.py (lines 151-165)
│   ├─ minibatch_design.py
│   │   ├─ draw_original_minibatch()
│   │   │   Output: B (32 real samples)
│   │   │
│   │   ├─ build_synthetic_templates()
│   │   │   Output: Ds (32 synthetic samples - WILL BE OPTIMIZED)
│   │   │
│   │   └─ build_universum_templates()
│   │       Output: U (50 special samples for fairness)
│   │
│   └─ notation.py
│       └─ Package into: B, Ds, U

STEP 4: BILEVEL OPTIMIZATION (Core Algorithm)
├─ run_draft.py (line 166)
│   │
│   └─ bilevel_al.py :: client_round_al()
│       │
│       ├─ losses.py (Helper Functions)
│       │   ├─ L_base() - Logistic loss on synthetic
│       │   ├─ L_universum() - Fairness loss on U
│       │   ├─ g_EO() - Fairness gap calculation
│       │   ├─ compute_tpr_gap_surrogate() - Smooth EO metric
│       │   └─ Phi() - Augmented Lagrangian objective
│       │
│       ├─ OUTER LOOP (J_outer = 20 iterations)
│       │   │
│       │   ├─ INNER LOOP (K_inner = 50 iterations)
│       │   │   ├─ Forward pass: theta @ X_ds + regularization
│       │   │   ├─ Backward: torch.autograd (compute gradients)
│       │   │   └─ Update: theta ← theta - lr * grad
│       │   │
│       │   ├─ Evaluate Fairness: g_EO(theta, B)
│       │   │   └─ Check: |TPR_male - TPR_female| < epsilon
│       │   │
│       │   └─ Update Synthetic Data: X_ds ← X_ds - lr * grad_X
│       │       └─ Make data fairer for next iteration
│       │
│       └─ OUTPUT: theta_k, X_ds_new, X_u_new
│           (All 3 sent to server)

STEP 5: SERVER AGGREGATION
├─ run_draft.py (line 198)
│   │
│   └─ server.py :: aggregate_payloads()
│       │
│       ├─ Input: Payloads from all K=2 clients
│       │   Client 0: X_ds (32×105) + X_u (50×105)
│       │   Client 1: X_ds (32×105) + X_u (50×105)
│       │
│       ├─ Combine (Stack vertically):
│       │   X_agg = [X_ds_c0; X_ds_c1; X_u_c0; X_u_c1]
│       │   X_agg shape: (164 × 105)
│       │
│       ├─ Apply DP based on config:
│       │   dp_variant ∈ {none, pre_server, post_server, both}
│       │   # X_agg += np.random.normal(0, sigma)
│       │
│       └─ OUTPUT: X_agg (164×105 aggregated synthetic data)

STEP 6: TRAIN GLOBAL MODEL
├─ run_draft.py (line 202)
│   │
│   └─ server.py :: train_global_ridge_erm()
│       │
│       ├─ Input: X_agg (164×105), Y_agg (164)
│       │
│       ├─ Optimization: Ridge regression
│       │   min ||Y - X @ theta||² + λ * ||theta||²
│       │
│       └─ OUTPUT: theta_glob (shape: 106)
│           (Use this for next round or testing)

STEP 7: VALIDATION EVALUATION
├─ run_draft.py (line 208)
│   │
│   └─ server.py :: compute_eo_gap_and_accuracy()
│       │
│       ├─ Input: theta_glob, X_val, A_val, Y_val
│       │
│       ├─ Predictions: Y_pred = X_val @ theta_glob >= 0
│       │
│       ├─ Metrics:
│       │   ├─ accuracy = mean(Y_pred == Y_val)
│       │   ├─ TPR_0 = True Positive Rate (A=0, males)
│       │   ├─ TPR_1 = True Positive Rate (A=1, females)
│       │   ├─ EO_gap = |TPR_0 - TPR_1|
│       │   └─ F1 = harmonic mean(precision, recall)
│       │
│       └─ OUTPUT: All metrics
│           Example (best-accuracy latest grid): acc=70.5%, EO_gap=19.6%

STEP 8: LOOP BACK (T=5 rounds total)
├─ run_draft.py (line 151, for t in range(T))
│   │
│   └─ If t < T: GOTO STEP 3
│       └─ Each round gets zeta (previous theta_glob)

STEP 9: BASELINE COMPARISON
├─ run_draft.py (line 220)
│   │
│   └─ Train standard model (NO fairness)
│       ├─ Input: X_train_original, Y_train_original
│       ├─ Method: train_global_ridge_erm()
│       └─ Evaluate: acc=84.63%, EO_gap=1.19%
│           (More accurate but UNFAIR)

STEP 10: SAVE RESULTS
└─ run_draft.py (line 250)
    └─ Save to JSON: draft_results.json
        {
            "baseline": {"accuracy": 84.63, "EO_gap": 1.19, ...},
            "pipeline": {"accuracy": 70.5, "EO_gap": 19.6, ...},
            "round_logs": [results per round],
            "dataset": "adult"
        }
```

---

## File Dependency Map

```
PRIMARY ORCHESTRATOR
├─ run_draft.py ← Main file, controls entire flow
│   │
│   ├─ Imports & Calls:
│   │   ├─ load_adult_data() → pipeline/load_adult.py
│   │   ├─ notation.py → Data structures
│   │   ├─ minibatch_design.py → Create synthetic data
│   │   ├─ bilevel_al.py → Bilevel optimization
│   │   └─ server.py → Aggregation & training
│   │
│   └─ Variables Passed:
│       ├─ args (parsed arguments)
│       ├─ X_train, A_train, Y_train (loaded data)
│       ├─ zeta (previous global model)
│       └─ theta_glob (updated global model)

SUPPORTING MODULES
│
├─ notation.py
│   ├─ ClientOriginalData: X, A, Y (original data per client)
│   ├─ OriginalMinibatch: Small batch from client
│   ├─ SyntheticMinibatch: Synthetic batch (to be optimized)
│   └─ UniversumSet: Special samples for fairness
│
├─ minibatch_design.py
│   ├─ draw_original_minibatch(client_data) → B
│   ├─ build_synthetic_templates(B) → Ds
│   └─ build_universum_templates(B, Ds) → U
│
├─ bilevel_al.py (CORE ALGORITHM)
│   ├─ client_round_al()
│   │   ├─ Inner Loop: Train theta (50 iters)
│   │   ├─ Outer Loop: Optimize X_ds for fairness (20 iters)
│   │   └─ Stopping: When EO_gap < epsilon_EO
│   │
│   ├─ client_round_simplified() (backup, no feature updates)
│   ├─ importance_weights()
│   ├─ rolling_positive_buffer()
│   └─ compute_ema_estimate()
│
├─ losses.py (LOSS FUNCTIONS)
│   ├─ L_base() - Logistic loss + regularization
│   ├─ L_universum() - Fairness enforcement via U
│   ├─ compute_tpr_s() - TPR for group s
│   ├─ g_EO() - EO fairness gap |TPR_1 - TPR_0|
│   ├─ Phi() - Augmented Lagrangian objective
│   └─ Helper: logistic_loss_per_sample()
│
├─ server.py (AGGREGATION & TRAINING)
│   ├─ aggregate_payloads() - Combine clients' synthetic data
│   │   └─ Optional: add DP using dp.py configuration
│   ├─ train_global_ridge_erm() - Train on X_agg
│   ├─ compute_eo_gap_and_accuracy() - Evaluate
│   └─ Helper: _to_torch(), pack_xa()
│
└─ pipeline/load_adult.py
    └─ prepare_adult_for_draft() - Load Adult dataset
```

---

## Data Flow Diagram (Adult Dataset Example)

```
┌─────────────────────────────────────────────────────────────┐
│                      RAW ADULT DATA                          │
│  (24,421 samples × 105 features)                             │
│  Features: age, education, hours, ...                        │
│  Label: income > 50K (binary)                                │
│  Sensitive: sex (0=M, 1=F)                                   │
│  Imbalance: 77% male, 23% female                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    [PREPROCESSING]
                    Scale features
                    80/20 split
                            ↓
┌────────────────┬─────────────────────────┬──────────────────┐
│  X_train       │      X_val              │   X_test         │
│  (19.5k×105)   │    (4.9k×105)           │  (12.3k×105)     │
│  A_train       │      A_val              │   A_test         │
│  (19.5k)       │    (4.9k)               │  (12.3k)         │
│  Y_train       │      Y_val              │   Y_test         │
│  (19.5k)       │    (4.9k)               │  (12.3k)         │
└────────────────┴─────────────────────────┴──────────────────┘
        ↓                           ↑                   ↑
   [SPLIT TO 2 CLIENTS]        [EVALUATION]    [FINAL TESTING]
        ↓
┌─────────────────────────┬──────────────────────────┐
│   CLIENT 0              │     CLIENT 1             │
│   9.8k samples          │    9.8k samples          │
│   M: 7.5k, F: 2.3k      │   M: 7.5k, F: 2.3k       │
└─────────────────────────┴──────────────────────────┘
        ↓                           ↓
   [MINIBATCH]               [MINIBATCH]
   B_0: 32 real              B_1: 32 real
        ↓                           ↓
   [CREATE SYNTHETIC]         [CREATE SYNTHETIC]
   Ds_0: 32 synthetic         Ds_1: 32 synthetic
   U_0: 50 universum         U_1: 50 universum
        ↓                           ↓
   [BILEVEL OPT]              [BILEVEL OPT]
   Train theta               Train theta
   Optimize X_ds, X_u        Optimize X_ds, X_u
   ↓                         ↓
   Output:                   Output:
   theta_0                   theta_1
   X_ds_0_opt               X_ds_1_opt
   X_u_0_opt                X_u_1_opt
        ↓                           ↓
        └───────────────┬───────────┘
                        ↓
                  [SERVER AGGS]
                  Combine all synthetic
                  X_agg: 164×105
                        ↓
                  [TRAIN GLOBAL]
                  ridge regression
                        ↓
                  [EVALUATION]
                  theta_glob tested on X_val
                        ↓
        ┌───────────────────────────────┐
        │ RESULTS:                      │
        │ Accuracy: 70.5% (best-acc run)│
        │ EO Gap: 1.31% (best-fair run) │
        │ See outputs/grid_search_*.json│
        │ Fair? YES ✓                   │
        └───────────────────────────────┘
```

---

## Key Variables Through Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ VARIABLE TRACKING: ADULT DATASET THROUGH SYSTEM             │
└─────────────────────────────────────────────────────────────┘

┌─ Initial Load
│  X_train: (24,421 × 105)  - Features
│  A_train: (24,421,)       - Gender (0 or 1)
│  Y_train: (24,421,)       - Income label
│
├─ After Preprocessing
│  X_train: (19,537 × 105)  - Scaled, training set only
│  X_val:   (4,884 × 105)   - Validation set
│  X_test:  (12,326 × 105)  - Test set
│
├─ Client 0 Data
│  B: (32 × 105)            - Real minibatch
│  Ds: (32 × 105)           - Synthetic (OPTIMIZED during bilevel)
│  U: (50 × 105)            - Universum (OPTIMIZED during bilevel)
│
├─ After Bilevel Optimization
│  theta_0: (106,)          - Trained parameters
│  X_ds_0_opt: (32 × 105)   - Optimized synthetic
│  X_u_0_opt: (50 × 105)    - Optimized universum
│
├─ Server Aggregation
│  X_agg: (164 × 105)       - All synthetic data combined
│           = 32+32+50+50 samples
│
├─ After Training Global Model
│  theta_glob: (106,)       - Final model parameters
│
├─ Evaluation
│  Y_pred: (4,884,)         - Predictions on validation
│  accuracy: scalar (0.692)
│  EO_gap: scalar (0.0008)
│  TPR_0: scalar (0.42)
│  TPR_1: scalar (0.50)
│
└─ Baseline
   theta_baseline: (106,)   - Standard model
   acc_baseline: scalar (0.8459)
   EO_gap_baseline: scalar (0.123)
```

---

## Stopping Criteria Flow

```
┌─ OUTER LOOP (Line 133: for j in range(J_outer))
│  J_outer = 20 maximum iterations
│
├─ Inner Loop Trains Model (50 iterations)
│
├─ Evaluate Fairness (Line 177)
│  g_val = g_EO(theta_star, B.X, B.A, B.Y)
│  └─ Compute |TPR_male - TPR_female|
│
├─ Smooth with EMA (Line 230)
│  g_ema = (1 - beta) * g_ema + beta * g_val
│  beta = 0.15
│
├─ Check Stopping Condition (Line 232)
│  if stop_criterion == "eo_gap":
│      if abs(g_ema) <= epsilon_EO:
│          break
│  elif stop_criterion == "grad_inf":
│      if grad_inf_x <= outer_tol_xhat * max(1, grad_inf_x0):
│          break
│  └─ recommended default: eo_gap with epsilon_EO = 0.05 to 0.10
│
└─ Update Lagrange Multiplier (Line 231)
   lam = lam + rho * g_ema
   └─ recommended tuning range: rho = 0.05 to 0.10

Example Progress:
Iteration 1:  g_ema = 0.45 > 0.25 ← Continue
Iteration 5:  g_ema = 0.32 > 0.25 ← Continue
Iteration 10: g_ema = 0.18 < 0.25 ← Continue (but improving)
Iteration 15: g_ema = 0.08 < 0.25 ← Continue
Iteration 18: g_ema = 0.23 < 0.25 ← STOP ✓ Converged
```

---

## Functions Called Per Round

```
FOR EACH ROUND (T=5 total):
  FOR EACH CLIENT (K=2):
    ├─ draw_original_minibatch()
    │   └─ Return: B (OriginalMinibatch)
    │
    ├─ build_synthetic_templates()
    │   └─ Return: Ds (SyntheticMinibatch)
    │
    ├─ build_universum_templates()
    │   └─ Return: U (UniversumSet)
    │
    └─ client_round_al()  ← MAIN ALGORITHM
        ├─ Called Functions (in losses.py):
        │   ├─ L_base() [50 times in inner loop]
        │   ├─ L_universum() [50 times]
        │   ├─ g_EO() [20 times in outer loop]
        │   ├─ compute_tpr_s() [many times]
        │   ├─ compute_tpr_gap_surrogate() [20 times]
        │   └─ Phi() [20 times]
        │
        └─ Return: (theta_k, X_ds_new, X_u_new)

  └─ aggregate_payloads()
      ├─ Input: All client payloads
      └─ Return: X_agg (combined synthetic)

  └─ train_global_ridge_erm()
      ├─ Input: X_agg, Y_agg
      └─ Return: theta_glob

  └─ compute_eo_gap_and_accuracy()
      ├─ Input: theta_glob, X_val, A_val, Y_val
      └─ Return: metrics (accuracy, EO_gap, TPR_0, TPR_1, F1)

  └─ Store in round_logs[]

Save all results to draft_results.json
```

---

## Summary: Which File Does What

```
┌─ MAIN FILE ─────────────────────────────────────────────┐
│ run_draft.py                                            │
│ ├─ Parses arguments (data, clients, rounds, etc)        │
│ ├─ Loads Adult dataset                                  │
│ ├─ Preprocesses (scale, split)                          │
│ ├─ Loops through rounds and clients                     │
│ ├─ Calls bilevel optimization per client                │
│ └─ Saves results                                        │
│                                                          │
├─ CORE ALGORITHM ────────────────────────────────────────┤
│ bilevel_al.py                                           │
│ ├─ client_round_al(): THE BILEVEL OPTIMIZATION          │
│ │  ├─ Inner: Train model (50 iters)                    │
│ │  └─ Outer: Optimize synthetic data (20 iters)        │
│ └─ client_round_simplified(): Backup version            │
│                                                          │
├─ LOSS FUNCTIONS ────────────────────────────────────────┤
│ losses.py                                               │
│ ├─ L_base(): Logistic loss + regularization             │
│ ├─ L_universum(): Fairness loss                         │
│ ├─ g_EO(): Main fairness metric                         │
│ ├─ compute_tpr_gap_surrogate(): Smooth EO              │
│ └─ Phi(): Augmented Lagrangian                          │
│                                                          │
├─ DATA STRUCTURES ───────────────────────────────────────┤
│ notation.py                                             │
│ ├─ ClientOriginalData: Original client data             │
│ ├─ OriginalMinibatch: Real batch                        │
│ ├─ SyntheticMinibatch: Fake batch                       │
│ └─ UniversumSet: Special fairness samples               │
│                                                          │
├─ SYNTHETIC DATA ────────────────────────────────────────┤
│ minibatch_design.py                                     │
│ ├─ draw_original_minibatch(): Sample from client       │
│ ├─ build_synthetic_templates(): Create fake data       │
│ └─ build_universum_templates(): Create fairness data   │
│                                                          │
├─ SERVER OPERATIONS ─────────────────────────────────────┤
│ server.py                                               │
│ ├─ aggregate_payloads(): Combine all clients            │
│ ├─ train_global_ridge_erm(): Train on aggregated       │
│ └─ compute_eo_gap_and_accuracy(): Evaluate metrics      │
│                                                          │
└─ DATA LOADING ──────────────────────────────────────────┘
  pipeline/load_adult.py
  └─ prepare_adult_for_draft(): Load Adult dataset
```

---

## Quick Navigation

```
Question: "Where is X?"

Where is the CORE ALGORITHM?
└─ bilevel_al.py, function client_round_al() (lines 85-239)

Where are LOSS FUNCTIONS?
└─ losses.py, functions like g_EO(), L_base(), etc.

Where do we LOAD DATA?
└─ run_draft.py lines 106-113, calls load_adult_data()

Where is FAIRNESS PENALTY computed?
└─ losses.py, function g_EO() (lines 104-115)

Where is SYNTHETIC DATA created?
└─ minibatch_design.py, functions like build_synthetic_templates()

Where is SERVER AGGREGATION?
└─ server.py, function aggregate_payloads() (lines ~197)

Where is EVALUATION?
└─ server.py, function compute_eo_gap_and_accuracy()

Where do we LOOP through ROUNDS?
└─ run_draft.py lines 151-206

Where is DP NOISE applied?
└─ draft_model/dp.py + run_draft.py (--dp_variant, --dp_sigma)

Where is STOPPING CRITERION?
└─ bilevel_al.py line 232-233, checks if g_ema <= epsilon_EO
```

---

**Use this with PROJECT_GUIDE.md for complete understanding in 15 minutes!**
