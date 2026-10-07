"""
Re-run the pipeline with the EMA stopping-rule fix (--ema_init first). Needs PyTorch.

Background: with the original EMA initialisation (0), the fairness estimate after the first outer
step is only beta*g ~ 0.05 < epsilon_EO = 0.1, so the outer loop stops after ONE tiny feature step
and the AL stage barely acts (outputs/reuse_summary.md: rho=0 control ~ ours on Credit/Law/COMPAS).
--ema_init first starts the EMA at the first observed gap, so the loop runs until the gap is
really within epsilon_EO (or J = 20 steps).

  python scripts/run_fix.py --stage 1 --jobs 4   # 40 runs: reported (rho, eps) + rho=0 control
  python scripts/run_fix.py --stage 2 --jobs 4   # full validation grid (adds the other configs)
  python scripts/run_fix.py --stage 3 --jobs 4   # sign-corrected feature step (--hg_sign correct), rho grid + control
  python scripts/run_fix.py --stage 5 --variant AB --jobs 4   # a fix variant chosen with scripts/diag_fix.py
  python scripts/run_fix.py --stage 6 --jobs 4   # final: selected configs + controls + Universum toggle, seeds 1-10
  python scripts/run_fix.py --stage 1 --dry_run  # print commands only

Runs whose result file exists are skipped. Settings other than the ones named are the reported ones
(scripts/run_final.sh, docs/RESULTS.md). The new runs are LONGER than before because the outer loop
now actually iterates (up to 20 outer steps per client per round).

Files: outputs/draft_results_<ds>_fix_rho<rho>_eps<eps>_seed<s>.json
       outputs/draft_results_<ds>_fixctl_seed<s>.json          (--fairness_off, same ema fix)
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SEEDS = [1, 2, 3, 4, 5]
COMMON = ["--num_clients", "5", "--rounds", "8", "--K_inner", "100", "--deterministic", "true",
          "--dp_variant", "none", "--ema_init", "first"]
CFG = {  # per-dataset reported settings WITHOUT rho/epsilon_EO
    "credit": ["--sensitive", "sex", "--add_intercept", "true", "--tune_threshold", "true"],
    "adult": ["--sensitive", "sex", "--add_intercept", "false", "--tune_threshold", "false"],
    "law": ["--sensitive", "race", "--add_intercept", "true", "--tune_threshold", "true"],
    "compas": ["--sensitive", "race", "--add_intercept", "true", "--tune_threshold", "true", "--no_universum"],
}
REPORTED_RHO = {"credit": 0.05, "adult": 0.1, "law": 0.1, "compas": 0.01}
GRID_RHO = [0.05, 0.1, 0.5, 2.0]
GRID_EPS = [0.1, 0.02]


VARIANT_FLAGS = {  # stage 5 fix variants (see scripts/diag_fix.py)
    "A": ["--fair_set", "client"],
    "B": ["--eo_surrogate", "loss_gap"],
    "AB": ["--fair_set", "client", "--eo_surrogate", "loss_gap"],
    "AC": ["--fair_set", "client", "--fair_grad_norm", "true"],
    "ABC": ["--fair_set", "client", "--eo_surrogate", "loss_gap", "--fair_grad_norm", "true"],
    "SA": ["--hg_sign", "correct", "--step_clip", "0.05", "--fair_set", "client"],
    "SAC": ["--hg_sign", "correct", "--step_clip", "0.05", "--fair_set", "client", "--fair_grad_norm", "true"],
    "SAB": ["--hg_sign", "correct", "--step_clip", "0.05", "--fair_set", "client", "--eo_surrogate", "loss_gap"],
}
STAGE5_RHO = [1.0, 10.0, 50.0]


FINAL2 = {  # stage 6: configurations selected on validation in stage 5 (see outputs/ema_vSA*_summary.md)
    "credit": ("SAC", "10.0"), "adult": ("SAC", "10.0"), "law": ("SA", "50.0"), "compas": ("SA", "50.0"),
}


def jobs_for(stage, variant="AB"):
    jobs = []
    if stage == 6:   # final runs: selected config + its rho=0 control, seeds 1-10; Universum toggled, seeds 1-10
        for ds, (v, rho) in FINAL2.items():
            vf, cfg = VARIANT_FLAGS[v], CFG[ds]
            tog = [a for a in cfg if a != "--no_universum"] if "--no_universum" in cfg else cfg + ["--no_universum"]
            for s in range(1, 11):
                base = ["--seed", str(s), *COMMON, *vf, "--epsilon_EO", "0.02", "--data", ds]
                jobs.append((f"draft_results_{ds}_final2_seed{s}.json", base + cfg + ["--rho", rho]))
                jobs.append((f"draft_results_{ds}_final2ctl_seed{s}.json", base + cfg + ["--rho", "0.0", "--fairness_off"]))
                if ds != "law":   # Law forms no Universum points, so toggling it changes nothing
                    jobs.append((f"draft_results_{ds}_final2univ_seed{s}.json", base + tog + ["--rho", rho]))
        return jobs
    if stage == 5:   # one fix variant: its own rho = 0 control + STAGE5_RHO, eps 0.02, all datasets
        vf = VARIANT_FLAGS[variant]
        for ds, cfg in CFG.items():
            for s in SEEDS:
                base = ["--data", ds, *cfg, "--seed", str(s), *COMMON, *vf, "--epsilon_EO", "0.02"]
                jobs.append((f"draft_results_{ds}_v{variant}ctl_seed{s}.json", base + ["--rho", "0.0", "--fairness_off"]))
                for rho in STAGE5_RHO:
                    jobs.append((f"draft_results_{ds}_v{variant}_rho{rho}_eps0.02_seed{s}.json", base + ["--rho", str(rho)]))
        return jobs
    if stage == 4:   # smooth surrogate + big rho + bounded step, Law and Adult only
        for ds in ("law", "adult"):
            for s in SEEDS:
                base = ["--data", ds, *CFG[ds], "--seed", str(s), *COMMON, "--tpr_alpha", "2.0",
                        "--step_clip", "0.05", "--epsilon_EO", "0.02"]
                jobs.append((f"draft_results_{ds}_smoothctl_seed{s}.json", base + ["--rho", "0.0", "--fairness_off"]))
                for rho in (1.0, 10.0, 50.0):
                    jobs.append((f"draft_results_{ds}_smooth_rho{rho}_eps0.02_seed{s}.json", base + ["--rho", str(rho)]))
        return jobs
    if stage == 3:   # sign-corrected feature step (--hg_sign correct)
        for ds, cfg in CFG.items():
            for s in SEEDS:
                jobs.append((f"draft_results_{ds}_sfixctl_seed{s}.json",
                             ["--data", ds, *cfg, "--seed", str(s), *COMMON, "--hg_sign", "correct",
                              "--rho", "0.0", "--epsilon_EO", "0.1", "--fairness_off"]))
                for rho in GRID_RHO:
                    jobs.append((f"draft_results_{ds}_sfix_rho{rho}_eps0.1_seed{s}.json",
                                 ["--data", ds, *cfg, "--seed", str(s), *COMMON, "--hg_sign", "correct",
                                  "--rho", str(rho), "--epsilon_EO", "0.1"]))
        return jobs
    for ds, cfg in CFG.items():
        for s in SEEDS:
            jobs.append((f"draft_results_{ds}_fixctl_seed{s}.json",
                         ["--data", ds, *cfg, "--seed", str(s), *COMMON, "--rho", "0.0",
                          "--epsilon_EO", "0.1", "--fairness_off"]))
            if stage == 1:
                grid = [(REPORTED_RHO[ds], 0.1)]
            else:
                grid = [(REPORTED_RHO[ds], 0.1)] + [(r, e) for r in GRID_RHO for e in GRID_EPS]
            for rho, eps in dict.fromkeys(grid):
                jobs.append((f"draft_results_{ds}_fix_rho{rho}_eps{eps}_seed{s}.json",
                             ["--data", ds, *cfg, "--seed", str(s), *COMMON, "--rho", str(rho),
                              "--epsilon_EO", str(eps)]))
    return jobs


def run_one(job, dry):
    fname, args = job
    cmd = [sys.executable, "-W", "ignore", "-m", "draft_model.run_draft", *args, "--results_file", fname]
    if os.path.isfile(os.path.join(OUT, fname)):
        return f"skip  {fname}"
    if dry:
        return "DRY   " + " ".join(cmd)
    r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        return f"FAIL  {fname}\n{r.stderr[-600:]}"
    return f"done  {fname}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5, 6], default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--variant", default="AB", choices=list(VARIANT_FLAGS), help="stage 5 only")
    a = ap.parse_args()
    jobs = jobs_for(a.stage, a.variant)
    print(f"{len(jobs)} runs ({a.jobs} in parallel)")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for msg in ex.map(lambda j: run_one(j, a.dry_run), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
