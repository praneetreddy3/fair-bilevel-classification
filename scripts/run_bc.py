"""
Driver for the two follow-up experiments (needs PyTorch; run from the repo root):

  python scripts/run_bc.py --exp B --jobs 4      # synthetic batch size sweep   (m = 64, 128)
  python scripts/run_bc.py --exp C --jobs 4      # reuse experiment pool + control
  python scripts/run_bc.py --exp all --jobs 4

Use --dry_run to print the commands without running them. Runs that already have a result file are
skipped, so an interrupted run can simply be restarted.

Per-dataset settings are exactly the reported ones (scripts/run_final.sh, docs/RESULTS.md); only
the options named below change.

  B: --syn_size m  -> outputs/draft_results_<ds>_syn<m>_seed<s>.json
     (m = 32 is the reported configuration: the existing *_final_* files.)
  C: --save_synthetic true -> outputs/draft_results_<ds>_reuse_seed<s>.json (+ _synthetic.npz)
     --fairness_off true --save_synthetic true (control, rho = 0)
                           -> outputs/draft_results_<ds>_reusectl_seed<s>.json (+ _synthetic.npz)
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
          "--dp_variant", "none", "--epsilon_EO", "0.1"]

# Reported per-dataset settings (docs/RESULTS.md).
CFG = {
    "credit": ["--sensitive", "sex", "--add_intercept", "true", "--tune_threshold", "true", "--rho", "0.05"],
    "adult": ["--sensitive", "sex", "--add_intercept", "false", "--tune_threshold", "false", "--rho", "0.1"],
    "law": ["--sensitive", "race", "--add_intercept", "true", "--tune_threshold", "true", "--rho", "0.1"],
    "compas": ["--sensitive", "race", "--add_intercept", "true", "--tune_threshold", "true",
               "--rho", "0.01", "--no_universum"],
}


def jobs_for(exp):
    jobs = []
    for ds, cfg in CFG.items():
        for s in SEEDS:
            if exp in ("B", "all"):
                for m in (64, 128):
                    jobs.append((f"draft_results_{ds}_syn{m}_seed{s}.json",
                                 ["--data", ds, *cfg, "--seed", str(s), *COMMON, "--syn_size", str(m)]))
            if exp in ("C", "all"):
                jobs.append((f"draft_results_{ds}_reuse_seed{s}.json",
                             ["--data", ds, *cfg, "--seed", str(s), *COMMON, "--save_synthetic", "true"]))
                jobs.append((f"draft_results_{ds}_reusectl_seed{s}.json",
                             ["--data", ds, *cfg, "--seed", str(s), *COMMON, "--save_synthetic", "true",
                              "--fairness_off"]))
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
    ap.add_argument("--exp", choices=["B", "C", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=4, help="parallel runs (each is single-threaded)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    jobs = jobs_for(a.exp)
    print(f"{len(jobs)} runs, {a.jobs} in parallel")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for msg in ex.map(lambda j: run_one(j, a.dry_run), jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
