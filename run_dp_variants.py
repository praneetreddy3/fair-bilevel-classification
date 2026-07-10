"""
Run multiple DP variants of the same model configuration.

Example:
python run_dp_variants.py --data adult --rounds 3 --sigma 1.0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Batch-run run_draft.py over DP variants.")
    p.add_argument("--data", choices=["dummy", "adult", "2d"], default="adult")
    p.add_argument("--sensitive", choices=["sex", "race"], default="sex")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument(
        "--variants",
        nargs="+",
        default=["none", "pre_server", "post_server", "both"],
        help="Subset of: none pre_server post_server both",
    )
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        out_file = f"draft_results_{args.data}_{variant}.json"
        cmd = [
            sys.executable,
            "-m",
            "draft_model.run_draft",
            "--data",
            args.data,
            "--sensitive",
            args.sensitive,
            "--rounds",
            str(args.rounds),
            "--num_clients",
            str(args.num_clients),
            "--dp_enabled",
            "true",
            "--dp_sigma",
            str(args.sigma),
            "--dp_variant",
            variant,
            "--out_dir",
            str(out_dir),
            "--results_file",
            out_file,
        ]
        print(f"\n=== Running variant: {variant} ===")
        subprocess.run(cmd, check=True, cwd=str(root))

    print(f"\nCompleted. Results in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
