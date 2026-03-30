"""
run_all.py
──────────
Master script that runs both tasks with all model × embedding combinations
and then prints a final consolidated summary.

Usage:
    python run_all.py                    # full run (all combos, default epochs)
    python run_all.py --epochs 5         # quick smoke-test
    python run_all.py --task1_only
    python run_all.py --task2_only
"""

import argparse
import subprocess
import sys
import os


def run(cmd: list):
    print(f"\n▶  {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--task1_only", action="store_true")
    parser.add_argument("--task2_only", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    if not args.task2_only:
        print("\n" + "=" * 70)
        print("  TASK 1 — TEXT GENERATION  (Shakespeare, all model×embedding combos)")
        print("=" * 70)
        run([
            python, "task1_text_generation/train.py",
            "--dataset", "shakespeare",
            "--epochs",  str(args.epochs),
            "--run_all",
        ])

    if not args.task1_only:
        print("\n" + "=" * 70)
        print("  TASK 2 — MACHINE TRANSLATION  (English→German, all combos)")
        print("=" * 70)
        run([
            python, "task2_machine_translation/train.py",
            "--epochs", str(args.epochs),
            "--run_all",
        ])

    print("\n" + "=" * 70)
    print("  All experiments complete.  Results saved to results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
