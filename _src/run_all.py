# -*- coding: utf-8 -*-
"""
One-shot runner for the Chinese Plan-B TaskMatch pipeline.

Usage:
  python run_all.py [--threshold 0.65] [--top_k 10]

Steps:
  1. build_task_library.py   Task_DWA.csv -> onet_tasks_dedup.csv
  2. translate_tasks.py      -> onet_tasks_zh.csv (pluggable backend)
  3. extract_sentences.py    test set.csv -> chinese_job_sentences.csv
  4. run_taskmatch.py        -> taskmatch_sentence_level.csv / taskmatch_job_level.csv
  5. report.py               -> report.md
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script, extra=None):
    cmd = [sys.executable, str(HERE / script)] + (extra or [])
    print(f"\n=== {script} {' '.join(extra or [])} ===", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--translate-backend", default="google")
    ap.add_argument("--skip-translate", action="store_true", help="skip translation step")
    args = ap.parse_args()

    run("build_task_library.py")
    if not args.skip_translate:
        run("translate_tasks.py", ["--backend", args.translate_backend, "--workers", "6", "--resume"])
    else:
        print("skipping translation (requires existing onet_tasks_zh.csv)")
    run("extract_sentences.py")
    run("run_taskmatch.py", ["--threshold", str(args.threshold), "--top_k", str(args.top_k)])
    run("report.py")


if __name__ == "__main__":
    main()