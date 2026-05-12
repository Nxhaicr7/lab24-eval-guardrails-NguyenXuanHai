"""Lightweight checker for Lab 24 deliverables."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "prompts.md",
    "phase-a/testset_v1.csv",
    "phase-a/testset_review_notes.md",
    "phase-a/ragas_results.csv",
    "phase-a/ragas_summary.json",
    "phase-a/failure_analysis.md",
    "phase-b/pairwise_results.csv",
    "phase-b/absolute_scores.csv",
    "phase-b/human_labels.csv",
    "phase-b/kappa_analysis.py",
    "phase-b/kappa_analysis_output.md",
    "phase-b/judge_bias_report.md",
    "phase-c/input_guard.py",
    "phase-c/output_guard.py",
    "phase-c/full_pipeline.py",
    "phase-c/pii_test_results.csv",
    "phase-c/adversarial_test_results.csv",
    "phase-c/latency_benchmark.csv",
    "phase-d/blueprint.md",
    ".github/workflows/eval-gate.yml",
]


def exists(path: str) -> bool:
    target = ROOT / path
    ok = target.exists()
    print(f"{'OK' if ok else 'MISS'} {path}")
    return ok


def count_rows(path: str) -> int:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not exists(path)]
    if missing:
        print(f"\nMissing {len(missing)} required files.")
        return 1

    print("\nRow checks")
    print("phase-a/testset_v1.csv:", count_rows("phase-a/testset_v1.csv"))
    print("phase-a/ragas_results.csv:", count_rows("phase-a/ragas_results.csv"))
    print("phase-b/pairwise_results.csv:", count_rows("phase-b/pairwise_results.csv"))
    print("phase-c/adversarial_test_results.csv:", count_rows("phase-c/adversarial_test_results.csv"))
    print("phase-c/latency_benchmark.csv:", count_rows("phase-c/latency_benchmark.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
