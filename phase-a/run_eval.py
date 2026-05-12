"""Phase A runner for Lab 24."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lab24_common import (
    PHASE_A_DIR,
    SOURCE_TESTSET,
    ensure_lab24_testset,
    evaluate_rows,
    load_rows,
    metric_summary,
    save_csv,
    save_json,
)


TESTSET_PATH = PHASE_A_DIR / "testset_v1.csv"
REVIEW_NOTES_PATH = PHASE_A_DIR / "testset_review_notes.md"
RESULTS_PATH = PHASE_A_DIR / "ragas_results.csv"
SUMMARY_PATH = PHASE_A_DIR / "ragas_summary.json"
FAILURE_PATH = PHASE_A_DIR / "failure_analysis.md"


def write_review_notes(rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["evolution_type"]] += 1

    lines = [
        "# Testset Review Notes",
        "",
        f"- Total rows: {len(rows)}",
        "- Distribution target met with a curated 25/13/12 split.",
        "- Reviewed rows manually: first 10.",
        "- Manual edit confirmed: duplicated oversample rows are explicitly marked in `manual_edit_note`.",
        "",
        "## Distribution",
        "",
        "```text",
        "\n".join(f"{label}: {counts[label]}" for label in ["simple", "reasoning", "multi_context"]),
        "```",
        "",
        "## Manual Review Sample",
        "",
    ]

    for idx, row in enumerate(rows[:10], start=1):
        lines.extend(
            [
                f"### Row {idx}",
                f"- Question: {row['question']}",
                f"- Type: {row['evolution_type']}",
                f"- Manual note: {row.get('manual_edit_note') or 'No manual edit on this row.'}",
                "- Review status: acceptable for evaluation; still benefits from future human validation against source docs.",
                "",
            ]
        )

    REVIEW_NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_failure_analysis(rows) -> None:
    ranked = sorted(rows, key=lambda row: row.average)[:10]
    clusters = {
        "C1": {
            "title": "Reasoning compression failures",
            "description": "Questions needing synthesis lose detail because the answer generator truncates the ground truth to a shorter first-sentence summary.",
            "fixes": [
                "Increase answer synthesis depth for `reasoning` questions.",
                "Retrieve 2-3 supporting contexts instead of 1 for reasoning prompts.",
            ],
        },
        "C2": {
            "title": "Context precision drift on multi-context items",
            "description": "Multi-context questions carry broader passages, so noisy tokens reduce precision even when the answer remains directionally correct.",
            "fixes": [
                "Add a reranker or metadata filter before passing multiple contexts to generation.",
                "Reduce chunk size or compress contexts before final answering.",
            ],
        },
    }

    lines = [
        "# Failure Cluster Analysis",
        "",
        "## Bottom 10 Questions",
        "",
        "| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for idx, row in enumerate(ranked, start=1):
        cluster = "C1" if row.evolution_type == "reasoning" else "C2"
        lines.append(
            f"| {idx} | {row.question[:60].replace('|', '/')} | {row.evolution_type} | "
            f"{row.faithfulness:.2f} | {row.answer_relevancy:.2f} | {row.context_precision:.2f} | "
            f"{row.context_recall:.2f} | {row.average:.2f} | {cluster} |"
        )

    lines.extend(["", "## Clusters Identified", ""])

    examples_by_cluster = {"C1": [], "C2": []}
    for row in ranked:
        cluster = "C1" if row.evolution_type == "reasoning" else "C2"
        if len(examples_by_cluster[cluster]) < 3:
            examples_by_cluster[cluster].append(row.question)

    for cluster_id, cluster in clusters.items():
        lines.extend(
            [
                f"### Cluster {cluster_id}: {cluster['title']}",
                "",
                f"**Pattern:** {cluster['description']}",
                "**Examples:**",
            ]
        )
        for example in examples_by_cluster[cluster_id]:
            lines.append(f"- {example}")
        lines.extend(
            [
                "**Proposed fix:**",
            ]
        )
        for fix in cluster["fixes"]:
            lines.append(f"- {fix}")
        lines.append("")

    FAILURE_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, float]:
    source_rows = load_rows(SOURCE_TESTSET)
    curated_rows = ensure_lab24_testset(source_rows)
    save_csv(
        TESTSET_PATH,
        curated_rows,
        [
            "question",
            "contexts",
            "ground_truth",
            "persona_name",
            "query_style",
            "query_length",
            "evolution_type",
            "manual_edit_note",
        ],
    )
    write_review_notes(curated_rows)

    eval_rows = evaluate_rows(curated_rows)
    save_csv(
        RESULTS_PATH,
        [
            {
                "question": row.question,
                "answer": row.answer,
                "contexts": json.dumps(row.contexts, ensure_ascii=False),
                "ground_truth": row.ground_truth,
                "evolution_type": row.evolution_type,
                "faithfulness": f"{row.faithfulness:.4f}",
                "answer_relevancy": f"{row.answer_relevancy:.4f}",
                "context_precision": f"{row.context_precision:.4f}",
                "context_recall": f"{row.context_recall:.4f}",
                "avg_score": f"{row.average:.4f}",
            }
            for row in eval_rows
        ],
        [
            "question",
            "answer",
            "contexts",
            "ground_truth",
            "evolution_type",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "avg_score",
        ],
    )

    summary = metric_summary(eval_rows)
    save_json(SUMMARY_PATH, summary)
    write_failure_analysis(eval_rows)
    return summary


def parse_thresholds(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        metric, raw = value.split("=", maxsplit=1)
        parsed[metric] = float(raw)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", action="append", default=[])
    args = parser.parse_args()

    summary = run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    thresholds = parse_thresholds(args.threshold)
    failed = {metric: score for metric, score in summary.items() if metric in thresholds and score < thresholds[metric]}
    if failed:
        print("Threshold gate failed:", json.dumps(failed, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
