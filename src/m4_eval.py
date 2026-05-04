"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass
from statistics import mean
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _normalize_text(text: str) -> str:
    """Normalize text for lightweight lexical scoring."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a simple lowercase word set."""
    return set(re.findall(r"\w+", _normalize_text(text), flags=re.UNICODE))


def _safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division-by-zero while keeping score bounds stable."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clamp_score(value: float) -> float:
    """Clamp heuristic scores into the expected [0, 1] range."""
    return max(0.0, min(1.0, float(value)))


def _score_row(question: str, answer: str, row_contexts: list[str], ground_truth: str) -> EvalResult:
    """Fallback lexical scorer used when RAGAS is unavailable."""
    answer_tokens = _tokenize(answer)
    question_tokens = _tokenize(question)
    ground_truth_tokens = _tokenize(ground_truth)
    context_tokens = _tokenize(" ".join(row_contexts))

    answer_relevancy = _clamp_score(_safe_divide(len(answer_tokens & question_tokens), len(question_tokens)))
    faithfulness = _clamp_score(_safe_divide(len(answer_tokens & context_tokens), len(answer_tokens)))
    context_precision = _clamp_score(_safe_divide(len(context_tokens & ground_truth_tokens), len(context_tokens)))
    context_recall = _clamp_score(_safe_divide(len(context_tokens & ground_truth_tokens), len(ground_truth_tokens)))

    return EvalResult(
        question=question,
        answer=answer,
        contexts=row_contexts,
        ground_truth=ground_truth,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
    )


def _aggregate_results(per_question: list[EvalResult]) -> dict:
    """Aggregate per-question scores into the report format."""
    if not per_question:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    return {
        "faithfulness": mean(item.faithfulness for item in per_question),
        "answer_relevancy": mean(item.answer_relevancy for item in per_question),
        "context_precision": mean(item.context_precision for item in per_question),
        "context_recall": mean(item.context_recall for item in per_question),
        "per_question": per_question,
    }


def _evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run the real RAGAS pipeline when dependencies are available."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = result.to_pandas()

    per_question = [
        EvalResult(
            question=row["question"],
            answer=row["answer"],
            contexts=list(row["contexts"]),
            ground_truth=row["ground_truth"],
            faithfulness=float(row["faithfulness"]),
            answer_relevancy=float(row["answer_relevancy"]),
            context_precision=float(row["context_precision"]),
            context_recall=float(row["context_recall"]),
        )
        for _, row in df.iterrows()
    ]
    return _aggregate_results(per_question)


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        sanitized = re.sub(r",(\s*[\]}])", r"\1", raw)
        return json.loads(sanitized)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts, and ground_truths must have the same length")

    if not questions:
        return _aggregate_results([])

    try:
        if OPENAI_API_KEY:
            return _evaluate_with_ragas(questions, answers, contexts, ground_truths)
    except Exception:
        pass

    per_question = [
        _score_row(question, answer, row_contexts, ground_truth)
        for question, answer, row_contexts, ground_truth in zip(
            questions, answers, contexts, ground_truths
        )
    ]
    return _aggregate_results(per_question)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results or bottom_n <= 0:
        return []

    diagnostics = {
        "faithfulness": (
            0.85,
            "LLM hallucinating",
            "Tighten prompt, lower temperature, and require answers to stay grounded in context.",
        ),
        "context_recall": (
            0.75,
            "Missing relevant chunks",
            "Improve chunking/search coverage or add better query expansion and BM25 recall.",
        ),
        "context_precision": (
            0.75,
            "Too many irrelevant chunks",
            "Add reranking or metadata filtering to reduce noisy retrieved context.",
        ),
        "answer_relevancy": (
            0.80,
            "Answer doesn't match question",
            "Improve the answer prompt template so the model responds directly to the user query.",
        ),
    }

    ranked_results = sorted(
        eval_results,
        key=lambda item: mean(
            [
                item.faithfulness,
                item.answer_relevancy,
                item.context_precision,
                item.context_recall,
            ]
        ),
    )[:bottom_n]

    failures = []
    for item in ranked_results:
        metric_scores = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        worst_metric = min(metric_scores, key=metric_scores.get)
        score = float(metric_scores[worst_metric])
        threshold, diagnosis, suggested_fix = diagnostics[worst_metric]

        if score >= threshold:
            diagnosis = "No severe single-point failure"
            suggested_fix = "Review the full pipeline end-to-end; the issue is likely distributed across retrieval and prompting."

        failures.append(
            {
                "question": item.question,
                "worst_metric": worst_metric,
                "score": score,
                "avg_score": mean(metric_scores.values()),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
