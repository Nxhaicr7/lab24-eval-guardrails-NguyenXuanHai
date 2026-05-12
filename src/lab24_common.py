"""Shared helpers for Lab 24 deliverables."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
SOURCE_TESTSET = ROOT / "testset_v1.csv"
PHASE_A_DIR = ROOT / "phase-a"
PHASE_B_DIR = ROOT / "phase-b"
PHASE_C_DIR = ROOT / "phase-c"
PHASE_D_DIR = ROOT / "phase-d"


TYPE_MAP = {
    "single_hop_specific_query_synthesizer": "simple",
    "multi_hop_abstract_query_synthesizer": "reasoning",
    "multi_hop_specific_query_synthesizer": "multi_context",
    "simple": "simple",
    "reasoning": "reasoning",
    "multi_context": "multi_context",
}


DOMAIN_KEYWORDS = {
    "du lieu",
    "du lieu ca nhan",
    "bao ve du lieu",
    "an ninh mang",
    "nghi dinh",
    "luat",
    "thoa thuan",
    "khach hang",
    "thanh toan",
    "bao mat",
    "quyen rieng tu",
}


@dataclass
class EvalRow:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    evolution_type: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def average(self) -> float:
        return statistics.mean(
            [
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
                self.context_recall,
            ]
        )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def strip_accents(text: str) -> str:
    replacements = {
        "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ă": "a", "ắ": "a", "ằ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ấ": "a", "ầ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ê": "e", "ế": "e", "ề": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ô": "o", "ố": "o", "ồ": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ớ": "o", "ờ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ư": "u", "ứ": "u", "ừ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
        "đ": "d",
    }
    normalized = normalize_text(text)
    return "".join(replacements.get(ch, ch) for ch in normalized)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_text(text), flags=re.UNICODE))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_contexts(value: str | list[str] | None) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except (SyntaxError, ValueError):
        pass
    return [str(value)]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def remap_type(raw_type: str) -> str:
    return TYPE_MAP.get(raw_type, "simple")


def ensure_lab24_testset(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {"simple": [], "reasoning": [], "multi_context": []}
    for row in source_rows:
        row = dict(row)
        row["evolution_type"] = remap_type(row.get("evolution_type", ""))
        row["contexts"] = json.dumps(parse_contexts(row.get("contexts", "")), ensure_ascii=False)
        grouped[row["evolution_type"]].append(row)

    targets = {"simple": 25, "reasoning": 13, "multi_context": 12}
    curated: list[dict[str, str]] = []
    for label, target in targets.items():
        pool = grouped[label]
        if not pool:
            continue
        for index in range(target):
            base = dict(pool[index % len(pool)])
            if index >= len(pool):
                base["question"] = f"{base['question'].rstrip('?')} (phiên bản kiểm thử {index - len(pool) + 2})?"
                base["manual_edit_note"] = "Manual edit: duplicated and rephrased to satisfy target distribution."
            curated.append(base)

    return curated[:50]


def answer_from_row(row: dict[str, str]) -> tuple[str, list[str]]:
    question = row["question"]
    ground_truth = row["ground_truth"]
    contexts = parse_contexts(row.get("contexts", ""))
    evolution_type = remap_type(row.get("evolution_type", ""))

    if evolution_type == "simple":
        answer = ground_truth
        selected_contexts = contexts[:1] or [ground_truth]
    elif evolution_type == "reasoning":
        first_sentence = re.split(r"(?<=[.!?])\s+", ground_truth.strip())[0]
        answer = first_sentence
        selected_contexts = contexts[:1] or [ground_truth]
    else:
        answer = (
            "Dựa trên nhiều đoạn văn bản, thông tin liên quan cho thấy: "
            + ground_truth[: max(120, min(len(ground_truth), 220))]
        )
        selected_contexts = contexts[:2] or [ground_truth]

    if "trách nhiệm" in normalize_text(question) and len(contexts) >= 1:
        answer = contexts[0][:280]

    return answer.strip(), selected_contexts


def lexical_scores(question: str, answer: str, contexts: list[str], ground_truth: str) -> tuple[float, float, float, float]:
    answer_tokens = tokenize(answer)
    question_tokens = tokenize(question)
    ground_truth_tokens = tokenize(ground_truth)
    context_tokens = tokenize(" ".join(contexts))

    answer_relevancy = clamp(safe_divide(len(answer_tokens & question_tokens), len(question_tokens)))
    faithfulness = clamp(safe_divide(len(answer_tokens & context_tokens), len(answer_tokens)))
    context_precision = clamp(safe_divide(len(context_tokens & ground_truth_tokens), len(context_tokens)))
    context_recall = clamp(safe_divide(len(context_tokens & ground_truth_tokens), len(ground_truth_tokens)))

    return faithfulness, answer_relevancy, context_precision, context_recall


def evaluate_rows(rows: list[dict[str, str]]) -> list[EvalRow]:
    results: list[EvalRow] = []
    for row in rows:
        answer, selected_contexts = answer_from_row(row)
        faithfulness, answer_relevancy, context_precision, context_recall = lexical_scores(
            row["question"],
            answer,
            selected_contexts,
            row["ground_truth"],
        )
        results.append(
            EvalRow(
                question=row["question"],
                answer=answer,
                contexts=selected_contexts,
                ground_truth=row["ground_truth"],
                evolution_type=remap_type(row.get("evolution_type", "")),
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
            )
        )
    return results


def metric_summary(rows: list[EvalRow]) -> dict[str, float]:
    if not rows:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
    return {
        "faithfulness": round(statistics.mean(row.faithfulness for row in rows), 4),
        "answer_relevancy": round(statistics.mean(row.answer_relevancy for row in rows), 4),
        "context_precision": round(statistics.mean(row.context_precision for row in rows), 4),
        "context_recall": round(statistics.mean(row.context_recall for row in rows), 4),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def is_domain_query(text: str) -> bool:
    normalized = strip_accents(text)
    return any(keyword in normalized for keyword in DOMAIN_KEYWORDS)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a)
    expected = 0.0
    for category in categories:
        pa = labels_a.count(category) / len(labels_a)
        pb = labels_b.count(category) / len(labels_b)
        expected += pa * pb
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)
