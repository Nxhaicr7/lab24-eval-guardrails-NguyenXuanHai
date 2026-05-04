"""Tests for Module 4: Evaluation."""
import sys, os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report, EvalResult

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics():
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))
    assert "per_question" in r and len(r["per_question"]) == 1
    assert isinstance(r["per_question"][0], EvalResult)

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]


def test_failure_analysis_maps_worst_metric():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.9, 0.85, 0.8, 0.2)]
    f = failure_analysis(results, bottom_n=1)
    assert f[0]["worst_metric"] == "context_recall"
    assert f[0]["diagnosis"] == "Missing relevant chunks"


def test_save_report_writes_expected_shape(tmp_path):
    results = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    failures = failure_analysis(results["per_question"], bottom_n=1)
    report_path = tmp_path / "ragas_report.json"

    save_report(results, failures, path=str(report_path))

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert "aggregate" in saved
    assert "num_questions" in saved and saved["num_questions"] == 1
    assert "failures" in saved
