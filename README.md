# Lab 24 - Full Evaluation & Guardrail System

## Overview

This repository packages a complete Lab 24 submission around an existing Vietnamese legal RAG corpus. The deliverables cover four layers: automated evaluation with a curated synthetic test set, LLM-as-judge style comparison and calibration, guardrails for both input and output, and a production blueprint with SLOs, incident handling, and cost planning. The implementation is intentionally offline-friendly: where external APIs are unavailable, the scripts fall back to deterministic lexical scoring and rule-based safety checks so the repo still produces the required artifacts end-to-end.

The corpus focuses on data privacy, cybersecurity, and legal compliance documents in Vietnamese. The resulting stack demonstrates how to measure answer quality, identify failure clusters, compare answer variants, block prompt-injection-like traffic, redact PII, and benchmark latency across the full path. This keeps the submission reproducible while staying close to the production framing of the lab.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Optional environment variables:

```bash
export OPENAI_API_KEY=...
export GROQ_API_KEY=...
```

## Run

```bash
./.venv/bin/python phase-a/run_eval.py
./.venv/bin/python phase-b/run_judge.py
./.venv/bin/python phase-c/full_pipeline.py
./.venv/bin/python check_lab24.py
```

## Results Summary

### Phase A

- Test set: 50 questions with curated `simple`, `reasoning`, and `multi_context` coverage
- Outputs: `phase-a/testset_v1.csv`, `phase-a/ragas_results.csv`, `phase-a/ragas_summary.json`
- Failure analysis: `phase-a/failure_analysis.md`
- Scores: Faithfulness `0.4469`, Answer Relevancy `0.6353`, Context Precision `0.1252`, Context Recall `0.4145`
- Observation: low `faithfulness`, `context_precision`, and `context_recall` indicate retrieval/context packaging is currently the main bottleneck, especially for reasoning and multi-context questions.

### Phase B

- Pairwise judging with swap-and-average mitigation across 30 questions
- Outputs: `phase-b/pairwise_results.csv`, `phase-b/absolute_scores.csv`, `phase-b/human_labels.csv`
- Calibration and bias notes: `phase-b/kappa_analysis_output.md`, `phase-b/judge_bias_report.md`
- Cohen's kappa on the 10-sample calibration set: `1.0`

### Phase C

- Input guardrails: PII redaction, topic validation, injection detection
- Output guardrail: rule-based safety classifier shaped like a Llama Guard layer
- Benchmark outputs: `phase-c/pii_test_results.csv`, `phase-c/adversarial_test_results.csv`, `phase-c/latency_benchmark.csv`
- PII recall: `0.80`, adversarial detection: `0.95`, unsafe-output detection: `0.80`, false positive rate: `0.00`
- Latency: P50 `0.033 ms`, P95 `0.109 ms`, P99 `0.131 ms`

### Phase D

- Production blueprint: `phase-d/blueprint.md`

## Demo Video

Add `demo/demo-video.mp4` or a YouTube link here before final submission.
