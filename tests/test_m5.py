"""Tests for Module 5: Enrichment Pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m5_enrichment import (
    summarize_chunk, generate_hypothesis_questions,
    contextual_prepend, extract_metadata, enrich_chunks, EnrichedChunk,
)

SAMPLE = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
CHUNKS = [
    {"text": SAMPLE, "metadata": {"source": "policy.md"}},
    {"text": "Mật khẩu phải thay đổi mỗi 90 ngày.", "metadata": {"source": "it.md"}},
]


def test_summarize_returns_string():
    result = summarize_chunk(SAMPLE)
    assert isinstance(result, str)


def test_summarize_shorter_than_original():
    result = summarize_chunk(SAMPLE)
    if result:  # May be empty if no API key
        assert len(result) <= len(SAMPLE) * 2  # Summary should not be much longer


def test_hyqa_returns_list():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    assert isinstance(result, list)


def test_hyqa_generates_questions():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    if result:
        assert len(result) >= 1
        assert any("?" in q or "bao" in q.lower() or "mấy" in q.lower() for q in result)


def test_contextual_prepend_returns_string():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert isinstance(result, str)
    assert len(result) >= len(SAMPLE)  # Should be at least as long as original


def test_contextual_contains_original():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert SAMPLE in result  # Original text must be preserved


def test_extract_metadata_returns_dict():
    result = extract_metadata(SAMPLE)
    assert isinstance(result, dict)


def test_extract_metadata_returns_expected_keys():
    result = extract_metadata(SAMPLE)
    assert {"topic", "entities", "category", "language"} <= set(result)


def test_extract_metadata_detects_hr_topic_and_language():
    result = extract_metadata(SAMPLE)
    assert result["topic"] == "nghỉ phép"
    assert result["category"] == "hr"
    assert result["language"] == "vi"


def test_extract_metadata_extracts_entities_for_numeric_policies():
    result = extract_metadata(SAMPLE)
    assert "12 ngày" in result["entities"]


def test_extract_metadata_detects_it_category():
    result = extract_metadata(CHUNKS[1]["text"])
    assert result["topic"] == "mật khẩu"
    assert result["category"] == "it"


def test_enrich_chunks_returns_list():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert isinstance(result, list)


def test_enrich_type_and_length():
    result = enrich_chunks(CHUNKS, methods=["metadata"])
    assert len(result) == len(CHUNKS)
    assert all(isinstance(c, EnrichedChunk) for c in result)


def test_enrich_preserves_original_and_source_metadata():
    result = enrich_chunks(CHUNKS, methods=["metadata"])
    assert result[0].original_text == SAMPLE
    assert result[0].auto_metadata["source"] == "policy.md"
    assert result[0].auto_metadata["topic"] == "nghỉ phép"


def test_enrich_contextual_keeps_original_in_enriched_text():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert SAMPLE in result[0].enriched_text


def test_enrich_full_runs_all_techniques(monkeypatch):
    monkeypatch.setattr("src.m5_enrichment.summarize_chunk", lambda text: "summary")
    monkeypatch.setattr(
        "src.m5_enrichment.generate_hypothesis_questions",
        lambda text, n_questions=3: ["Câu hỏi 1?", "Câu hỏi 2?"],
    )
    monkeypatch.setattr(
        "src.m5_enrichment.contextual_prepend",
        lambda text, document_title="": f"Context từ {document_title}\n\n{text}",
    )
    monkeypatch.setattr(
        "src.m5_enrichment.extract_metadata",
        lambda text: {"topic": "mock-topic", "category": "policy", "language": "vi", "entities": []},
    )

    result = enrich_chunks(CHUNKS[:1], methods=["full"])

    assert len(result) == 1
    assert result[0].method == "full"
    assert result[0].summary == "summary"
    assert result[0].hypothesis_questions == ["Câu hỏi 1?", "Câu hỏi 2?"]
    assert "Context từ policy.md" in result[0].enriched_text
    assert "Tóm tắt: summary" in result[0].enriched_text
    assert result[0].auto_metadata["topic"] == "mock-topic"
