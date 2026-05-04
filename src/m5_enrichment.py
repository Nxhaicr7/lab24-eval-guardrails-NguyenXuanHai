"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving readable text."""
    return " ".join(text.split())


def _detect_language(text: str) -> str:
    """Detect whether a chunk is primarily Vietnamese or English."""
    lowered = text.lower()
    vietnamese_markers = [
        "nhân viên",
        "nghỉ phép",
        "mật khẩu",
        "chính sách",
        "quy định",
        "được",
        "không",
        "và",
    ]
    if any(marker in lowered for marker in vietnamese_markers):
        return "vi"

    if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", lowered):
        return "vi"

    return "en"


def _classify_category(text: str) -> str:
    """Map chunk text to a coarse retrieval category."""
    lowered = text.lower()
    category_keywords = {
        "hr": [
            "nhân viên",
            "nghỉ phép",
            "thâm niên",
            "chấm công",
            "lương",
            "phúc lợi",
            "bảo hiểm",
            "tuyển dụng",
        ],
        "it": [
            "mật khẩu",
            "vpn",
            "email",
            "tài khoản",
            "hệ thống",
            "bảo mật",
            "máy tính",
            "phần mềm",
            "2fa",
        ],
        "finance": [
            "chi phí",
            "ngân sách",
            "hóa đơn",
            "thanh toán",
            "hoàn ứng",
            "công tác phí",
            "tài chính",
            "doanh thu",
        ],
        "policy": [
            "chính sách",
            "quy định",
            "quy trình",
            "hướng dẫn",
            "sổ tay",
            "bắt buộc",
            "tuân thủ",
        ],
    }

    best_category = "policy"
    best_score = -1
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_category = category
            best_score = score

    return best_category


def _infer_topic(text: str) -> str:
    """Infer a concise topic phrase for metadata filtering."""
    lowered = text.lower()
    known_topics = [
        "nghỉ phép",
        "mật khẩu",
        "bảo mật thông tin",
        "chấm công",
        "tuyển dụng",
        "chi phí công tác",
        "thanh toán",
        "vpn",
        "email",
        "ngân sách",
    ]
    for topic in known_topics:
        if topic in lowered:
            return topic

    first_sentence = re.split(r"(?<=[.!?])\s+", _normalize_whitespace(text).strip())[0]
    topic_words = first_sentence.split()[:6]
    return " ".join(topic_words).strip(" .,:;") or "general"


def _extract_entities(text: str) -> list[str]:
    """Extract lightweight entities useful for search filters."""
    entities: list[str] = []

    for match in re.findall(r"\b\d+\s*(?:ngày|tháng|năm|giờ|%|triệu|tỷ)\b", text, flags=re.IGNORECASE):
        cleaned = _normalize_whitespace(match)
        if cleaned not in entities:
            entities.append(cleaned)

    for match in re.findall(r"\b[A-ZĐ]{2,}(?:[A-Z0-9-]+)?\b", text):
        if match not in entities:
            entities.append(match)

    for match in re.findall(r"\b(?:[A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)+)\b", text):
        cleaned = _normalize_whitespace(match)
        if cleaned not in entities:
            entities.append(cleaned)

    return entities


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    # TODO: Implement chunk summarization
    # Option A (với OpenAI):
    #   from openai import OpenAI
    #   client = OpenAI()
    #   resp = client.chat.completions.create(
    #       model="gpt-4o-mini",
    #       messages=[
    #           {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
    #           {"role": "user", "content": text},
    #       ],
    #       max_tokens=150,
    #   )
    #   return resp.choices[0].message.content.strip()
    #
    # Option B (không cần API — extractive):
    #   sentences = text.split(". ")
    #   return ". ".join(sentences[:2]) + "."  # Lấy 2 câu đầu
    return ""


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text or not text.strip():
        return []

    try:
        from config import OPENAI_API_KEY
    except ImportError:
        OPENAI_API_KEY = ""

    if not OPENAI_API_KEY:
        return ["Nội dung này nói về cái gì?", "Có bao nhiêu ý chính trong đây?"][:n_questions]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi "
                        "mà đoạn văn có thể trả lời. "
                        "Trả về mỗi câu hỏi trên 1 dòng, không đánh số, không giải thích."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            max_tokens=200,
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        # Fallback if openai is not installed or API fails
        content = "\n".join([f"Nội dung này bao gồm những thông tin gì {i}?" for i in range(n_questions)])

    questions = []
    for line in content.splitlines():
        q = line.strip().lstrip("0123456789.-) ").strip()
        if q:
            questions.append(q)

    return questions[:n_questions]

# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    # TODO: Implement contextual prepend
    # 1. from openai import OpenAI
    #    client = OpenAI()
    # 2. resp = client.chat.completions.create(
    #        model="gpt-4o-mini",
    #        messages=[
    #            {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
    #            {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    #        ],
    #        max_tokens=80,
    #    )
    # 3. context = resp.choices[0].message.content.strip()
    # 4. return f"{context}\n\n{text}"
    #
    # Ví dụ output:
    #   "Trích từ Chương 3 - Chính sách nghỉ phép, Sổ tay VinUni 2024.
    #    Nhân viên chính thức được nghỉ phép năm 12 ngày..."
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return {
            "topic": "general",
            "entities": [],
            "category": "policy",
            "language": "en",
        }

    return {
        "topic": _infer_topic(cleaned),
        "entities": _extract_entities(cleaned),
        "category": _classify_category(cleaned),
        "language": _detect_language(cleaned),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []
    expanded_methods = ["summary", "hyqa", "contextual", "metadata"] if "full" in methods else methods

    for chunk in chunks:
        text = chunk.get("text", "")
        base_metadata = dict(chunk.get("metadata", {}))

        summary = summarize_chunk(text) if "summary" in expanded_methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in expanded_methods else []
        contextual_text = (
            contextual_prepend(text, base_metadata.get("source", ""))
            if "contextual" in expanded_methods
            else text
        )
        auto_meta = extract_metadata(text) if "metadata" in expanded_methods else {}

        enrichment_sections = [contextual_text or text]
        if summary:
            enrichment_sections.append(f"Tóm tắt: {summary}")
        if questions:
            joined_questions = "\n".join(f"- {question}" for question in questions)
            enrichment_sections.append(f"Câu hỏi giả định:\n{joined_questions}")

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text="\n\n".join(section for section in enrichment_sections if section).strip(),
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**base_metadata, **auto_meta},
                method="full" if "full" in methods else "+".join(expanded_methods),
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
