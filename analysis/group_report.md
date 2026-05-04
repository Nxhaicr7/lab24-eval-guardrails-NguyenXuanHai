# Group Report — Lab 18: Production RAG

**Nhóm:** [Tên]  
**Ngày:**

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Hàn Quang Hiếu (2A202600056) | M1: Chunking + M5 `summarize_chunk()` | ✅ | 13/13 (M1) · 2/2 (M5 summarize) |
| Nguyễn Hải | M2: Hybrid Search + M5 `generate_hypothesis_questions()` | ☐ | /5 |
| Lộc | M3: Reranking + M5 `contextual_prepend()` | ☐ | /5 |
| Nguyễn Triệu Gia Khánh | M4: Evaluation + M5 `extract_metadata()` + `enrich_chunks()` | ☐ | /4 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | | | |
| Answer Relevancy | | | |
| Context Precision | | | |
| Context Recall | | | |

## Đóng góp chi tiết — Hàn Quang Hiếu

**M1 — Advanced Chunking Strategies** (`src/m1_chunking.py`):
- `chunk_semantic()`: nhóm câu theo cosine similarity dùng OpenAI `text-embedding-3-small`, có TF-IDF fallback
- `chunk_hierarchical()`: parent (2048 chars) + child (256 chars), mỗi child có `parent_id`
- `chunk_structure_aware()`: parse markdown headers H1–H3, chunk theo section
- `compare_strategies()`: bảng so sánh 4 strategies
- `load_documents()`: load `.md`/`.txt` + OCR PDF qua LlamaParse

**M5 — Enrichment** (`src/m5_enrichment.py`):
- `summarize_chunk()`: gọi `gpt-4o-mini` tóm tắt chunk 2-3 câu; fallback extractive khi không có API key



1. **Biggest improvement:**
2. **Biggest challenge:**
3. **Surprise finding:**

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production):
2. Biggest win — module nào, tại sao:
3. Case study — 1 failure, Error Tree walkthrough:
4. Next optimization nếu có thêm 1 giờ:
