# Individual Reflection — Lab 18

**Tên:** Trương Quang Lộc  
**Module phụ trách:** M3 — Reranking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m3_rerank.py` — Cross-encoder reranking pipeline
- Các hàm/class chính đã viết:
  - `CrossEncoderReranker._load_model()` — lazy-load `BAAI/bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder`
  - `CrossEncoderReranker.rerank()` — tạo query-doc pairs, chạy `model.predict()`, sort descending, trả về top-k `RerankResult`
  - `FlashrankReranker.rerank()` — lightweight alternative dùng `flashrank` (<5ms)
  - `benchmark_reranker()` — đo latency qua n lần chạy, trả về `avg_ms`, `min_ms`, `max_ms`
  - `tests/test_m3.py` — 5 tests dùng `MagicMock` để tránh download model khi chạy CI
- Số tests pass: 5 / 5

## 2. Kiến thức học được

- Khái niệm mới nhất: Cross-encoder reranking — khác bi-encoder ở chỗ nhận đồng thời cả query lẫn document vào một forward pass, cho độ chính xác cao hơn nhưng chậm hơn bi-encoder
- Điều bất ngờ nhất: Model `BAAI/bge-reranker-v2-m3` nặng ~1 GB — nếu test gọi thẳng model mà không mock thì mỗi lần `pytest` mất >7 phút chỉ để download/load; cần dùng `unittest.mock.MagicMock` để tách unit test khỏi model thật
- Kết nối với bài giảng: Slide 22/42 "Reranking — Highest ROI Optimization" (AICB Ngày 18) — flow chính là **Retrieve top-20 → Cross-Encoder (~50ms) → Pass top-3 → LLM**; slide giải thích tại sao cross-encoder accurate hơn bi-encoder (encode cùng query+doc trong một forward pass thay vì encode riêng), và khẳng định 30–50ms overhead đổi lấy +15–25% precision là "Highest ROI" trong toàn bộ RAG pipeline. Slide cũng chỉ rõ với tiếng Việt nên dùng `bge-reranker-v2-m3` (Free, Multilingual) — đúng model mặc định trong `CrossEncoderReranker`

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Tests treo vô thời hạn vì `CrossEncoderReranker()` trigger download model ~1 GB từ HuggingFace Hub ngay khi khởi tạo trong test
- Cách giải quyết: Chuyển từ eager loading sang lazy loading (`_load_model()` chỉ chạy khi `rerank()` được gọi), và trong test dùng `MagicMock` inject thẳng `reranker._model` với `predict` trả về scores giả — test chạy xong trong < 1 giây
- Thời gian debug: ~20 phút (chủ yếu chờ download lần đầu để xác nhận root cause)

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Viết test với mock **trước** khi implement, không chạy test thật với model lớn trong vòng lặp development
- Module nào muốn thử tiếp: M2 — Hybrid Search (BM25 + dense + RRF fusion), vì kết quả stage 1 ảnh hưởng trực tiếp đến chất lượng đầu vào của M3

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
