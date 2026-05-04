# Individual Reflection — Lab 18

**Tên:** Trương Quang Lộc  
**Module phụ trách:** M3 — Reranking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: `src/m3_rerank.py` — Cross-encoder reranking pipeline
- Các hàm/class chính đã viết:
  - `CrossEncoderReranker._load_model()` — lazy-load `BAAI/bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder`
  - `CrossEncoderReranker.rerank()` — tạo query-doc pairs, chạy `model.predict()`, sort descending, trả về top-k `RerankResult`
  - `FlashrankReranker.rerank()` — lightweight alternative dùng `flashrank`
  - `benchmark_reranker()` — đo latency qua nhiều lần chạy, trả về `avg_ms`, `min_ms`, `max_ms`
  - `contextual_prepend()` trong M5 — thêm ngữ cảnh ngắn vào chunk trước khi indexing
- Số tests pass:
  - `pytest tests/test_m3.py -v` -> `5 passed`
  - `test_contextual_*` trong `tests/test_m5.py` -> `2/2 passed`

---

## 2. Kiến thức học được

- Khái niệm mới nhất: Cross-encoder reranking — khác bi-encoder ở chỗ nhận đồng thời cả query lẫn document vào một forward pass, cho độ chính xác cao hơn nhưng chậm hơn bi-encoder.
- Điều bất ngờ nhất: Nếu retrieval ban đầu kéo về nhiều chunk gần đúng nhưng còn nhiễu, chỉ cần reranking tốt hơn cũng có thể cải thiện đáng kể đầu vào của mô hình trả lời.
- Kết nối với bài giảng: Phần M3 gắn trực tiếp với ý tưởng “retrieve nhiều rồi lọc lại” trong production RAG, giúp precision tốt hơn so với chỉ lấy top-k retrieval thô.

---

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Bước reranking phụ thuộc model ngoài, nên khi môi trường chưa tải được model hoặc chưa có đủ dependency thì rất dễ fail khi test và chạy pipeline.
- Cách giải quyết: Dùng hướng triển khai vừa hỗ trợ model thật, vừa có fallback để pipeline vẫn giữ đúng interface và tiếp tục chạy được khi môi trường chưa hoàn chỉnh.
- Thời gian debug: Khoảng 1-2 giờ để kiểm tra logic xếp hạng, kiểu dữ liệu đầu ra và benchmark latency.

---

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Tôi sẽ benchmark thêm nhiều kiểu query hơn để đo rõ reranker cải thiện ở trường hợp nào nhiều nhất.
- Module nào muốn thử tiếp: Tôi muốn thử thêm M2 hoặc M5 để xem khi kết hợp retrieval tốt hơn với contextual enrichment thì precision có tăng mạnh hay không.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
