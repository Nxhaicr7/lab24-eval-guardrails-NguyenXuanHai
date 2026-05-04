# Individual Reflection — Lab 18

**Tên:** Hàn Quang Hiếu  
**Mã học viên:** 2A202600056  
**Module phụ trách:** M1 — Advanced Chunking Strategies + M5 `summarize_chunk()`

---

## 1. Đóng góp kỹ thuật

- **Module M1** (`src/m1_chunking.py`): Implement đầy đủ 4 chunking strategies:
  - `chunk_basic()` — baseline paragraph split (có sẵn, dùng để so sánh)
  - `chunk_semantic()` — nhóm câu theo cosine similarity dùng OpenAI `text-embedding-3-small`; có TF-IDF fallback khi không có API key
  - `chunk_hierarchical()` — parent (2048 chars) + child (256 chars), mỗi child có `parent_id` trỏ về parent
  - `chunk_structure_aware()` — parse markdown headers (H1–H3), chunk theo section logic
  - `compare_strategies()` — chạy cả 4 strategies, in bảng so sánh stats
  - `load_documents()` — load `.md`/`.txt` và OCR PDF qua LlamaParse API
- **Module M5** (`src/m5_enrichment.py`): Implement `summarize_chunk()` — gọi `gpt-4o-mini` để tóm tắt chunk thành 2-3 câu; fallback extractive (2 câu đầu) khi không có API key
- **Tests pass:** M1: 13/13 · M5 (phần summarize): 2/2

---

## 2. Kiến thức học được

### LlamaParse — OCR thông minh cho PDF

Trước lab này mình chỉ biết dùng `PyPDF2` hay `pdfminer` để extract text từ PDF — kết quả thường bị mất formatting, bảng biểu bị vỡ, tiếng Việt có dấu bị sai. LlamaParse giải quyết vấn đề này bằng cách dùng vision model để "đọc" PDF như người đọc thật, trả về Markdown có cấu trúc.

Điều thú vị là LlamaParse hỗ trợ `language="vi"` — nó biết đây là tài liệu tiếng Việt và xử lý dấu thanh chính xác hơn. Kết quả trả về là Markdown với headers, bảng, bullet points được giữ nguyên, rất thuận tiện cho `chunk_structure_aware()` xử lý tiếp.

Hạn chế: LlamaParse là cloud API, mỗi lần parse mất vài giây và tốn credit. Trong production thực tế nên cache kết quả ra file `.md` (như repo này đã làm với `data/*_ocr.md`) thay vì parse lại mỗi lần chạy.

### Semantic Chunking với LLM Embeddings

Ý tưởng cốt lõi: thay vì cắt text theo số ký tự cố định, semantic chunking đo **độ tương đồng ngữ nghĩa** giữa các câu liên tiếp. Khi cosine similarity giữa câu `i` và câu `i+1` giảm xuống dưới threshold, đó là ranh giới tự nhiên giữa 2 ý — cắt ở đó.

Trong implementation, mình dùng `text-embedding-3-small` của OpenAI để embed tất cả câu trong 1 API call duy nhất (batch), sau đó tính dot product giữa các vector đã L2-normalize (dot product của unit vector = cosine similarity). Cách này nhanh hơn nhiều so với gọi API từng câu một.

Điều học được quan trọng: **threshold rất nhạy cảm**. Với tiếng Việt, các câu trong cùng 1 đoạn thường có similarity ~0.85–0.92. Nếu threshold = 0.85 thì gần như không bao giờ cắt. Phải thực nghiệm để tìm giá trị phù hợp với corpus cụ thể.

TF-IDF fallback cũng thú vị — sparse vector có distribution similarity khác hẳn dense embedding, nên phải dùng `effective_threshold = min(threshold, 0.15)` thay vì dùng nguyên threshold gốc.

### Hierarchical Chunking — Tại sao đây là best practice

Hierarchical chunking giải quyết tension cơ bản trong RAG: **chunk nhỏ** thì retrieval chính xác hơn (ít noise), nhưng **chunk lớn** thì LLM có đủ context để trả lời. Giải pháp: index chunk nhỏ (child, 256 chars) để search, nhưng khi trả về context cho LLM thì dùng chunk lớn (parent, 2048 chars) chứa child đó.

Mỗi child chunk có `parent_id` để pipeline có thể lookup parent tương ứng. Đây là pattern "small-to-big retrieval" được nhiều production RAG system dùng.

### Chunk Summarization trong M5

`summarize_chunk()` dùng `gpt-4o-mini` để tóm tắt chunk thành 2-3 câu trước khi embed. Lý do: chunk gốc có thể chứa nhiều chi tiết thừa, boilerplate, hay cách diễn đạt vòng vo. Summary ngắn gọn hơn thường có embedding gần với query của user hơn, cải thiện retrieval recall.

Chi phí rất thấp với `gpt-4o-mini` (~$0.00015/1K tokens) và chỉ chạy 1 lần lúc indexing — không ảnh hưởng latency lúc query.

---

## 3. Khó khăn & Cách giải quyết

**Khó khăn 1 — Semantic chunking tạo quá nhiều chunk nhỏ:**  
Ban đầu threshold = 0.85 khiến gần như mỗi câu thành 1 chunk riêng vì similarity giữa các câu tiếng Việt thường cao. Giải pháp: thêm điều kiện `len(current_group) >= 2` — chỉ cắt khi nhóm hiện tại đã có ít nhất 2 câu, tránh chunk đơn câu vô nghĩa.

**Khó khăn 2 — TF-IDF fallback cho kết quả khác hẳn dense embedding:**  
Sparse TF-IDF vector có cosine similarity thấp hơn nhiều so với dense embedding cho cùng 1 cặp câu. Nếu dùng cùng threshold thì TF-IDF fallback không bao giờ cắt chunk. Giải pháp: `effective_threshold = min(threshold, 0.15)` — scale down threshold cho phù hợp với distribution của TF-IDF.

**Khó khăn 3 — `chunk_structure_aware()` edge case không có header:**  
Nếu text không có markdown header nào, hàm trả về list rỗng. Giải pháp: thêm fallback cuối hàm — nếu `chunks` rỗng và text không rỗng thì wrap toàn bộ text thành 1 chunk duy nhất.

**Thời gian debug:** ~25 phút cho semantic chunking, ~10 phút cho structure-aware.

---

## 4. Nếu làm lại

- **Thêm overlap cho hierarchical:** Hiện tại child chunks non-overlapping (start = end). Overlap 50% (`start = start + child_size // 2`) sẽ giảm nguy cơ cắt giữa câu quan trọng, đặc biệt với văn bản pháp lý tiếng Việt có câu dài.
- **Cache LlamaParse output:** Parse PDF mỗi lần chạy tốn thời gian và API credit. Nên save kết quả ra `.md` ngay sau lần parse đầu tiên và đọc từ cache cho các lần sau.
- **Thử sentence-transformers thay OpenAI cho semantic chunking:** `bge-m3` đã có sẵn trong project (dùng cho M2), có thể dùng luôn để embed câu thay vì gọi OpenAI API — tiết kiệm chi phí và chạy offline.
- **Module nào muốn thử tiếp:** M2 (Hybrid Search) — muốn hiểu cách RRF fusion hoạt động thực tế và so sánh BM25 vs dense trên tiếng Việt.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | 4 | Nắm được lý thuyết chunking, còn muốn đào sâu hơn về RRF |
| Code quality | 4 | Có type hints, comments, fallback — thiếu unit test cho edge cases |
| Teamwork | 4 | Xong phần mình đúng hạn, hỗ trợ implement M5 cho cả nhóm |
| Problem solving | 4 | Tự debug được semantic threshold issue, TF-IDF fallback |
