# Individual Reflection Template — Lab 18

**Họ và tên:** Nguyễn Xuân Hải  
**Mã sinh viên:** 2A202600245
**Phần phụ trách chính:** M2  
**Phần hỗ trợ thêm:** M5 (Hàm generate_hypothesis_questions)

---

## 1. Tổng quan đóng góp

Trong bài lab này, tôi phụ trách phần cá nhân là **Module 2: Hybrid Search**.  
Mục tiêu của phần này là:
- Triển khai tìm kiếm BM25 hỗ trợ tách từ tiếng Việt (Vietnamese Segmentation).
- Kết hợp tìm kiếm Dense (Vector Search) sử dụng Qdrant database.
- Sử dụng thuật toán Reciprocal Rank Fusion (RRF) để kết hợp kết quả từ BM25 và Dense Search thành Hybrid Search.

Hỗ trợ thêm module M5:
- Triển khai hàm `generate_hypothesis_questions` (HyQA) trong Module 5 để làm giàu dữ liệu trước khi index.

---

## 2. Công việc đã thực hiện

### Module chính: `src/m2_search.py`

Các phần tôi đã triển khai:
- `segment_vietnamese`: Sử dụng `underthesea` để chuẩn hóa và tách từ tiếng Việt cho BM25.
- `BM25Search`: Xây dựng index BM25 và hàm tìm kiếm từ khóa.
- `DenseSearch`: Tích hợp QdrantClient để lưu trữ và tìm kiếm vector (Dense Search).
- `reciprocal_rank_fusion`: Triển khai công thức RRF để trộn kết quả từ nhiều nguồn tìm kiếm.
- `HybridSearch`: Class tổng hợp kết hợp BM25 + Dense Search.

### Phần hỗ trợ thêm

Đóng góp ngoài module chính:
- `src/m5_enrichment.py`: Hoàn thiện hàm `generate_hypothesis_questions`.
- Tích hợp cơ chế fallback cho HyQA: Khi không có API Key hoặc thiếu thư viện `openai`, hàm sẽ trả về kết quả giả lập để đảm bảo pipeline và test không bị lỗi.
- Quản lý Git: Xử lý merge conflict, quản lý nhánh cá nhân (`nxhai`) và thực hiện merge chọn lọc (selective merge) các file phụ trách vào nhánh `main`.

---

## 3. Kết quả test

### Kết quả module chính

Lệnh chạy:

```bash
pytest tests/test_m2.py -v
```

Kết quả:

```text
tests/test_m2.py::test_segment_returns_string PASSED
tests/test_m2.py::test_bm25_search PASSED
tests/test_m2.py::test_bm25_relevant_first PASSED
tests/test_m2.py::test_rrf_merges PASSED
tests/test_m2.py::test_rrf_method PASSED
============================== 5 passed in 0.01s ===============================
```

Ý nghĩa:
- Hàm tách từ tiếng Việt hoạt động đúng định dạng.
- BM25 search tìm được kết quả liên quan và trả về đúng phương thức "bm25".
- Thuật toán RRF trộn kết quả chính xác và đánh dấu phương thức là "hybrid".

### Kết quả phần hỗ trợ thêm

Hỗ trợ module M5 (HyQA):

```bash
pytest tests/test_m5.py -k hyqa -v
```

```text
tests/test_m5.py::test_hyqa_returns_list PASSED
tests/test_m5.py::test_hyqa_generates_questions PASSED
======================= 2 passed, 14 deselected in 0.01s =======================
```

---

## 4. Kiến thức học được

- Hiểu sâu về cơ chế Hybrid Search và cách kết hợp ưu điểm của BM25 (keyword matching) và Dense Search (semantic matching).
- Cách sử dụng thuật toán RRF để xếp hạng lại kết quả từ nhiều nguồn tìm kiếm khác nhau mà không cần chuẩn hóa điểm số (score).
- Kỹ thuật Enrichment "HyQA" giúp thu hẹp khoảng cách từ vựng (vocabulary gap) giữa câu hỏi của người dùng và văn bản gốc.
- Quản lý workflow Git nâng cao: Xử lý conflict, stash thay đổi và merge từng file cụ thể (`git checkout <branch> -- <file>`).

---

## 5. Khó khăn và cách giải quyết

### Khó khăn 1: Xử lý tách từ tiếng Việt cho BM25

Trong Tiếng Việt, các từ có thể gồm nhiều tiếng cách nhau bằng dấu cách. Nếu chỉ sử dụng hàm `split()` thông thường, BM25 sẽ không hiểu được các từ ghép (vd: "nghỉ phép"), dẫn đến kết quả tìm kiếm kém chính xác.

**Cách giải quyết:**  
Sử dụng thư viện `underthesea` với hàm `word_tokenize` để thực hiện segmentation trước khi đưa vào index BM25. Giúp hệ thống nhận diện đúng các từ khóa chuyên ngành trong văn bản pháp quy.

### Khó khăn 2: Chuẩn hóa và kết hợp điểm số từ các nguồn tìm kiếm khác nhau

Điểm số từ Dense Search (Cosine Similarity, thường từ 0-1) và BM25 (điểm số không giới hạn) có thang đo hoàn toàn khác nhau. Việc cộng trực tiếp hoặc lấy trung bình các điểm số này để xếp hạng là không khả thi và thiếu chính xác.

**Cách giải quyết:**  
Triển khai thuật toán **Reciprocal Rank Fusion (RRF)**. Thay vì dùng điểm số thô, thuật toán này sử dụng thứ hạng (rank) của các kết quả để tính toán điểm số mới. Cách tiếp cận này giúp việc kết hợp các phương thức tìm kiếm trở nên khách quan và không phụ thuộc vào thang đo của từng model.

### Khó khăn 3: Đảm bảo tính ổn định của Enrichment Pipeline

Phần HyQA (Module 5) phụ thuộc hoàn toàn vào API bên thứ ba (OpenAI). Nếu API gặp lỗi hoặc môi trường thiếu thư viện, toàn bộ quá trình xử lý dữ liệu và đánh giá hệ thống sẽ bị dừng lại.

**Cách giải quyết:**  
Xây dựng cấu trúc hàm có khả năng tự phục hồi (self-healing) với cơ chế fallback. Nếu việc gọi LLM thất bại, hệ thống sẽ tự động chuyển sang sử dụng các câu hỏi heuristic dựa trên nội dung văn bản. Điều này đảm bảo pipeline luôn chạy thông suốt ngay cả trong điều kiện offline hoặc lỗi API.

---

## 6. Tự đánh giá theo rubric cá nhân

| Tiêu chí | Tự đánh giá |
|----------|-------------|
| Module implementation đúng logic | Excellent |
| Test pass | 100% Pass |
| Code quality | Clean & Reusable |
| TODO markers phần mình phụ trách | Đã hoàn thành hết |
| Đóng góp thêm ngoài module chính | Có (M5 & Git workflow) |

---

## 7. Kết luận

Tôi đã hoàn thành phần cá nhân **Module 2 (Hybrid Search)** với các hạng mục chính là Tách từ tiếng Việt, BM25, Dense Search và RRF.  
Ngoài ra, tôi có hỗ trợ thêm phần HyQA của Module 5 và tối ưu hóa quy trình đẩy code lên GitHub.  
Kết quả test hiện tại cho thấy phần việc tôi phụ trách hoạt động ổn định và đã được tích hợp thành công vào nhánh main của dự án.
