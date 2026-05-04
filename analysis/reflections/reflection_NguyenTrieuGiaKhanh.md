# Individual Reflection — Lab 18

**Họ và tên:** Nguyễn Triệu Gia Khánh  
**Mã sinh viên:** 2A202600225  
**Phần phụ trách chính:** Module 4 (RAGAS Evaluation)  
**Phần hỗ trợ thêm:** Module 5 (extract metadata và enrichment pipeline)

---

## 1. Tổng quan đóng góp

Trong bài lab này, tôi phụ trách phần cá nhân là **Module 4: RAGAS Evaluation**. Mục tiêu của phần này là xây dựng pipeline đánh giá cho hệ thống RAG, bao gồm tính các metric chính, phân tích lỗi theo Diagnostic Tree, và lưu báo cáo để nhóm có thể dùng trong phần tổng hợp.

Ngoài phần M4, tôi cũng hỗ trợ thêm một phần ở **Module 5**, cụ thể là:
- `extract_metadata()`
- `enrich_chunks()`
- các test `test_extract_metadata_*`
- các test `test_enrich_*` trong `tests/test_m5.py`

---

## 2. Công việc đã thực hiện

### Module 4: `src/m4_eval.py`

Các phần tôi đã triển khai:
- Implement `evaluate_ragas()` để trả về đầy đủ 4 metric:
  - `faithfulness`
  - `answer_relevancy`
  - `context_precision`
  - `context_recall`
- Bổ sung `per_question` để lưu kết quả đánh giá theo từng câu hỏi dưới dạng `EvalResult`.
- Thiết kế cơ chế **fallback heuristic scoring** khi môi trường chưa có đủ điều kiện để gọi RAGAS/LLM thật. Điều này giúp test vẫn chạy ổn định và pipeline không bị gãy trong lúc phát triển.
- Implement `failure_analysis()` để:
  - tính điểm trung bình mỗi câu hỏi
  - lấy `bottom_n` câu hỏi tệ nhất
  - xác định `worst_metric`
  - map sang `diagnosis` và `suggested_fix` theo đúng tinh thần rubric và Diagnostic Tree
- Cải thiện `load_test_set()` để đọc được `test_set.json` an toàn hơn, kể cả khi file có lỗi dấu phẩy cuối.
- Giữ `save_report()` tương thích với cấu trúc báo cáo JSON của pipeline nhóm.

### Module 5: `src/m5_enrichment.py`

Các phần tôi hỗ trợ thêm:
- Implement `extract_metadata()` theo hướng heuristic ổn định:
  - suy ra `topic`
  - trích xuất `entities`
  - phân loại `category`
  - nhận diện `language`
- Implement `enrich_chunks()` để ghép pipeline enrichment:
  - preserve `original_text`
  - merge metadata gốc với metadata tự động
  - build `enriched_text` từ các bước được bật như `contextual`, `summary`, `hyqa`, `metadata`

---

## 3. Kết quả test

### Kết quả Module 4

Lệnh chạy:

```bash
pytest tests/test_m4.py -v
```

Kết quả:

```text
tests/test_m4.py::test_load_test_set PASSED
tests/test_m4.py::test_evaluate_returns_metrics PASSED
tests/test_m4.py::test_failure_analysis_returns PASSED
tests/test_m4.py::test_failure_has_diagnosis PASSED
tests/test_m4.py::test_failure_analysis_maps_worst_metric PASSED
tests/test_m4.py::test_save_report_writes_expected_shape PASSED

6 passed in 0.05s
```

Ý nghĩa:
- `load_test_set()` đọc được test set
- `evaluate_ragas()` trả về đúng cấu trúc metric
- `failure_analysis()` trả về diagnosis và suggested fix
- `save_report()` ghi được file report đúng shape

### Kết quả Module 5 liên quan phần tôi làm

Lệnh chạy:

```bash
pytest tests/test_m5.py -v
```

Kết quả:

```text
16 passed in 0.04s
```

Các test liên quan trực tiếp phần tôi làm đều pass:
- `test_extract_metadata_returns_dict`
- `test_extract_metadata_returns_expected_keys`
- `test_extract_metadata_detects_hr_topic_and_language`
- `test_extract_metadata_extracts_entities_for_numeric_policies`
- `test_extract_metadata_detects_it_category`
- `test_enrich_chunks_returns_list`
- `test_enrich_type_and_length`
- `test_enrich_preserves_original_and_source_metadata`
- `test_enrich_contextual_keeps_original_in_enriched_text`
- `test_enrich_full_runs_all_techniques`

---

## 4. Kiến thức học được

- Tôi hiểu rõ hơn cách đánh giá một hệ thống RAG bằng các metric tách biệt giữa chất lượng câu trả lời và chất lượng retrieval.
- Tôi học được rằng trong quá trình phát triển, nên có **fallback implementation** để code vẫn test được ngay cả khi thiếu API key hoặc dependency bên ngoài.
- Tôi thấy `failure_analysis()` rất quan trọng vì điểm số tổng hợp chỉ cho biết hệ thống tốt hay chưa, còn diagnosis mới chỉ ra cần sửa ở chunking, search, rerank hay prompt.
- Ở phần M5, tôi hiểu thêm giá trị của enrichment trước bước embedding, đặc biệt là metadata extraction để hỗ trợ filter và tăng precision khi retrieve.

---

## 5. Khó khăn và cách giải quyết

### Khó khăn 1: Phụ thuộc vào dịch vụ bên ngoài

Phần `evaluate_ragas()` theo thiết kế chuẩn có thể cần mô hình và môi trường phù hợp để chạy đầy đủ. Nếu phụ thuộc hoàn toàn vào API hoặc RAGAS thật thì trong quá trình code và test sẽ dễ phát sinh lỗi môi trường.

**Cách giải quyết:**  
Tôi xây dựng hướng xử lý hai tầng:
- nếu có điều kiện thì dùng RAGAS thật
- nếu chưa có API key hoặc môi trường chưa sẵn sàng thì dùng heuristic fallback

Cách làm này giúp phần M4 vẫn đúng interface, dễ tích hợp nhóm, và pass test ổn định.

### Khó khăn 2: Test set trong repo còn thô

`test_set.json` hiện tại là placeholder và có lỗi dấu phẩy cuối, nên đọc JSON trực tiếp sẽ fail.

**Cách giải quyết:**  
Tôi làm `load_test_set()` robust hơn bằng cách thử parse trực tiếp trước, nếu lỗi thì sanitize chuỗi JSON rồi parse lại.

---

## 6. Tự đánh giá theo rubric cá nhân

| Tiêu chí | Tự đánh giá |
|----------|-------------|
| Module implementation đúng logic | Hoàn thành tốt |
| Test pass | `tests/test_m4.py`: 6/6 pass |
| Code quality | Code có type hints, helper functions rõ ràng, tách logic hợp lý |
| TODO markers phần M4 | Đã hoàn thành |
| Đóng góp thêm ngoài module chính | Có hỗ trợ thêm phần M5 |

---

## 7. Kết luận

Tôi đã hoàn thành phần cá nhân **M4** với đầy đủ các thành phần cốt lõi: evaluation, failure analysis và report output. Ngoài ra tôi còn hỗ trợ thêm phần **M5** ở nhánh metadata extraction và enrichment pipeline. Kết quả test hiện tại cho thấy phần việc tôi phụ trách hoạt động ổn định và sẵn sàng để nhóm ghép vào pipeline chung.
