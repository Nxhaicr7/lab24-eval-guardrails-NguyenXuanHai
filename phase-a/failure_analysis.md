# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | What responsibilities do organizations have under the law re | multi_context | 0.18 | 0.27 | 0.01 | 0.03 | 0.12 | C2 |
| 2 | What are the responsibilities of organizations regarding the | multi_context | 0.19 | 0.29 | 0.00 | 0.01 | 0.12 | C2 |
| 3 | What are the implications of cross-border data transfer on n | reasoning | 0.00 | 0.55 | 0.00 | 0.00 | 0.14 | C1 |
| 4 | What are the implications of cross-border data transfer on n | reasoning | 0.00 | 0.55 | 0.00 | 0.00 | 0.14 | C1 |
| 5 | What are the definitions and implications of personal data a | multi_context | 0.25 | 0.30 | 0.01 | 0.03 | 0.15 | C2 |
| 6 | What measures are implemented to protect personal data durin | reasoning | 0.00 | 0.60 | 0.00 | 0.00 | 0.15 | C1 |
| 7 | What are the difficulties and risks associated with the cros | reasoning | 0.00 | 0.60 | 0.00 | 0.00 | 0.15 | C1 |
| 8 | What are the specific compliance assessment methods and the  | reasoning | 0.00 | 0.62 | 0.00 | 0.00 | 0.16 | C1 |
| 9 | What are the difficulties and risks associated with cross-bo | reasoning | 0.00 | 0.65 | 0.00 | 0.00 | 0.16 | C1 |
| 10 | What are the risks associated with the transfer of personal  | reasoning | 0.00 | 0.65 | 0.00 | 0.00 | 0.16 | C1 |

## Clusters Identified

### Cụm C1: Lỗi nén thông tin trong câu hỏi suy luận

**Mẫu lỗi:**  
Các câu hỏi cần tổng hợp và suy luận bị mất nhiều chi tiết vì bộ sinh câu trả lời rút gọn ground truth thành phần tóm tắt ngắn, thường chỉ dựa trên câu đầu tiên. Do đó, câu trả lời không bao phủ đủ các ý cần thiết, đặc biệt với các câu hỏi yêu cầu liên kết nhiều khía cạnh như an ninh quốc gia, chuyển dữ liệu xuyên biên giới, biện pháp bảo vệ và đánh giá tuân thủ.

**Ví dụ:**
- Việc chuyển dữ liệu xuyên biên giới có tác động gì đến an ninh quốc gia, và các dịch vụ bảo vệ dữ liệu giải quyết những lo ngại này như thế nào?
- Việc chuyển dữ liệu xuyên biên giới có tác động gì đến an ninh quốc gia, đặc biệt liên quan đến các biện pháp dành cho dịch vụ bảo vệ dữ liệu?
- Những biện pháp nào được áp dụng để bảo vệ dữ liệu cá nhân trong quá trình chuyển dữ liệu xuyên biên giới, và việc tuân thủ quy định bảo vệ dữ liệu được đánh giá như thế nào?

**Đề xuất khắc phục:**
- Tăng độ sâu của bước tổng hợp câu trả lời cho các câu hỏi loại `reasoning`.
- Truy xuất từ 2 đến 3 ngữ cảnh hỗ trợ thay vì chỉ 1 ngữ cảnh đối với các prompt suy luận.
- Yêu cầu bộ sinh câu trả lời bao phủ đủ các phần của câu hỏi, không chỉ tóm tắt câu đầu tiên.
- Thêm bước kiểm tra sau sinh để đảm bảo câu trả lời có đủ các ý chính: tác động, rủi ro, biện pháp xử lý và cách đánh giá tuân thủ.

### Cụm C2: Trôi độ chính xác ngữ cảnh ở câu hỏi đa ngữ cảnh

**Mẫu lỗi:**  
Các câu hỏi loại `multi_context` thường sử dụng nhiều đoạn ngữ cảnh rộng. Khi đưa nhiều passage vào cùng lúc, các token nhiễu làm giảm độ chính xác ngữ cảnh, dù câu trả lời vẫn có thể đúng về hướng tổng quát. Vấn đề chính không hẳn là câu trả lời hoàn toàn sai, mà là ngữ cảnh được chọn chưa đủ tập trung vào phần cần trả lời.

**Ví dụ:**
- Các tổ chức có trách nhiệm gì theo pháp luật về quản lý và bảo vệ dữ liệu cá nhân trên không gian mạng, đặc biệt liên quan đến an ninh quốc gia?
- Các tổ chức có trách nhiệm gì trong việc bảo vệ dữ liệu cá nhân trong bối cảnh quy định an ninh mạng, và các trách nhiệm này liên hệ như thế nào với các quy định chung trong luật an ninh mạng?
- Định nghĩa và hệ quả của dữ liệu cá nhân theo quy định là gì, và chúng liên quan như thế nào đến việc đánh giá các biện pháp bảo vệ dữ liệu cá nhân nhạy cảm?

**Đề xuất khắc phục:**
- Thêm bộ reranker hoặc bộ lọc metadata trước khi đưa nhiều ngữ cảnh vào bước sinh câu trả lời.
- Giảm kích thước chunk hoặc nén ngữ cảnh trước khi trả lời cuối cùng.
- Tách câu hỏi đa ngữ cảnh thành các truy vấn con, sau đó tổng hợp lại câu trả lời.
- Ưu tiên các đoạn nguồn có điều khoản, định nghĩa hoặc trách nhiệm trực tiếp thay vì các đoạn mô tả chung.
- Loại bỏ các đoạn ngữ cảnh có mức liên quan thấp để cải thiện chỉ số context precision.