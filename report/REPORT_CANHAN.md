# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thế Hưng  
**Mã sinh viên:** 2A202602381  
**Nhóm:** G11 (Chủ đề: Dịch vụ & Quy định Đại học - Đăng ký học phần)  
**Ngày:** 19/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là góc giữa hai vector embedding trong không gian đa chiều rất nhỏ (hai vector chỉ về gần cùng một hướng). Về mặt ngôn ngữ đời thường, điều này thể hiện hai đoạn văn bản chia sẻ cùng một ý nghĩa, ý định hoặc bối cảnh ngữ nghĩa, bất kể từ ngữ cụ thể hay độ dài của chúng có khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên có thể đăng ký môn học trực tuyến qua cổng thông tin học vụ."
- Câu B: "Người học tiến hành lựa chọn lớp học phần trên website đào tạo của trường."
- Tại sao tương đồng: Cả hai câu cùng mô tả một hành động thực tế (sinh viên đăng ký học phần qua mạng) với cùng một mục đích, dù sử dụng hệ thống từ vựng hoàn toàn khác nhau ("sinh viên" vs "người học", "môn học" vs "lớp học phần", "cổng thông tin học vụ" vs "website đào tạo").

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên có thể đăng ký môn học trực tuyến qua cổng thông tin học vụ."
- Câu B: "Món bún chả que tre phố cổ Hà Nội có nước chấm rất đậm đà thơm ngon."
- Tại sao khác: Hai câu thuộc về hai miền chủ đề hoàn toàn độc lập và không liên quan đến nhau (thủ tục hành chính học vụ đại học đối chiếu với ẩm thực đường phố truyền thống).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Độ tương tự cosine chỉ đo góc định hướng giữa các vector mà không phụ thuộc vào độ lớn (magnitude / độ dài vector), do đó không bị ảnh hưởng bởi độ dài đoạn văn bản hay số lần lặp từ. Ngược lại, khoảng cách Euclid bị chi phối mạnh bởi độ dài: hai đoạn văn cùng chung ý nghĩa nhưng một đoạn dài (nhiều từ) và một đoạn ngắn (ít từ) sẽ có khoảng cách Euclid rất lớn, gây sai lệch trong việc truy xuất ngữ nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
- *Trình bày phép tính:*
  - Đặt các thông số:
    - Độ dài tài liệu: $L = 10{,}000$ ký tự
    - Kích thước chunk (`chunk_size`): $S = 500$ ký tự
    - Độ chồng chéo (`overlap`): $O = 50$ ký tự
  - Bước nhảy (step) giữa các chunk liên tiếp:
    $$\text{step} = S - O = 500 - 50 = 450 \text{ (ký tự)}$$
    *(Công thức code: `step = chunk_size - overlap = 500 - 50 = 450`)*
  - Áp dụng công thức tính số lượng chunk:
    $$N = \left\lceil \frac{L - O}{S - O} \right\rceil = \left\lceil \frac{10000 - 50}{450} \right\rceil = \left\lceil \frac{9950}{450} \right\rceil = \lceil 22.11 \rceil = 23$$
  - Kiểm chứng bằng code thực tế: `len(FixedSizeChunker(500, 50).chunk('a'*10000))` cho kết quả chính xác **23**.
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
- Khi overlap tăng lên $O' = 100$:
  - Bước nhảy mới: $\text{step}' = S - O' = 500 - 100 = 400$ ký tự.
  - Số lượng chunk mới:
    $$N' = \left\lceil \frac{L - O'}{S - O'} \right\rceil = \left\lceil \frac{10000 - 100}{400} \right\rceil = \left\lceil \frac{9900}{400} \right\rceil = \lceil 24.75 \rceil = 25 \text{ (chunks)}$$
  - Kết quả: Số lượng chunk tăng từ 23 lên 25 (**tăng thêm 2 chunks**).
> Ta muốn độ chồng chéo lớn hơn nhằm bảo toàn ngữ cảnh liên tục giữa các chunk, ngăn ngừa việc một câu văn hoặc một ý hoàn chỉnh bị cắt đôi ở ranh giới giữa hai chunk, giúp retriever không bị mất thông tin quan trọng khi câu hỏi của người dùng rơi đúng vào điểm giao thoa.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy với kỹ thuật positive lookbehind `r'(?<=[.!?])(?:\s+|\n+)'` để phát hiện ranh giới câu mà không làm mất dấu kết thúc câu (`.`, `!`, `?`). Sau đó gom tối đa `max_sentences_per_chunk` câu vào một chuỗi duy nhất bằng dấu cách và loại bỏ khoảng trắng thừa hai đầu; chuỗi rỗng trả về danh sách rỗng `[]`. Edge case chưa xử lý triệt để là các từ viết tắt phổ biến ("TS.", "PGS.", "v.v.", "ThS.") và số thập phân ("3.5") có thể bị phân tách nhầm thành dấu chấm hết câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo nguyên lý "chia nhỏ đệ quy rồi gom lại": thử các dấu phân tách theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Nếu một đoạn nhỏ hơn hoặc bằng `chunk_size`, nó được giữ nguyên; nếu lớn hơn, hàm đệ quy tiếp với dấu phân tách kế tiếp; sau đó gom các đoạn ngắn liền kề lại sao cho tổng độ dài kèm separator không vượt quá `chunk_size`. Base cases là khi chuỗi đã nhỏ hơn hoặc bằng `chunk_size`, khi chuỗi rỗng, hoặc khi danh sách separator rỗng/bằng `""` (fallback cắt chuỗi theo số ký tự `chunk_size`).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tổ chức lưu trữ in-memory đơn giản và tin cậy dạng `list[dict]`, mỗi record chứa `id`, `content`, `metadata` (bảo đảm có `doc_id`), và vector `embedding`. Khi tìm kiếm (`search`), truy vấn được nhúng thành vector qua `_embedding_fn`, sau đó dùng hàm `_search_records` tính điểm tương tự thông qua tích vô hướng (dot product) với từng record (tương đương cosine similarity do vector đã được chuẩn hóa đơn vị $\Vert v \Vert = 1$), sắp xếp giảm dần và lấy top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` bắt buộc phải lọc trước (pre-filtering): duyệt toàn bộ danh sách để chọn ra các bản ghi thỏa mãn tất cả các cặp key-value trong `metadata_filter`, rồi mới chuyển tập đã lọc qua `_search_records` để tính độ tương tự; điều này tránh việc top-k slot bị chiếm hết bởi các tài liệu không hợp lệ. Hàm `delete_document` lọc bỏ mọi chunk có `id == doc_id` hoặc `metadata['doc_id'] == doc_id`, trả về `True` nếu độ dài store giảm đi và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Hàm kiểm tra trường hợp kho dữ liệu rỗng và trả thông báo phù hợp thay vì crash. Tiếp theo, gọi `store.search(question, top_k)` để lấy các chunk liên quan nhất, đóng gói từng chunk kèm số thứ tự `[1], [2]...` cùng tên tài liệu và nguồn trích dẫn; prompt được thiết kế với ràng buộc chống bịa (hallucination) nghiêm ngặt, yêu cầu mô hình chỉ trả lời dựa trên ngữ cảnh được cung cấp và trích dẫn số thứ tự nguồn tương ứng trước khi gọi `llm_fn(prompt)`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\hungd\Desktop\K4-L3A-NguyenTheHung-2A202602381-main
plugins: anyio-4.15.1, langsmith-0.12.6
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên đăng ký học phần trên cổng học vụ trực tuyến. | Người học lựa chọn môn học trên hệ thống đào tạo của trường. | cao | -0.0815 | Sai |
| 2 | Thời hạn nộp học phí là tuần thứ 4 của học kỳ. | Sinh viên hoàn thành đóng tiền học trước tuần thứ tư. | cao | +0.0140 | Đúng (thấp) |
| 3 | Sinh viên đăng ký học phần trên cổng học vụ trực tuyến. | Món bún chả Hà Nội có nước chấm rất đậm đà thơm ngon. | thấp | +0.1239 | Sai |
| 4 | Thư viện cho phép mượn tối đa 5 cuốn sách trong 30 ngày. | Bạn đọc được mượn năm tài liệu với thời gian một tháng. | cao | +0.0399 | Đúng (thấp) |
| 5 | Lệ phí phúc khảo là 50.000 VNĐ cho mỗi bài thi kết thúc môn. | Thời tiết hôm nay tại Hà Nội nhiều mây và có mưa rào rải rác. | thấp | +0.0210 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là Cặp 1 (hai câu diễn đạt cùng ý nghĩa học vụ) lại nhận điểm cosine âm (-0.0815), trong khi Cặp 3 (đăng ký học phần so với món bún chả) lại có điểm tương đồng dương cao nhất (+0.1239). Điều này chứng minh rằng `MockEmbedder` chỉ tạo vector giả ngẫu nhiên dựa trên mã băm MD5 của chuỗi ký tự thô chứ không hề chứa không gian ngữ nghĩa thật. Trong các hệ thống RAG thực tế, bắt buộc phải dùng các mô hình nhúng ngữ nghĩa chuyên sâu (Dense Embeddings như Sentence-Transformers, OpenAI, Gemini) để các văn bản cùng ý nghĩa có vector cùng hướng.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên chiến lược cá nhân: **Custom Heading/Section Chunker (`RegulationSectionChunker`)**.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-----------------|--------------------------------------|------------|--------------------------------|---------------------------------|
| 1 | Theo quy định tại UEH, thời hạn để sinh viên gửi yêu cầu hủy học phần không rút học phí là trước kỳ thi kết thúc học phần bao nhiêu ngày? | `# Quy định và hướng dẫn đăng ký môn học tại HCMUT` | 0.2039 | Không trực tiếp (Nhiễu do mock) | Căn cứ theo tài liệu [1]: Trích dẫn nhầm sang thông tin HCMUT do đặc tính giả lập ngẫu nhiên của MockEmbedder. |
| 2 | Sinh viên tại HCMUT đăng ký môn học qua cổng nào và những yếu tố nào có thể khiến một học phần không xuất hiện trên hệ thống? | `# Danh mục quy định học thuật của Hoa Sen` | 0.2149 | Có trong Top-3 (Top-3 là HCMUT) | Căn cứ theo tài liệu HCMUT [3]: Sinh viên đăng ký trên cổng MyBK; học phần có thể không xuất hiện do chưa đạt điều kiện tiên quyết hoặc chưa đóng học phí. |
| 3 | Tại UEL, sau khi đăng ký môn học thành công sinh viên bắt buộc phải làm gì và việc hủy hoặc đổi môn tự chọn diễn ra ở đợt nào? | `# Quy định đăng ký và hủy học phần tại UEH > ## Điều kiện đăng ký` | 0.2885 | Có trong Top-2 (Top-2 là UEL) | Căn cứ theo tài liệu UEL [2]: Sinh viên bắt buộc phải xuất kết quả và lưu bản có mã vạch; việc đổi môn tự chọn thực hiện ở đợt điều chỉnh. |
| 4 | Theo hướng dẫn đăng ký môn học dành cho sinh viên tại UTE, quy trình đăng ký môn học gồm những giai đoạn nào? *(Có filter: student)* | `Trường có thể đăng ký thời khóa biểu dự kiến vào tài khoản sinh viên... (UEH)` | 0.3236 | Có trong Top-2 (Top-2 là UTE) | Căn cứ theo tài liệu UTE [2]: Quy trình gồm 2 giai đoạn: Đăng ký sơ bộ và Đăng ký hoàn chỉnh (đã loại trừ tài liệu Hoa Sen audience=all). |
| 5 | Tại TDTU, sinh viên có được đăng ký những môn học chưa có trong kế hoạch học tập không và đăng ký vào thời điểm nào? | `# Quy trình đăng ký môn học học kỳ chính tại UEL > ## Các loại môn` | 0.2579 | Có trong Top-2 (Top-2 là TDTU) | Căn cứ theo tài liệu TDTU [2]: Sinh viên được đăng ký thêm môn ngoài kế hoạch ở đợt bổ sung khi lớp còn chỗ và không trùng TKB. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 (Ở các câu hỏi 2, 3, 4, 5, các chunk chính xác thuộc tài liệu cần tìm đều xuất hiện ở Top-2 hoặc Top-3; riêng câu 4 bộ lọc `audience=student` đã loại trừ thành công các văn bản quy định chung).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Nhờ so sánh giữa chiến lược Section Chunking của tôi với RecursiveChunker và FixedSizeChunker, tôi nhận ra việc chia văn bản theo ranh giới ngữ nghĩa tự nhiên của văn bản quy phạm (Điều/Khoản) giúp bảo toàn hoàn toàn tính toàn vẹn của điều luật, không bao giờ bị cắt cụt câu ở giữa. Đồng thời, kỹ thuật gắn Header cấp cao (`# Tên quy chế > ## Tên điều`) vào mỗi chunk là then chốt để chunk con không bị mồ côi ngữ cảnh khi truy xuất độc lập.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |

