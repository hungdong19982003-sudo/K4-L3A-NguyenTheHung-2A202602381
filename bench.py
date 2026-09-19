import os
import re
import sys
from pathlib import Path
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore


class RegulationSectionChunker:
    """
    Chiến lược chia nhỏ tùy chỉnh cho văn bản quy chế / dịch vụ đại học.

    Lý do thiết kế:
    Văn bản quy định đại học được biên soạn theo từng Điều / Mục (ví dụ: '## Điều 1...', '## Điều 2...').
    Mỗi điều khoản là một đơn vị ngữ nghĩa độc lập và trọn vẹn.
    Chiến lược này tách văn bản theo các đề mục cấp 2 ('## '), đồng thời gắn kèm tiêu đề tài liệu ('# ...')
    vào đầu mỗi chunk để đảm bảo mô hình embedding và retrieval giữ được ngữ cảnh cấp cao của quy chế.
    """

    def __init__(self, max_section_size: int = 600) -> None:
        self.max_section_size = max_section_size
        self.fallback = RecursiveChunker(chunk_size=max_section_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Tách tiêu đề chính của tài liệu (nếu có # Title)
        lines = text.strip().splitlines()
        doc_header = ""
        for line in lines:
            if line.startswith("# "):
                doc_header = line.strip()
                break

        # Tách các section bắt đầu bằng "## "
        raw_sections = re.split(r"(?m)(?=^##\s+)", text.strip())
        chunks: list[str] = []

        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # Nếu là phần mở đầu (chưa có ##)
            if not sec_clean.startswith("## ") and doc_header and not sec_clean.startswith("# "):
                content = f"{doc_header}\n\n{sec_clean}"
            elif not sec_clean.startswith("# ") and doc_header:
                content = f"{doc_header} > {sec_clean}"
            else:
                content = sec_clean

            if len(content) > self.max_section_size:
                chunks.extend(self.fallback.chunk(content))
            else:
                chunks.append(content)

        return chunks if chunks else [text.strip()]


def parse_markdown_file(path: Path) -> tuple[dict[str, Any], str]:
    """Tách YAML frontmatter và phần content thân bài."""
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()
            metadata = {}
            for line in fm_text.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip().strip("\"'")
            return metadata, body
    return {}, raw.strip()


def run_benchmark():
    data_dir = Path("data/dang-ky-hoc-phan-final")
    if not data_dir.exists():
        data_dir = Path("data/university")
    md_files = sorted(data_dir.glob("*.md"))

    print(f"Loaded {len(md_files)} documents from {data_dir}")

    # 1. Baseline analysis using ChunkingStrategyComparator
    print("\n=== 1. Phân tích Baseline (ChunkingStrategyComparator) ===")
    comparator = ChunkingStrategyComparator()
    baseline_sample_files = ["ueh-course-registration-regulation.md", "hcmut-course-registration-rules.md"]
    baseline_stats = {}

    for sample_name in baseline_sample_files:
        sample_path = data_dir / sample_name
        if not sample_path.exists():
            continue
        meta, body = parse_markdown_file(sample_path)
        cmp_result = comparator.compare(body, chunk_size=250)
        baseline_stats[sample_name] = cmp_result
        print(f"\nTài liệu: {sample_name}")
        for strat, s in cmp_result.items():
            print(f"  - {strat:15}: {s['count']} chunks, độ dài TB: {s['avg_length']:.1f} ký tự")

    # 2. Xây dựng 3 chiến lược
    strategies = {
        "Strategy_1_CustomSection": RegulationSectionChunker(max_section_size=500),
        "Strategy_2_Recursive": RecursiveChunker(chunk_size=300),
        "Strategy_3_FixedSize": FixedSizeChunker(chunk_size=300, overlap=50),
    }

    # 3. 5 Benchmark Queries dựa trên dữ liệu mới
    benchmark_queries = [
        {
            "id": 1,
            "query": "Theo quy định tại UEH, thời hạn để sinh viên gửi yêu cầu hủy học phần không rút học phí là trước kỳ thi kết thúc học phần bao nhiêu ngày?",
            "gold": "Trước kỳ thi kết thúc học phần 10 ngày đối với trường hợp hủy không rút học phí.",
            "target_doc": "ueh-course-registration-regulation",
            "filter": None,
        },
        {
            "id": 2,
            "query": "Sinh viên tại HCMUT đăng ký môn học qua cổng nào và những yếu tố nào có thể khiến một học phần không xuất hiện trên hệ thống?",
            "gold": "Sinh viên đăng ký trên cổng MyBK; học phần có thể không xuất hiện vì lớp đã đủ chỗ, sinh viên chưa đạt điều kiện tiên quyết hoặc chưa cập nhật điều kiện học vụ, hoặc chưa hoàn thành học phí.",
            "target_doc": "hcmut-course-registration-rules",
            "filter": None,
        },
        {
            "id": 3,
            "query": "Tại UEL, sau khi đăng ký môn học thành công sinh viên bắt buộc phải làm gì và việc hủy hoặc đổi môn tự chọn diễn ra ở đợt nào?",
            "gold": "Sau khi đăng ký thành công bắt buộc phải xuất kết quả và lưu bản có mã vạch; việc hủy môn hoặc đổi môn tự chọn được thực hiện ở đợt điều chỉnh.",
            "target_doc": "uel-course-registration-process",
            "filter": None,
        },
        {
            "id": 4,
            "query": "Theo hướng dẫn đăng ký môn học dành cho sinh viên tại UTE, quy trình đăng ký môn học gồm những giai đoạn nào?",
            "gold": "Kế hoạch đăng ký gồm hai giai đoạn: Giai đoạn 1 là Đăng ký sơ bộ và Giai đoạn 2 là Đăng ký hoàn chỉnh.",
            "target_doc": "ute-course-registration-guide",
            "filter": {"audience": "student"},
            "is_ab_test": True,
        },
        {
            "id": 5,
            "query": "Tại TDTU, sinh viên có được đăng ký những môn học chưa có trong kế hoạch học tập không và đăng ký vào thời điểm nào?",
            "gold": "Môn chưa có trong kế hoạch chỉ được đăng ký thêm ở đợt bổ sung, khi môn còn chỗ và không trùng thời khóa biểu với các môn đã đăng ký.",
            "target_doc": "tdtu-course-registration-guide",
            "filter": None,
        },
    ]

    all_results = {}

    for strat_name, chunker in strategies.items():
        store = EmbeddingStore(collection_name=f"bench_{strat_name}", embedding_fn=_mock_embed)
        docs_to_add = []
        total_chunks = 0

        for p in md_files:
            meta, body = parse_markdown_file(p)
            doc_id = p.stem
            chunks = chunker.chunk(body)
            for i, c in enumerate(chunks):
                chunk_meta = {
                    **meta,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_strategy": strat_name,
                }
                docs_to_add.append(Document(id=f"{doc_id}#{i}", content=c, metadata=chunk_meta))
            total_chunks += len(chunks)

        store.add_documents(docs_to_add)

        strat_query_results = []
        for q in benchmark_queries:
            if q.get("filter"):
                results = store.search_with_filter(q["query"], top_k=3, metadata_filter=q["filter"])
            else:
                results = store.search(q["query"], top_k=3)
            strat_query_results.append({
                "query": q,
                "top3": results,
            })
        all_results[strat_name] = {
            "total_chunks": total_chunks,
            "results": strat_query_results,
            "store": store,
        }

    # In kết quả chi tiết
    output_lines = []
    output_lines.append("================================================================")
    output_lines.append("              KẾT QUẢ BENCHMARK RETRIEVAL LAB 07 (L3A)          ")
    output_lines.append("================================================================\n")

    for strat_name, data in all_results.items():
        output_lines.append(f"\n--- CHIẾN LƯỢC: {strat_name} (Tổng số chunk: {data['total_chunks']}) ---")
        for item in data["results"]:
            q = item["query"]
            output_lines.append(f"\n[Câu hỏi #{q['id']}]: {q['query']}")
            output_lines.append(f"  Đáp án chuẩn: {q['gold']}")
            output_lines.append("  Top-3 kết quả truy xuất được:")
            for rank, res in enumerate(item["top3"], 1):
                doc_source = res["metadata"].get("doc_id")
                score = res["score"]
                preview = res["content"][:90].replace("\n", " ")
                match = "YES" if doc_source == q["target_doc"] else "NO"
                output_lines.append(f"    {rank}. [Score={score:.4f}] Doc={doc_source} (Match: {match}) | {preview}...")

    # A/B Test Query 4
    output_lines.append("\n================================================================")
    output_lines.append("             KẾT QUẢ A/B TEST TRUY XUẤT CHO CÂU HỎI #4           ")
    output_lines.append("================================================================\n")
    q4 = benchmark_queries[3]
    custom_store = all_results["Strategy_1_CustomSection"]["store"]
    res_filtered = custom_store.search_with_filter(q4["query"], top_k=3, metadata_filter={"audience": "student"})
    res_unfiltered = custom_store.search(q4["query"], top_k=3)

    output_lines.append("Câu hỏi: " + q4["query"])
    output_lines.append("\n[1] KHI CÓ BỘ LỌC METADATA (audience=student):")
    for rank, res in enumerate(res_filtered, 1):
        output_lines.append(f"  {rank}. [Score={res['score']:.4f}] Doc={res['metadata'].get('doc_id')} (Audience={res['metadata'].get('audience')})")
        output_lines.append(f"     Nội dung: {res['content'][:110].replace(chr(10), ' ')}...")

    output_lines.append("\n[2] KHI KHÔNG CÓ BỘ LỌC METADATA (Không filter):")
    for rank, res in enumerate(res_unfiltered, 1):
        output_lines.append(f"  {rank}. [Score={res['score']:.4f}] Doc={res['metadata'].get('doc_id')} (Audience={res['metadata'].get('audience')})")
        output_lines.append(f"     Nội dung: {res['content'][:110].replace(chr(10), ' ')}...")

    full_output = "\n".join(output_lines)
    Path("ket_qua_benchmark.txt").write_text(full_output, encoding="utf-8")
    print(full_output)


if __name__ == "__main__":
    run_benchmark()
