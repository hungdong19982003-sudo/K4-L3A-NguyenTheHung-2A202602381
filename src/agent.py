from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin trong cơ sở tri thức (kho dữ liệu rỗng)."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy tài liệu phù hợp để trả lời câu hỏi."

        context_blocks = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source_url") or r.get("metadata", {}).get("source") or r.get("id")
            title = r.get("metadata", {}).get("title") or r.get("id")
            context_blocks.append(f"[{i}] {title} ({source}):\n{r['content']}")

        context = "\n\n".join(context_blocks)
        prompt = (
            "Bạn là trợ lý học vụ đại học. Hãy trả lời câu hỏi dựa trên các đoạn ngữ cảnh trích xuất dưới đây. "
            "Chỉ sử dụng thông tin được cung cấp, trích dẫn số thứ tự nguồn [1], [2] nếu có, và không suy đoán quy định ngoài ngữ cảnh.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )
        return self.llm_fn(prompt)
