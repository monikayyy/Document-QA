from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.retrieval.retriever import RetrievalItem


def build_context(items: list[RetrievalItem]) -> str:
    lines: list[str] = []
    for item in items:
        meta = item.document.metadata
        citation = f"{meta.get('source_path')}|doc={meta.get('doc_id')}|page={meta.get('page')}|chunk={meta.get('chunk_id')}"
        lines.append(f"[CITATION] {citation}\n{item.document.page_content}")
    return "\n\n".join(lines)


def generate_answer(llm, question: str, items: list[RetrievalItem]) -> str:
    context = build_context(items)
    messages = [
        SystemMessage(
            content=(
                "You are a retrieval QA assistant. Answer only using provided context. "
                "If context is insufficient, say you do not have enough evidence. "
                "Always include short citation references from provided citation blocks."
            )
        ),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
    ]

    response = llm.invoke(messages)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)
