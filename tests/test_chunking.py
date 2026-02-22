from __future__ import annotations

from langchain_core.documents import Document

from src.ingest.chunking import chunk_documents


def test_chunking_is_deterministic() -> None:
    docs = [
        Document(page_content="A " * 500, metadata={"source_path": "x.txt", "doc_id": "x", "page": None}),
    ]

    chunks_a = chunk_documents(docs, chunk_size=120, chunk_overlap=20)
    chunks_b = chunk_documents(docs, chunk_size=120, chunk_overlap=20)

    assert [c.page_content for c in chunks_a] == [c.page_content for c in chunks_b]
    assert [c.metadata["chunk_id"] for c in chunks_a] == [c.metadata["chunk_id"] for c in chunks_b]
