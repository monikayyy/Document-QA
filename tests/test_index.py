from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.index.faiss_store import build_index, load_index, save_index


def test_index_save_and_load_consistency(tmp_path: Path, fake_embeddings) -> None:
    docs = [Document(page_content="alpha beta gamma", metadata={"source_path": "a.txt", "doc_id": "a", "chunk_id": 0})]
    store = build_index(docs, fake_embeddings)
    save_index(store, tmp_path, {"chunks_indexed": 1})

    loaded = load_index(tmp_path, fake_embeddings)
    results = loaded.similarity_search("alpha", k=1)

    assert len(results) == 1
    assert "alpha" in results[0].page_content
