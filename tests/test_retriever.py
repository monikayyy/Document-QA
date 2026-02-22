from __future__ import annotations

from langchain_core.documents import Document

from src.retrieval.retriever import RetrievalItem, should_abstain


def test_should_abstain_when_not_enough_support() -> None:
    items = [
        RetrievalItem(Document(page_content="x", metadata={}), score=0.9),
        RetrievalItem(Document(page_content="y", metadata={}), score=0.2),
    ]

    assert should_abstain(items, confidence_threshold=0.8, min_supporting_chunks=2)
    assert not should_abstain(items, confidence_threshold=0.8, min_supporting_chunks=1)
