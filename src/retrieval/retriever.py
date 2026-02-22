from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


@dataclass
class RetrievalItem:
    document: Document
    score: float


def _normalize_distance(distance: float) -> float:
    return float(1.0 / (1.0 + max(float(distance), 0.0)))


def retrieve_with_scores(vectorstore: FAISS, query: str, top_k: int) -> list[RetrievalItem]:
    pairs = vectorstore.similarity_search_with_score(query, k=top_k)
    items: list[RetrievalItem] = []
    for doc, score in pairs:
        normalized = round(_normalize_distance(score), 6)
        items.append(RetrievalItem(document=doc, score=float(normalized)))
    return items


def should_abstain(
    items: list[RetrievalItem],
    confidence_threshold: float,
    min_supporting_chunks: int,
) -> bool:
    strong_items = [item for item in items if item.score >= confidence_threshold]
    return len(strong_items) < min_supporting_chunks
