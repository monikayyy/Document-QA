from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Settings, get_settings
from src.index.faiss_store import build_index, load_index, load_manifest, save_index
from src.ingest.chunking import chunk_documents
from src.ingest.loaders import load_documents
from src.qa.chain import generate_answer
from src.retrieval.retriever import RetrievalItem, retrieve_with_scores, should_abstain
from src.schemas import Citation


@dataclass
class IngestResult:
    corpus_id: str
    files_ingested: int
    chunks_indexed: int
    index_path: str
    rebuilt: bool


class DocumentQAService:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embeddings=None,
        llm=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key or None,
        )
        self.llm = llm or ChatOpenAI(
            model=self.settings.chat_model,
            api_key=self.settings.openai_api_key or None,
        )

    def corpus_dir(self, corpus_id: str) -> Path:
        return self.settings.index_base_path / corpus_id

    def ingest(self, paths: list[str], corpus_id: str = "default", rebuild: bool = False) -> IngestResult:
        docs, files = load_documents(paths)
        if not docs:
            raise ValueError("No supported documents found to ingest")
        chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("No chunks generated from provided documents")

        corpus_dir = self.corpus_dir(corpus_id)
        vectorstore = build_index(chunks, self.embeddings)
        metadata = {
            "corpus_id": corpus_id,
            "files_ingested": len(files),
            "chunks_indexed": len(chunks),
            "source_paths": [str(p) for p in files],
        }

        save_index(vectorstore, corpus_dir, metadata)

        return IngestResult(
            corpus_id=corpus_id,
            files_ingested=len(files),
            chunks_indexed=len(chunks),
            index_path=str(corpus_dir),
            rebuilt=rebuild,
        )

    def stats(self, corpus_id: str = "default") -> dict:
        corpus_dir = self.corpus_dir(corpus_id)
        manifest = load_manifest(corpus_dir)
        return {
            "corpus_id": corpus_id,
            "index_exists": corpus_dir.exists(),
            "index_path": str(corpus_dir),
            "manifest": manifest,
        }

    def ask(self, question: str, corpus_id: str = "default", top_k: Optional[int] = None) -> dict:
        k = top_k or self.settings.top_k
        corpus_dir = self.corpus_dir(corpus_id)
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus '{corpus_id}' has no persisted index at {corpus_dir}")
        vectorstore = load_index(corpus_dir, self.embeddings)

        items = retrieve_with_scores(vectorstore, question, top_k=k)
        citations = self._citations(items)
        scores = [item.score for item in items]

        if should_abstain(
            items,
            confidence_threshold=self.settings.confidence_threshold,
            min_supporting_chunks=self.settings.min_supporting_chunks,
        ):
            return {
                "status": "abstain",
                "message": "Not enough high-confidence evidence found in indexed documents.",
                "citations": [c.model_dump() for c in citations],
                "scores": scores,
            }

        answer = generate_answer(self.llm, question, items)
        return {
            "status": "ok",
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "scores": scores,
        }

    @staticmethod
    def _citations(items: list[RetrievalItem]) -> list[Citation]:
        out: list[Citation] = []
        for item in items:
            meta = item.document.metadata
            citation = Citation(
                source_path=str(meta.get("source_path", "")),
                doc_id=str(meta.get("doc_id", "")),
                page=meta.get("page"),
                chunk_id=int(meta.get("chunk_id", -1)),
            )
            out.append(citation)
        return out
