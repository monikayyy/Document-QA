from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.service import DocumentQAService


def test_ingest_and_ask_success(tmp_path: Path, fake_embeddings, fake_llm) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "doc.md").write_text("LangChain works with FAISS for retrieval.", encoding="utf-8")

    settings = Settings(
        openai_api_key="",
        index_base_path=tmp_path / "index",
        confidence_threshold=0.1,
        min_supporting_chunks=1,
    )
    service = DocumentQAService(settings=settings, embeddings=fake_embeddings, llm=fake_llm)

    ingest_result = service.ingest(paths=[str(raw)], corpus_id="default")
    ask_result = service.ask("What tools are used for retrieval?", corpus_id="default", top_k=2)

    assert ingest_result.files_ingested == 1
    assert ask_result["status"] == "ok"
    assert len(ask_result["citations"]) >= 1


def test_ask_abstains_on_weak_evidence(tmp_path: Path, fake_embeddings, fake_llm) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    (raw / "doc.txt").write_text("The launch date is April 4, 2028.", encoding="utf-8")

    settings = Settings(
        openai_api_key="",
        index_base_path=tmp_path / "index",
        confidence_threshold=0.95,
        min_supporting_chunks=2,
    )
    service = DocumentQAService(settings=settings, embeddings=fake_embeddings, llm=fake_llm)

    service.ingest(paths=[str(raw)], corpus_id="default")
    result = service.ask("What is the weather in Tokyo?", corpus_id="default", top_k=2)

    assert result["status"] == "abstain"
    assert "message" in result
    assert len(result["citations"]) >= 1
