from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_path: str
    doc_id: str
    page: Optional[int] = None
    chunk_id: int


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    corpus_id: str = Field(default="default")
    rebuild: bool = Field(default=False)


class IngestResponse(BaseModel):
    corpus_id: str
    files_ingested: int
    chunks_indexed: int
    index_path: str
    rebuilt: bool


class AskRequest(BaseModel):
    question: str
    corpus_id: str = Field(default="default")
    top_k: Optional[int] = Field(default=None, ge=1)


class AskResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    message: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
