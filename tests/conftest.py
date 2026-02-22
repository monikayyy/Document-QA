from __future__ import annotations

import hashlib
from typing import List

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage


class DeterministicEmbeddings(Embeddings):
    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * 8
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = digest[0] % len(vec)
            vec[idx] += (digest[1] / 255.0) + 0.01
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class FakeLLM:
    def invoke(self, _messages):
        return AIMessage(content="LangChain and FAISS are used together for document Q&A. [source]")


@pytest.fixture
def fake_embeddings() -> DeterministicEmbeddings:
    return DeterministicEmbeddings()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
