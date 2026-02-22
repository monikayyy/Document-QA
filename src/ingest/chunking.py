from __future__ import annotations

from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    chunk_counter: dict[str, int] = defaultdict(int)
    chunks: list[Document] = []

    for chunk in raw_chunks:
        source_key = f"{chunk.metadata.get('source_path', '')}:{chunk.metadata.get('page', -1)}"
        chunk_id = chunk_counter[source_key]
        chunk_counter[source_key] += 1

        chunk.metadata = {
            **chunk.metadata,
            "chunk_id": chunk_id,
        }
        chunks.append(chunk)

    return chunks
