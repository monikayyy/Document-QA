from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


MANIFEST_FILE = "manifest.json"


def build_index(documents: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(documents, embeddings)


def save_index(vectorstore: FAISS, corpus_dir: Path, metadata: dict) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(corpus_dir))
    manifest_path = corpus_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_index(corpus_dir: Path, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(
        str(corpus_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_manifest(corpus_dir: Path) -> dict:
    manifest_path = corpus_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
