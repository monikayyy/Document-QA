from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
            continue
        if path.is_dir():
            files.extend(
                p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
    return files


def _load_single_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    return []


def load_documents(paths: list[str]) -> tuple[list[Document], list[Path]]:
    files = collect_files(paths)
    documents: list[Document] = []

    for file_path in files:
        loaded_docs = _load_single_file(file_path)
        for idx, doc in enumerate(loaded_docs):
            doc.metadata = {
                **doc.metadata,
                "source_path": str(file_path),
                "doc_id": file_path.stem,
                "page": doc.metadata.get("page"),
                "source_doc_index": idx,
            }
        documents.extend(loaded_docs)

    return documents, files
