from __future__ import annotations

from fastapi.testclient import TestClient

import src.api.app as api_module


class StubService:
    def stats(self, _corpus_id):
        return {"index_exists": True}

    def ingest(self, paths, corpus_id, rebuild):
        return type(
            "IngestResult",
            (),
            {
                "__dict__": {
                    "corpus_id": corpus_id,
                    "files_ingested": len(paths),
                    "chunks_indexed": 2,
                    "index_path": "data/index/default",
                    "rebuilt": rebuild,
                }
            },
        )()

    def ask(self, question, corpus_id, top_k=None):
        if question == "missing":
            raise FileNotFoundError("index missing")
        return {
            "status": "ok",
            "answer": "answer",
            "citations": [{"source_path": "a.txt", "doc_id": "a", "page": None, "chunk_id": 0}],
            "scores": [0.9],
        }


def test_api_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "service", StubService())
    client = TestClient(api_module.app)

    home = client.get("/")
    assert home.status_code == 200
    assert "Document Q&A" in home.text

    health = client.get("/health")
    assert health.status_code == 200

    ingest = client.post("/ingest", json={"paths": ["data/raw"], "corpus_id": "default", "rebuild": False})
    assert ingest.status_code == 200
    assert ingest.json()["files_ingested"] == 1

    ask = client.post("/ask", json={"question": "hello", "corpus_id": "default", "top_k": 2})
    assert ask.status_code == 200
    assert ask.json()["status"] == "ok"
