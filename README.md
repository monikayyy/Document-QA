# Document Q&A System (LangChain + FAISS)

## Abstract
This project implements a retrieval-augmented document question answering system for local corpora. The pipeline combines semantic chunking, dense vector indexing (FAISS), and LLM-based answer synthesis constrained by retrieved context. The system exposes three interfaces (CLI, REST API, and browser UI) and includes confidence-based abstention to reduce unsupported responses.

## Motivation
Large language models are strong at synthesis but unreliable when asked to reason over unseen private documents. This repository explores a practical RAG baseline for:
- grounded responses with explicit citations,
- reproducible local indexing workflows,
- clear failure behavior when evidence is weak.

## System Overview
The end-to-end flow is:
1. Ingest files (`PDF`, `TXT`, `MD`).
2. Split text into overlapping chunks.
3. Embed chunks and persist FAISS index by corpus ID.
4. Retrieve top-k chunks for a query.
5. Apply confidence threshold and minimum-support rule.
6. Either return an evidence-grounded answer or abstain.

## Key Features
- Multi-format ingestion (`PDF`, `TXT`, `MD`)
- Corpus-scoped persistence (`data/index/<corpus_id>`)
- Citation metadata (`source_path`, `doc_id`, `page`, `chunk_id`)
- Confidence-aware abstention path
- CLI + API + browser interface

## Repository Layout
- `/src/service.py`: orchestration for ingest/query lifecycle
- `/src/api/app.py`: FastAPI app (`/`, `/health`, `/ingest`, `/ask`)
- `/src/cli/main.py`: command-line interface
- `/src/ingest/*`: loaders and chunking logic
- `/src/index/*`: FAISS storage and manifest management
- `/src/retrieval/*`: retrieval/scoring/abstention logic
- `/src/qa/*`: answer-generation prompt chain
- `/tests/*`: unit and integration tests
- `/docs/REPRODUCIBILITY.md`: experiment and reporting guidance

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
```

Add your key to `.env`:
```bash
OPENAI_API_KEY=...
```

## Usage
### CLI
```bash
qa ingest --path data/raw --corpus default
qa ask --corpus default --question "What is this corpus about?"
qa stats --corpus default
```

### API + Web UI
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
- Web UI: `GET /`
- Health: `GET /health`
- Ingest: `POST /ingest`
- Ask: `POST /ask`

## Reproducibility and Evaluation
For structured experiment logging (datasets, parameter settings, abstention thresholds, qualitative errors), see:
- `/docs/REPRODUCIBILITY.md`

Recommended minimum evaluation includes:
- in-domain factoid queries,
- out-of-domain queries (abstention checks),
- citation correctness spot checks.

## Development
```bash
python -m pytest
```

## Security
- `.env` is gitignored; do not commit API keys.
- Rotate keys immediately after exposure.
- See `/SECURITY.md` for disclosure policy.

## License
This project is released under the MIT License. See `/LICENSE`.
