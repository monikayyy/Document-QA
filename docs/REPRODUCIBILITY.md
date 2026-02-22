# Reproducibility Guide

This document provides a compact protocol for reporting results in a research-style format.

## 1. Environment Snapshot
Record:
- Python version
- OS / architecture
- package install command (`pip install -e .[dev]`)
- model versions (`EMBEDDING_MODEL`, `CHAT_MODEL`)
- retrieval parameters (`TOP_K`, `CONFIDENCE_THRESHOLD`, `MIN_SUPPORTING_CHUNKS`)

## 2. Corpus Definition
For each corpus ID, report:
- source location
- number of files ingested
- number of chunks indexed
- dominant document formats

Use:
```bash
qa stats --corpus <id>
```

## 3. Evaluation Design
Use at least three query groups:
1. In-domain factual queries (answerable from corpus)
2. Cross-document synthesis queries
3. Out-of-domain/no-evidence queries

For each group, track:
- answer status (`ok` vs `abstain`)
- citation presence
- citation relevance (manual binary check)

## 4. Error Analysis Template
For each failure case, log:
- query
- expected behavior
- actual behavior
- likely failure source (chunking, retrieval, generation, thresholding)
- proposed fix

## 5. Suggested Reporting Table
| Query Type | N | OK | Abstain | Citation Present | Correct Citation |
|-----------|---|----|---------|------------------|------------------|
| In-domain |   |    |         |                  |                  |
| Synthesis |   |    |         |                  |                  |
| OOD       |   |    |         |                  |                  |

## 6. Reproducibility Notes
- Keep corpus ID stable between runs.
- Re-run ingest when corpus contents change.
- Save `.env` configuration values used for each reported run.
