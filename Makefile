PYTHON ?= python

.PHONY: install test run-api run-cli-ingest run-cli-ask

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest

run-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000

run-cli-ingest:
	$(PYTHON) -m src.cli.main ingest --path data/raw --corpus default

run-cli-ask:
	$(PYTHON) -m src.cli.main ask --corpus default --question "What is this corpus about?"
