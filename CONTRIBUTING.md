# Contributing

Thanks for contributing to Document Q&A (LangChain + FAISS).

## Development setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Copy environment template:
   ```bash
   cp .env.example .env
   ```

## Local checks
Run tests before submitting changes:
```bash
python -m pytest
```

## Pull requests
- Keep PRs focused and small where possible.
- Include tests for new behavior.
- Update `README.md` when CLI/API behavior changes.
- Never commit secrets (`.env`, API keys, tokens).

## Commit guidance
Use clear commit messages in imperative mood, for example:
- `Add API error handling for ingestion failures`
- `Fix score serialization in CLI output`
