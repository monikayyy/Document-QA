from __future__ import annotations

import json

import typer

from src.service import DocumentQAService

app = typer.Typer(help="Document Q&A CLI")


@app.command("ingest")
def ingest(
    path: list[str] = typer.Option(..., "--path", help="File or directory path. Can be provided multiple times."),
    corpus: str = typer.Option("default", "--corpus"),
    rebuild: bool = typer.Option(False, "--rebuild"),
) -> None:
    service = DocumentQAService()
    result = service.ingest(paths=path, corpus_id=corpus, rebuild=rebuild)
    typer.echo(json.dumps(result.__dict__, indent=2))


@app.command("ask")
def ask(
    corpus: str = typer.Option("default", "--corpus"),
    question: str = typer.Option(..., "--question"),
    top_k: int = typer.Option(4, "--top-k"),
) -> None:
    service = DocumentQAService()
    result = service.ask(question=question, corpus_id=corpus, top_k=top_k)
    typer.echo(json.dumps(result, indent=2))


@app.command("stats")
def stats(corpus: str = typer.Option("default", "--corpus")) -> None:
    service = DocumentQAService()
    result = service.stats(corpus_id=corpus)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
