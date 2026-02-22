from __future__ import annotations

from pathlib import Path

from src.ingest.loaders import collect_files


def test_collect_files_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("hello", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("not real pdf", encoding="utf-8")
    (tmp_path / "d.csv").write_text("skip", encoding="utf-8")

    files = collect_files([str(tmp_path)])
    names = {f.name for f in files}

    assert "a.txt" in names
    assert "b.md" in names
    assert "c.pdf" in names
    assert "d.csv" not in names
