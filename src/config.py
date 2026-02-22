from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="")
    embedding_model: str = Field(default="text-embedding-3-small")
    chat_model: str = Field(default="gpt-4o-mini")

    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)

    top_k: int = Field(default=4, ge=1)
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    min_supporting_chunks: int = Field(default=1, ge=1)

    index_base_path: Path = Field(default=Path("data/index"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
