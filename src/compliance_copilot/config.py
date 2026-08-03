from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"

    # Database. Host-side port is 5433 (not 5432) to avoid colliding with a
    # locally-installed Postgres; inside docker-compose the api service talks to
    # db:5432 directly, unaffected by this host-side remap.
    database_url: str = (
        "postgresql+asyncpg://compliance:compliance@localhost:5433/compliance_copilot"
    )

    # LLM provider (agent reasoning nodes — DeepSeek's OpenAI-compatible API)
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Embeddings
    embedding_model_name: str = "BAAI/bge-m3"

    # Tracing
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Retrieval tunables — swept against the eval harness (scripts/tune_retrieval.py),
    # not chosen by intuition. Re-tuned after fixing a real bug in hybrid.py:
    # sparse-only discoveries used to have their dense_score fabricated as 0.0
    # instead of their real cosine similarity looked up, which silently
    # floor-filtered genuinely relevant chunks and made fusion_weight
    # practically inert. The first tuning pass (top_k=5, floor=0.35) was
    # measured against that buggy behavior and is now superseded.
    #
    # Current choice (fusion=0.5, top_k=10, floor=0.45): recall_at_k 0.125 ->
    # 0.250 (doubled — real matches the old bug discarded are now found) with
    # only a small give-back on regulation_set_f1 (0.726 -> 0.712) and
    # false_positive_rate (0.524 -> 0.571) versus the pre-bug-fix baseline at
    # the same top_k/floor. Recall was the most persistently weak metric
    # across every phase of this project; a config that doubles it for a
    # single-digit cost elsewhere is the better trade, especially since the
    # agent's own applicability reasoning is a second, independent filter
    # this retrieval-only harness can't credit — raw retrieval FPR overstates
    # the agent's real false-positive rate.
    retrieval_top_k: int = Field(default=10, gt=0)
    retrieval_fusion_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="RRF weight favoring dense (1.0) vs sparse (0.0) retrieval",
    )
    retrieval_relevance_floor: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Below this fused score, retrieval returns empty instead of weak matches",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
