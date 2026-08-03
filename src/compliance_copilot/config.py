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

    # Retrieval tunables — swept against the eval harness in Phase 5 (see
    # scripts/tune_retrieval.py), not chosen by intuition. top_k=5 measurably
    # beat top_k=10 (false_positive_rate 0.524 -> 0.190, regulation_set_f1
    # 0.726 -> 0.757, recall unchanged) — a smaller top_k leaves less room for
    # generic boilerplate preamble text (nearly identical across all five
    # regulations) to sneak into the merged result set.
    #
    # fusion_weight is currently inert at the shipped relevance_floor: the
    # floor is checked against dense_score specifically (see hybrid.py), so
    # any chunk sparse search favors but dense search doesn't independently
    # rank gets floor-filtered out regardless of fusion_weight — confirmed by
    # re-sweeping at floor=0.0, where fusion_weight did change every metric.
    # Left at the neutral 0.5 default: no measured downside at this floor, and
    # if the floor's dense-only anchoring is ever revisited, 1.0 (dense-heavy)
    # looked best on precision/F1/FPR in the floor=0.0 sweep, at some cost to
    # recall vs. 0.5.
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_fusion_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="RRF weight favoring dense (1.0) vs sparse (0.0) retrieval",
    )
    retrieval_relevance_floor: float = Field(
        default=0.35,
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
