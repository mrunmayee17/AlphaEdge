"""Application settings — loaded from .env, validated at startup."""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at project root (one level above backend/)
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_PATH), env_file_encoding="utf-8")

    # NVIDIA Nemotron API — required
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_api_key: str
    nvidia_model: str = "nvidia/nemotron-3-super-120b-a12b"

    # Redis — required
    redis_url: str
    redis_session_ttl_seconds: int = 86400  # 24 hours

    # Brave Search — required
    brave_api_key: str

    # Bright Data (Reddit) — required
    brightdata_api_key: str
    brightdata_dataset_id: str = "gd_lvz8ah06191smkebj4"

    # FRED — required
    fred_api_key: str

    # Chronos-2 time-series foundation model (HuggingFace ID)
    chronos_model_id: str = "amazon/chronos-bolt-base"

    # W&B
    wandb_api_key: str = ""
    wandb_entity: str = ""
    wandb_project: str = "bam-alpha"

    # Observability
    otel_endpoint: str = "http://localhost:4317"

    @field_validator("nvidia_api_key", "redis_url", "brave_api_key", "fred_api_key", "brightdata_api_key")
    @classmethod
    def must_not_be_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must be set — no defaults, no empty values")
        return v


def get_settings() -> Settings:
    """Singleton-style settings loader."""
    return Settings()
