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
    hf_token: str = ""

    # Fine-tuned FinCast LoRA runtime
    fincast_checkpoint_path: str = ""
    fincast_checkpoint_url: str = ""
    fincast_adapter_dir: str = ""
    fincast_results_zip_path: str = ""
    fincast_results_zip_url: str = ""
    fincast_adapter_subdir: str = "lora_adapter_best"
    fincast_extract_dir: str = str(Path(__file__).resolve().parent.parent.parent / "models" / "fincast_runtime")
    fincast_download_timeout_seconds: int = 1800
    fincast_device: str = "cpu"
    fincast_context_length: int = 128
    fincast_step_horizon: int = 5
    fincast_num_experts: int = 4
    fincast_gating_top_n: int = 2

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
