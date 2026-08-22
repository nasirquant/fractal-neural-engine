from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Global configuration for Fractal Neural Simulation Engine."""

    # Model Routing (LiteLLM)
    default_model: str = Field(default="gpt-4o-mini", description="Default LLM model for agents")
    model_provider: str = Field(default="openai", description="Model provider (openai, anthropic, etc.)")
    model_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Max tokens per completion")
    
    # LiteLLM specific settings
    litellm_fallback_models: List[str] = Field(
        default_factory=lambda: ["gpt-3.5-turbo", "claude-3-haiku-20240307"],
        description="Fallback models in order of preference"
    )
    litellm_request_timeout: int = Field(default=60, description="Request timeout in seconds")
    litellm_max_retries: int = Field(default=3, description="Max retries for failed requests")
    litellm_num_retries: int = Field(default=3, description="Number of retries")
    
    # API keys (loaded from environment)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    together_api_key: Optional[str] = Field(default=None, description="Together AI API key")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_key_prefix: str = Field(default="fnse:", description="Prefix for all Redis keys")
    redis_ttl: int = Field(default=3600, description="Default TTL for cached states (seconds)")

    # Simulation
    max_agents: int = Field(default=100, gt=0, le=1000, description="Maximum agents per simulation")
    max_ticks_per_epoch: int = Field(default=1000, gt=0, description="Maximum ticks per simulation epoch")
    tick_timeout_seconds: float = Field(default=30.0, gt=0, description="Timeout per tick")
    global_loss_threshold: float = Field(default=0.01, ge=0.0, description="Convergence threshold for global loss")
    checkpoint_interval: int = Field(default=10, gt=0, description="Checkpoint every N ticks")

    # Safety Thresholds
    max_divergence_score: float = Field(default=10.0, gt=0, description="Max allowed divergence before circuit break")
    max_state_size_bytes: int = Field(default=1_000_000, gt=0, description="Max serialized state size per agent")
    max_recursion_depth: int = Field(default=10, gt=0, description="Max recursive skill compilation depth")
    allowed_imports: List[str] = Field(
        default_factory=lambda: [
            "math", "random", "statistics", "itertools", "collections",
            "json", "re", "datetime", "typing", "dataclasses"
        ],
        description="Whitelisted imports for dynamic skill compilation"
    )
    blocked_keywords: List[str] = Field(
        default_factory=lambda: [
            "eval", "exec", "compile", "__import__", "open", "os.", "sys.",
            "subprocess", "socket", "requests", "urllib", "importlib"
        ],
        description="Blocked keywords in compiled skills"
    )

    # API
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, gt=0, le=65535, description="FastAPI port")
    api_workers: int = Field(default=1, gt=0, description="Number of uvicorn workers")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or text)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Singleton instance
settings = Settings()