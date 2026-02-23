"""PIANO configuration management using pydantic-settings.

Loads settings from environment variables (with PIANO_ prefix) and .env files.
Nested settings use '__' as delimiter (e.g., PIANO_REDIS__PORT=6380).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseModel):
    """Redis connection settings."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


class LLMSettings(BaseModel):
    """LLM provider settings."""

    default_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str = ""
    tier_models: dict[str, str] = Field(
        default_factory=lambda: {
            "fast": "gpt-4o-mini",
            "mid": "gpt-4o-mini",
            "slow": "gpt-4o",
        }
    )
    cost_limit_usd: float = 100.0
    calls_per_minute_limit: int = 100


class BridgeSettings(BaseModel):
    """ZMQ bridge connection settings."""

    host: str = "localhost"
    command_port: int = 5555
    event_port: int = 5556


class MinecraftSettings(BaseModel):
    """Minecraft server connection settings."""

    host: str = "localhost"
    port: int = 25565


class AgentSettings(BaseModel):
    """Per-agent default settings."""

    agent_id: str = "agent-001"
    tick_interval_ms: int = 500
    cc_interval_ms: int = 3000


class SchedulerSettings(BaseModel):
    """Module scheduler tick settings."""

    fast_tick_ms: int = 500
    mid_tick_multiplier: int = 3
    slow_tick_multiplier: int = 10


class QdrantSettings(BaseModel):
    """Qdrant vector database settings for LTM."""

    url: str = "http://localhost:6333"
    collection_prefix: str = "ltm"
    embedding_dim: int = 1536


class LocalLLMSettings(BaseModel):
    """Local LLM inference server settings."""

    provider: str = "ollama"  # "ollama" or "vllm"
    url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout: float = 30.0


class CheckpointSettings(BaseModel):
    """Agent state checkpoint settings."""

    dir: str = "checkpoints"
    max_count: int = 10
    interval_seconds: float = 300.0  # 5 minutes


class ConsolidationSettings(BaseModel):
    """Memory consolidation (STM -> LTM) settings."""

    batch_size: int = 10
    min_importance: float = 0.3


class LogSettings(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "json"


class ObservabilitySettings(BaseModel):
    """Observability stack settings (metrics, tracing, logging)."""

    metrics_enabled: bool = True
    tracing_enabled: bool = True
    log_format: str = "json"
    prometheus_port: int = 9090


class ScalingSettings(BaseModel):
    """Scaling and worker management settings."""

    agents_per_worker: int = 250
    max_workers: int = 8
    num_shards: int = 4
    sharding_strategy: str = "consistent_hash"
    health_check_interval: float = 30.0


class VelocitySettings(BaseModel):
    """Velocity proxy settings."""

    proxy_host: str = "localhost"
    proxy_port: int = 25577
    max_players_per_server: int = 200
    strategy: str = "least_connections"


class ResourceLimiterSettings(BaseModel):
    """Per-agent and per-worker resource limiter settings."""

    max_concurrent_llm_per_agent: int = 3
    max_concurrent_llm_per_worker: int = 50
    max_memory_per_agent_mb: int = 64


class PromptCacheSettings(BaseModel):
    """Prompt cache settings."""

    max_entries: int = 1000
    ttl_seconds: float = 3600.0
    default_threshold: float = 0.95
    enable_semantic: bool = True


class MultiProviderSettings(BaseModel):
    """Multi-provider LLM router settings."""

    routing_strategy: str = "weighted_round_robin"


class PianoSettings(BaseSettings):
    """Root settings for the PIANO system.

    Settings are loaded from environment variables with the PIANO_ prefix
    and from .env files. Nested settings use '__' as delimiter.

    Examples:
        PIANO_REDIS__HOST=redis-server
        PIANO_LLM__API_KEY=sk-xxx
        PIANO_LOG__LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="PIANO_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    bridge: BridgeSettings = Field(default_factory=BridgeSettings)
    minecraft: MinecraftSettings = Field(default_factory=MinecraftSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    log: LogSettings = Field(default_factory=LogSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    local_llm: LocalLLMSettings = Field(default_factory=LocalLLMSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    consolidation: ConsolidationSettings = Field(default_factory=ConsolidationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    scaling: ScalingSettings = Field(default_factory=ScalingSettings)
    velocity: VelocitySettings = Field(default_factory=VelocitySettings)
    resource_limiter: ResourceLimiterSettings = Field(default_factory=ResourceLimiterSettings)
    prompt_cache: PromptCacheSettings = Field(default_factory=PromptCacheSettings)
    multi_provider: MultiProviderSettings = Field(default_factory=MultiProviderSettings)


def get_settings(**overrides: object) -> PianoSettings:
    """Create a PianoSettings instance with optional overrides."""
    return PianoSettings(**overrides)
