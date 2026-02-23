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


class LogSettings(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "json"


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


def get_settings(**overrides: object) -> PianoSettings:
    """Create a PianoSettings instance with optional overrides."""
    return PianoSettings(**overrides)
