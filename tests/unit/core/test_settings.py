"""Tests for PIANO settings configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from piano.config.settings import (
    AgentSettings,
    BridgeSettings,
    CheckpointSettings,
    ConsolidationSettings,
    LLMSettings,
    LogSettings,
    MinecraftSettings,
    PianoSettings,
    QdrantSettings,
    RedisSettings,
    SchedulerSettings,
    get_settings,
)


class TestDefaultSettings:
    """Test that default settings are loaded correctly."""

    def test_default_redis_settings(self) -> None:
        settings = PianoSettings()
        assert settings.redis.host == "localhost"
        assert settings.redis.port == 6379
        assert settings.redis.db == 0
        assert settings.redis.password is None

    def test_default_llm_settings(self) -> None:
        settings = PianoSettings()
        assert settings.llm.default_model == "gpt-4o-mini"
        assert settings.llm.temperature == 0.0
        assert settings.llm.max_tokens == 1024
        assert settings.llm.api_key == ""
        assert settings.llm.tier_models == {
            "fast": "gpt-4o-mini",
            "mid": "gpt-4o-mini",
            "slow": "gpt-4o",
        }

    def test_default_bridge_settings(self) -> None:
        settings = PianoSettings()
        assert settings.bridge.host == "localhost"
        assert settings.bridge.base_command_port == 5555
        assert settings.bridge.base_event_port == 5556
        # Redundant command_port / event_port should not exist
        assert not hasattr(settings.bridge, "command_port")
        assert not hasattr(settings.bridge, "event_port")

    def test_default_minecraft_settings(self) -> None:
        settings = PianoSettings()
        assert settings.minecraft.host == "localhost"
        assert settings.minecraft.port == 25565

    def test_default_agent_settings(self) -> None:
        settings = PianoSettings()
        assert settings.agent.agent_id == "agent-001"
        assert settings.agent.tick_interval_ms == 500
        assert settings.agent.cc_interval_ms == 3000

    def test_default_scheduler_settings(self) -> None:
        settings = PianoSettings()
        assert settings.scheduler.fast_tick_ms == 500
        assert settings.scheduler.mid_tick_multiplier == 3
        assert settings.scheduler.slow_tick_multiplier == 10

    def test_default_log_settings(self) -> None:
        settings = PianoSettings()
        assert settings.log.level == "INFO"
        assert settings.log.format == "json"

    def test_default_qdrant_settings(self) -> None:
        settings = PianoSettings()
        assert settings.qdrant.url == "http://localhost:6333"
        assert settings.qdrant.collection_prefix == "ltm"
        assert settings.qdrant.embedding_dim == 1536

    def test_default_checkpoint_settings(self) -> None:
        settings = PianoSettings()
        assert settings.checkpoint.dir == "checkpoints"
        assert settings.checkpoint.max_count == 10
        assert settings.checkpoint.interval_seconds == 300.0

    def test_default_consolidation_settings(self) -> None:
        settings = PianoSettings()
        assert settings.consolidation.batch_size == 10
        assert settings.consolidation.min_importance == 0.3


class TestEnvironmentOverride:
    """Test that environment variables override defaults."""

    def test_redis_host_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_REDIS__HOST": "redis-server"}):
            settings = PianoSettings()
            assert settings.redis.host == "redis-server"

    def test_redis_port_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_REDIS__PORT": "6380"}):
            settings = PianoSettings()
            assert settings.redis.port == 6380

    def test_redis_password_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_REDIS__PASSWORD": "secret123"}):
            settings = PianoSettings()
            assert settings.redis.password == "secret123"

    def test_llm_api_key_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_LLM__API_KEY": "sk-test-key"}):
            settings = PianoSettings()
            assert settings.llm.api_key == "sk-test-key"

    def test_llm_model_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_LLM__DEFAULT_MODEL": "gpt-4o"}):
            settings = PianoSettings()
            assert settings.llm.default_model == "gpt-4o"

    def test_log_level_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_LOG__LEVEL": "DEBUG"}):
            settings = PianoSettings()
            assert settings.log.level == "DEBUG"

    def test_agent_tick_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_AGENT__TICK_INTERVAL_MS": "1000"}):
            settings = PianoSettings()
            assert settings.agent.tick_interval_ms == 1000

    def test_minecraft_host_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_MINECRAFT__HOST": "mc-server"}):
            settings = PianoSettings()
            assert settings.minecraft.host == "mc-server"

    def test_qdrant_url_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_QDRANT__URL": "http://qdrant:6333"}):
            settings = PianoSettings()
            assert settings.qdrant.url == "http://qdrant:6333"

    def test_qdrant_embedding_dim_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_QDRANT__EMBEDDING_DIM": "384"}):
            settings = PianoSettings()
            assert settings.qdrant.embedding_dim == 384

    def test_checkpoint_dir_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_CHECKPOINT__DIR": "/data/checkpoints"}):
            settings = PianoSettings()
            assert settings.checkpoint.dir == "/data/checkpoints"

    def test_checkpoint_interval_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_CHECKPOINT__INTERVAL_SECONDS": "600.0"}):
            settings = PianoSettings()
            assert settings.checkpoint.interval_seconds == 600.0

    def test_consolidation_batch_size_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_CONSOLIDATION__BATCH_SIZE": "20"}):
            settings = PianoSettings()
            assert settings.consolidation.batch_size == 20

    def test_consolidation_min_importance_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_CONSOLIDATION__MIN_IMPORTANCE": "0.5"}):
            settings = PianoSettings()
            assert settings.consolidation.min_importance == 0.5

    def test_multiple_overrides(self) -> None:
        env = {
            "PIANO_REDIS__HOST": "redis-prod",
            "PIANO_REDIS__PORT": "6380",
            "PIANO_LLM__DEFAULT_MODEL": "gpt-4o",
            "PIANO_LOG__LEVEL": "WARNING",
        }
        with patch.dict(os.environ, env):
            settings = PianoSettings()
            assert settings.redis.host == "redis-prod"
            assert settings.redis.port == 6380
            assert settings.llm.default_model == "gpt-4o"
            assert settings.log.level == "WARNING"


class TestEnvFileLoading:
    """Test .env file loading."""

    def test_env_file_loading(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("PIANO_REDIS__HOST=from-env-file\nPIANO_LLM__API_KEY=sk-from-file\n")
        settings = PianoSettings(_env_file=str(env_file))
        assert settings.redis.host == "from-env-file"
        assert settings.llm.api_key == "sk-from-file"

    def test_env_var_overrides_env_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("PIANO_REDIS__HOST=from-file\n")
        with patch.dict(os.environ, {"PIANO_REDIS__HOST": "from-env"}):
            settings = PianoSettings(_env_file=str(env_file))
            assert settings.redis.host == "from-env"

    def test_missing_env_file_is_ok(self) -> None:
        settings = PianoSettings(_env_file="/nonexistent/.env")
        assert settings.redis.host == "localhost"


class TestValidation:
    """Test settings validation."""

    def test_invalid_port_type(self) -> None:
        env = {"PIANO_REDIS__PORT": "not-a-number"}
        with pytest.raises(Exception), patch.dict(os.environ, env):  # noqa: B017
            PianoSettings()

    def test_extra_fields_ignored(self) -> None:
        with patch.dict(os.environ, {"PIANO_UNKNOWN__FIELD": "value"}):
            settings = PianoSettings()
            assert settings.redis.host == "localhost"


class TestGetSettings:
    """Test the get_settings helper function."""

    def test_get_settings_defaults(self) -> None:
        settings = get_settings()
        assert isinstance(settings, PianoSettings)
        assert settings.redis.host == "localhost"

    def test_get_settings_returns_piano_settings(self) -> None:
        settings = get_settings()
        assert isinstance(settings.redis, RedisSettings)
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.bridge, BridgeSettings)
        assert isinstance(settings.minecraft, MinecraftSettings)
        assert isinstance(settings.agent, AgentSettings)
        assert isinstance(settings.scheduler, SchedulerSettings)
        assert isinstance(settings.log, LogSettings)
        assert isinstance(settings.qdrant, QdrantSettings)
        assert isinstance(settings.checkpoint, CheckpointSettings)
        assert isinstance(settings.consolidation, ConsolidationSettings)


class TestNestedSettingsModels:
    """Test individual nested settings models."""

    def test_redis_settings_model(self) -> None:
        r = RedisSettings(host="my-redis", port=6380, db=1, password="pass")
        assert r.host == "my-redis"
        assert r.port == 6380
        assert r.db == 1
        assert r.password == "pass"

    def test_llm_settings_custom_tiers(self) -> None:
        custom_tiers = {"fast": "gpt-4o-mini", "mid": "gpt-4o", "slow": "gpt-4o"}
        llm = LLMSettings(tier_models=custom_tiers)
        assert llm.tier_models["mid"] == "gpt-4o"

    def test_qdrant_settings_model(self) -> None:
        q = QdrantSettings(url="http://qdrant:6333", collection_prefix="test", embedding_dim=384)
        assert q.url == "http://qdrant:6333"
        assert q.collection_prefix == "test"
        assert q.embedding_dim == 384

    def test_checkpoint_settings_model(self) -> None:
        cp = CheckpointSettings(dir="/data/cp", max_count=5, interval_seconds=120.0)
        assert cp.dir == "/data/cp"
        assert cp.max_count == 5
        assert cp.interval_seconds == 120.0

    def test_consolidation_settings_model(self) -> None:
        c = ConsolidationSettings(batch_size=20, min_importance=0.5)
        assert c.batch_size == 20
        assert c.min_importance == 0.5


class TestBridgeSettingsExtended:
    """Test extended BridgeSettings fields and methods."""

    def test_bridge_settings_new_defaults(self) -> None:
        b = BridgeSettings()
        assert b.base_command_port == 5555
        assert b.base_event_port == 5556
        assert b.connect_timeout_s == 30.0
        assert b.connect_retry_count == 5
        assert b.bot_name_prefix == "PIANOBot"
        assert b.perception_interval_ms == 1000

    def test_bridge_settings_base_ports(self) -> None:
        b = BridgeSettings()
        assert b.base_command_port == 5555
        assert b.base_event_port == 5556

    def test_bridge_settings_connect_timeout(self) -> None:
        b = BridgeSettings()
        assert b.connect_timeout_s == 30.0

    def test_bridge_settings_retry_count(self) -> None:
        b = BridgeSettings()
        assert b.connect_retry_count == 5

    def test_bridge_settings_bot_prefix(self) -> None:
        b = BridgeSettings()
        assert b.bot_name_prefix == "PIANOBot"

    def test_bridge_settings_perception_interval(self) -> None:
        b = BridgeSettings()
        assert b.perception_interval_ms == 1000

    def test_bridge_get_ports_for_index(self) -> None:
        b = BridgeSettings()
        assert b.get_ports_for_index(0) == (5555, 5556)
        assert b.get_ports_for_index(3) == (5561, 5562)

    def test_bridge_settings_env_override(self) -> None:
        with patch.dict(os.environ, {"PIANO_BRIDGE__BASE_COMMAND_PORT": "6000"}):
            settings = PianoSettings()
            assert settings.bridge.base_command_port == 6000
