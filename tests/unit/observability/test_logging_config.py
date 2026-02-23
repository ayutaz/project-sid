"""Tests for PIANO structured logging configuration (Loki-compatible)."""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import Any

import pytest
import structlog

from piano.observability.logging_config import (
    LogFilter,
    LogFormat,
    LoggingConfig,
    _add_log_level,
    _add_timestamp,
    _build_processor_chain,
    _make_context_injector,
    _make_module_level_filter,
    _rename_event_key,
    configure_piano_logging,
)

# ---------------------------------------------------------------------------
# LogFormat enum
# ---------------------------------------------------------------------------


class TestLogFormat:
    """Test LogFormat enum values."""

    def test_json_value(self) -> None:
        assert LogFormat.JSON == "json"

    def test_console_value(self) -> None:
        assert LogFormat.CONSOLE == "console"

    def test_loki_value(self) -> None:
        assert LogFormat.LOKI == "loki"

    def test_from_string(self) -> None:
        assert LogFormat("json") is LogFormat.JSON
        assert LogFormat("console") is LogFormat.CONSOLE
        assert LogFormat("loki") is LogFormat.LOKI

    def test_is_str_enum(self) -> None:
        assert isinstance(LogFormat.JSON, str)


# ---------------------------------------------------------------------------
# LogFilter
# ---------------------------------------------------------------------------


class TestLogFilter:
    """Test LogFilter processor."""

    def test_empty_filter_passes_all(self) -> None:
        f = LogFilter()
        event = {"event": "test_event", "agent_id": "agent-001"}
        result = f(None, "info", event)
        assert result == event

    def test_by_agent_passes_matching(self) -> None:
        f = LogFilter().by_agent("agent-001")
        event = {"event": "tick", "agent_id": "agent-001"}
        result = f(None, "info", event)
        assert result["agent_id"] == "agent-001"

    def test_by_agent_drops_non_matching(self) -> None:
        f = LogFilter().by_agent("agent-001")
        event = {"event": "tick", "agent_id": "agent-999"}
        with pytest.raises(structlog.DropEvent):
            f(None, "info", event)

    def test_by_agent_drops_when_no_agent_id(self) -> None:
        f = LogFilter().by_agent("agent-001")
        event = {"event": "tick"}
        with pytest.raises(structlog.DropEvent):
            f(None, "info", event)

    def test_by_agent_multiple_agents(self) -> None:
        f = LogFilter().by_agent("agent-001").by_agent("agent-002")
        event1 = {"event": "tick", "agent_id": "agent-001"}
        event2 = {"event": "tick", "agent_id": "agent-002"}
        assert f(None, "info", event1)["agent_id"] == "agent-001"
        assert f(None, "info", event2)["agent_id"] == "agent-002"

    def test_by_module_passes_matching(self) -> None:
        f = LogFilter().by_module("goal_generation")
        event = {"event": "tick_done", "module": "goal_generation"}
        result = f(None, "info", event)
        assert result["module"] == "goal_generation"

    def test_by_module_drops_non_matching(self) -> None:
        f = LogFilter().by_module("goal_generation")
        event = {"event": "tick_done", "module": "planning"}
        with pytest.raises(structlog.DropEvent):
            f(None, "info", event)

    def test_by_module_multiple(self) -> None:
        f = LogFilter().by_module("goal_generation").by_module("planning")
        event = {"event": "tick", "module": "planning"}
        result = f(None, "info", event)
        assert result["module"] == "planning"

    def test_by_level_passes_at_level(self) -> None:
        f = LogFilter().by_level("warning")
        event = {"event": "something"}
        result = f(None, "warning", event)
        assert result == event

    def test_by_level_passes_above_level(self) -> None:
        f = LogFilter().by_level("warning")
        event = {"event": "bad_thing"}
        result = f(None, "error", event)
        assert result == event

    def test_by_level_drops_below_level(self) -> None:
        f = LogFilter().by_level("warning")
        event = {"event": "debug_info"}
        with pytest.raises(structlog.DropEvent):
            f(None, "info", event)

    def test_by_level_case_insensitive(self) -> None:
        f = LogFilter().by_level("WARNING")
        event = {"event": "err"}
        result = f(None, "error", event)
        assert result == event

    def test_by_level_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown log level"):
            LogFilter().by_level("notavalidlevel")

    def test_combined_agent_and_module(self) -> None:
        f = LogFilter().by_agent("agent-001").by_module("planning")
        # Both match
        event_ok = {"event": "x", "agent_id": "agent-001", "module": "planning"}
        assert f(None, "info", event_ok) == event_ok
        # Agent mismatch
        event_bad_agent = {"event": "x", "agent_id": "agent-002", "module": "planning"}
        with pytest.raises(structlog.DropEvent):
            f(None, "info", event_bad_agent)
        # Module mismatch
        event_bad_mod = {"event": "x", "agent_id": "agent-001", "module": "goals"}
        with pytest.raises(structlog.DropEvent):
            f(None, "info", event_bad_mod)

    def test_combined_all_three(self) -> None:
        f = LogFilter().by_agent("agent-001").by_module("cc").by_level("warning")
        event = {"event": "warn", "agent_id": "agent-001", "module": "cc"}
        # Pass
        assert f(None, "warning", event) == event
        # Fail on level
        with pytest.raises(structlog.DropEvent):
            f(None, "debug", event)

    def test_active_properties_initial(self) -> None:
        f = LogFilter()
        assert f.active_agent_ids is None
        assert f.active_module_names is None
        assert f.active_min_level is None

    def test_active_properties_after_set(self) -> None:
        f = LogFilter().by_agent("a").by_module("m").by_level("error")
        assert f.active_agent_ids == {"a"}
        assert f.active_module_names == {"m"}
        assert f.active_min_level == logging.ERROR

    def test_chaining(self) -> None:
        """LogFilter methods return self for chaining."""
        f = LogFilter()
        result = f.by_agent("x").by_module("y").by_level("info")
        assert result is f


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == LogFormat.JSON
        assert cfg.output == "stdout"
        assert cfg.module_levels == {}
        assert cfg.context == {}
        assert cfg.log_filter is None

    def test_configure_level(self) -> None:
        cfg = LoggingConfig().configure(level="debug")
        assert cfg.level == "DEBUG"

    def test_configure_format(self) -> None:
        cfg = LoggingConfig().configure(fmt=LogFormat.CONSOLE)
        assert cfg.format == LogFormat.CONSOLE

    def test_configure_format_from_string(self) -> None:
        cfg = LoggingConfig().configure(fmt="loki")
        assert cfg.format == LogFormat.LOKI

    def test_configure_output(self) -> None:
        cfg = LoggingConfig().configure(output="stderr")
        assert cfg.output == "stderr"

    def test_configure_chaining(self) -> None:
        cfg = LoggingConfig()
        result = cfg.configure(level="debug", fmt="console", output="stderr")
        assert result is cfg

    def test_set_level_module(self) -> None:
        cfg = LoggingConfig().set_level("goal_generation", "debug")
        assert cfg.module_levels["goal_generation"] == "DEBUG"

    def test_set_level_multiple_modules(self) -> None:
        cfg = LoggingConfig().set_level("goal_generation", "debug").set_level("planning", "warning")
        assert cfg.module_levels["goal_generation"] == "DEBUG"
        assert cfg.module_levels["planning"] == "WARNING"

    def test_set_level_chaining(self) -> None:
        cfg = LoggingConfig()
        result = cfg.set_level("cc", "info")
        assert result is cfg

    def test_add_context(self) -> None:
        cfg = LoggingConfig().add_context("agent_id", "agent-001")
        assert cfg.context["agent_id"] == "agent-001"

    def test_add_context_multiple(self) -> None:
        cfg = (
            LoggingConfig()
            .add_context("agent_id", "agent-001")
            .add_context("worker_id", "w-1")
            .add_context("trace_id", "abc123")
        )
        assert cfg.context == {
            "agent_id": "agent-001",
            "worker_id": "w-1",
            "trace_id": "abc123",
        }

    def test_add_context_chaining(self) -> None:
        cfg = LoggingConfig()
        result = cfg.add_context("k", "v")
        assert result is cfg

    def test_with_log_filter(self) -> None:
        f = LogFilter().by_agent("agent-001")
        cfg = LoggingConfig(log_filter=f)
        assert cfg.log_filter is f


# ---------------------------------------------------------------------------
# Processor functions
# ---------------------------------------------------------------------------


class TestProcessors:
    """Test individual structlog processor functions."""

    def test_add_timestamp(self) -> None:
        event: dict[str, Any] = {"event": "test"}
        result = _add_timestamp(None, "info", event)
        assert "timestamp" in result
        # ISO-8601 format with timezone
        assert result["timestamp"].endswith("+00:00")

    def test_add_log_level(self) -> None:
        event: dict[str, Any] = {"event": "test"}
        result = _add_log_level(None, "warning", event)
        assert result["level"] == "warning"

    def test_context_injector(self) -> None:
        ctx = {"agent_id": "agent-001", "worker_id": "w-1"}
        injector = _make_context_injector(ctx)
        event: dict[str, Any] = {"event": "test"}
        result = injector(None, "info", event)
        assert result["agent_id"] == "agent-001"
        assert result["worker_id"] == "w-1"

    def test_context_injector_does_not_overwrite(self) -> None:
        ctx = {"agent_id": "default"}
        injector = _make_context_injector(ctx)
        event: dict[str, Any] = {"event": "test", "agent_id": "specific"}
        result = injector(None, "info", event)
        assert result["agent_id"] == "specific"

    def test_module_level_filter_passes(self) -> None:
        f = _make_module_level_filter({"cc": "WARNING"})
        event: dict[str, Any] = {"event": "err", "module": "cc"}
        result = f(None, "error", event)
        assert result == event

    def test_module_level_filter_drops(self) -> None:
        f = _make_module_level_filter({"cc": "WARNING"})
        event: dict[str, Any] = {"event": "dbg", "module": "cc"}
        with pytest.raises(structlog.DropEvent):
            f(None, "debug", event)

    def test_module_level_filter_ignores_unconfigured(self) -> None:
        f = _make_module_level_filter({"cc": "WARNING"})
        event: dict[str, Any] = {"event": "dbg", "module": "planning"}
        result = f(None, "debug", event)
        assert result == event

    def test_rename_event_key_stringifies(self) -> None:
        event: dict[str, Any] = {"event": 12345}
        result = _rename_event_key(None, "info", event)
        assert result["event"] == "12345"


# ---------------------------------------------------------------------------
# Processor chain building
# ---------------------------------------------------------------------------


class TestBuildProcessorChain:
    """Test _build_processor_chain output."""

    def test_json_chain_ends_with_json_renderer(self) -> None:
        cfg = LoggingConfig(format=LogFormat.JSON)
        chain = _build_processor_chain(cfg)
        assert isinstance(chain[-1], structlog.processors.JSONRenderer)

    def test_loki_chain_ends_with_json_renderer(self) -> None:
        cfg = LoggingConfig(format=LogFormat.LOKI)
        chain = _build_processor_chain(cfg)
        assert isinstance(chain[-1], structlog.processors.JSONRenderer)

    def test_console_chain_ends_with_console_renderer(self) -> None:
        cfg = LoggingConfig(format=LogFormat.CONSOLE)
        chain = _build_processor_chain(cfg)
        assert isinstance(chain[-1], structlog.dev.ConsoleRenderer)

    def test_chain_includes_context_injector_when_context_set(self) -> None:
        cfg = LoggingConfig().add_context("agent_id", "a-1")
        chain = _build_processor_chain(cfg)
        # Should have more processors than a chain without context
        cfg_no_ctx = LoggingConfig()
        chain_no_ctx = _build_processor_chain(cfg_no_ctx)
        assert len(chain) > len(chain_no_ctx)

    def test_chain_includes_log_filter(self) -> None:
        f = LogFilter().by_agent("a")
        cfg = LoggingConfig(log_filter=f)
        chain = _build_processor_chain(cfg)
        assert f in chain

    def test_chain_includes_module_level_filter(self) -> None:
        cfg = LoggingConfig().set_level("cc", "warning")
        chain = _build_processor_chain(cfg)
        cfg_plain = LoggingConfig()
        chain_plain = _build_processor_chain(cfg_plain)
        assert len(chain) > len(chain_plain)


# ---------------------------------------------------------------------------
# configure_piano_logging
# ---------------------------------------------------------------------------


class TestConfigurePianoLogging:
    """Test the entry-point function."""

    def test_returns_default_config_when_none(self) -> None:
        result = configure_piano_logging(None)
        assert isinstance(result, LoggingConfig)
        assert result.level == "INFO"
        assert result.format == LogFormat.JSON

    def test_returns_same_config(self) -> None:
        cfg = LoggingConfig(level="DEBUG", format=LogFormat.CONSOLE)
        result = configure_piano_logging(cfg)
        assert result is cfg

    def test_sets_root_logging_level(self) -> None:
        configure_piano_logging(LoggingConfig(level="WARNING"))
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_sets_module_level_in_stdlib(self) -> None:
        cfg = LoggingConfig().set_level("piano.core", "DEBUG")
        configure_piano_logging(cfg)
        mod_logger = logging.getLogger("piano.core")
        assert mod_logger.level == logging.DEBUG

    def test_json_output_is_valid_json(self) -> None:
        """After configure, a log event should produce valid JSON."""
        cfg = LoggingConfig(format=LogFormat.JSON)
        cfg.add_context("agent_id", "agent-001")
        configure_piano_logging(cfg)

        buf = StringIO()
        structlog.configure(
            processors=_build_processor_chain(cfg),
            logger_factory=structlog.PrintLoggerFactory(file=buf),
            cache_logger_on_first_use=False,
        )

        log = structlog.get_logger()
        log.info("module_tick_complete", module="goal_generation", duration_ms=45.2)

        output = buf.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["event"] == "module_tick_complete"
        assert parsed["level"] == "info"
        assert parsed["agent_id"] == "agent-001"
        assert parsed["module"] == "goal_generation"
        assert parsed["duration_ms"] == 45.2
        assert "timestamp" in parsed

    def test_loki_compatible_json_format(self) -> None:
        """Loki-compatible JSON includes timestamp, level, event keys."""
        cfg = LoggingConfig(format=LogFormat.LOKI)
        cfg.add_context("worker_id", "w-1")
        cfg.add_context("trace_id", "abc123")

        buf = StringIO()
        structlog.configure(
            processors=_build_processor_chain(cfg),
            logger_factory=structlog.PrintLoggerFactory(file=buf),
            cache_logger_on_first_use=False,
        )

        log = structlog.get_logger()
        log.info("cc_decision", action="mine")

        parsed = json.loads(buf.getvalue().strip())
        assert parsed["timestamp"].endswith("+00:00")
        assert parsed["level"] == "info"
        assert parsed["event"] == "cc_decision"
        assert parsed["worker_id"] == "w-1"
        assert parsed["trace_id"] == "abc123"
        assert parsed["action"] == "mine"

    def test_filter_integration(self) -> None:
        """LogFilter correctly filters events via configure_piano_logging."""
        f = LogFilter().by_agent("agent-001")
        cfg = LoggingConfig(format=LogFormat.JSON, log_filter=f)

        buf = StringIO()
        structlog.configure(
            processors=_build_processor_chain(cfg),
            logger_factory=structlog.PrintLoggerFactory(file=buf),
            cache_logger_on_first_use=False,
        )

        log = structlog.get_logger()
        log.info("tick", agent_id="agent-001")
        log.info("tick", agent_id="agent-002")

        lines = [line for line in buf.getvalue().strip().split("\n") if line]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["agent_id"] == "agent-001"


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestExports:
    """Verify public API exports."""

    def test_all_exports(self) -> None:
        from piano.observability import logging_config

        expected = {"LogFilter", "LogFormat", "LoggingConfig", "configure_piano_logging"}
        assert set(logging_config.__all__) == expected
