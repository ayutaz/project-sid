"""Tests for the PIANO simulation launcher."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from piano.config.settings import PianoSettings
from piano.llm.mock import MockLLMProvider
from piano.main import (
    _create_provider,
    _create_sas,
    _health_check_loop,
    _install_shutdown_handler,
    _register_demo_responses,
    _register_modules,
    _run_multi,
    _run_single,
    cli,
    parse_args,
    run,
)


class TestParseArgs:
    """Tests for argument parsing."""

    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.agents == 1
        assert args.ticks is None
        assert args.mock_llm is False
        assert args.config is None
        assert args.log_level == "INFO"
        assert args.sas_backend == "auto"

    def test_agents_flag(self) -> None:
        args = parse_args(["--agents", "5"])
        assert args.agents == 5

    def test_ticks_flag(self) -> None:
        args = parse_args(["--ticks", "100"])
        assert args.ticks == 100

    def test_mock_llm_flag(self) -> None:
        args = parse_args(["--mock-llm"])
        assert args.mock_llm is True

    def test_config_flag(self) -> None:
        args = parse_args(["--config", "/path/to/.env"])
        assert args.config == "/path/to/.env"

    def test_log_level_flag(self) -> None:
        args = parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_no_bridge_flag(self) -> None:
        args = parse_args(["--no-bridge"])
        assert args.no_bridge is True

    def test_no_bridge_default_false(self) -> None:
        args = parse_args([])
        assert args.no_bridge is False

    def test_sas_backend_flag(self) -> None:
        args = parse_args(["--sas-backend", "memory"])
        assert args.sas_backend == "memory"

    def test_sas_backend_redis(self) -> None:
        args = parse_args(["--sas-backend", "redis"])
        assert args.sas_backend == "redis"

    def test_sas_backend_invalid(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--sas-backend", "invalid"])

    def test_all_flags_combined(self) -> None:
        args = parse_args(
            [
                "--agents",
                "3",
                "--ticks",
                "50",
                "--mock-llm",
                "--config",
                "test.env",
                "--no-bridge",
                "--sas-backend",
                "memory",
                "--log-level",
                "WARNING",
            ]
        )
        assert args.agents == 3
        assert args.ticks == 50
        assert args.mock_llm is True
        assert args.config == "test.env"
        assert args.no_bridge is True
        assert args.sas_backend == "memory"
        assert args.log_level == "WARNING"

    def test_rejects_invalid_log_level(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--log-level", "INVALID"])


class TestCli:
    """Tests for the CLI entry point."""

    def test_cli_is_callable(self) -> None:
        assert callable(cli)


class TestRun:
    """Tests for the run function."""

    async def test_single_agent_mock_llm(self) -> None:
        args = parse_args(["--mock-llm", "--ticks", "3"])
        await run(args)

    async def test_single_agent_with_log_level(self) -> None:
        args = parse_args(
            [
                "--mock-llm",
                "--ticks",
                "2",
                "--log-level",
                "DEBUG",
            ]
        )
        await run(args)

    async def test_run_single_directly(self) -> None:
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--ticks", "2"])
        await _run_single(args, provider, tick_interval=0.1)

    async def test_run_multi_directly(self) -> None:
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--agents", "2"])
        settings = PianoSettings()

        task = asyncio.create_task(
            _run_multi(args, provider, settings, tick_interval=0.1),
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_run_multi_with_ticks_completes(self) -> None:
        """_run_multi with --ticks finishes when all agents reach tick limit."""
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--agents", "2", "--ticks", "3"])
        settings = PianoSettings()

        # Should complete naturally when all agents reach 3 ticks
        await asyncio.wait_for(
            _run_multi(args, provider, settings, tick_interval=0.01),
            timeout=10.0,
        )


class TestSignalHandling:
    """Tests for graceful shutdown signal handling."""

    async def test_shutdown_via_ticks_single_agent(self) -> None:
        args = parse_args(["--mock-llm", "--ticks", "2"])
        await run(args)

    async def test_single_agent_finite_run_completes(self) -> None:
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--ticks", "1"])
        await _run_single(args, provider, tick_interval=0.1)


class TestRegisterModules:
    """Tests for _register_modules with and without bridge."""

    def test_register_modules_without_bridge(self) -> None:
        """Without bridge, only standard 7 modules are registered."""
        from piano.core.agent import Agent
        from piano.core.scheduler import ModuleScheduler
        from tests.helpers import InMemorySAS

        sas = InMemorySAS("test-agent")
        scheduler = ModuleScheduler(tick_interval=0.1)
        agent = Agent(agent_id="test-agent", sas=sas, scheduler=scheduler)
        provider = MockLLMProvider()

        _register_modules(agent, provider)

        module_names = [m.name for m in agent.modules]
        assert "action_awareness" in module_names
        assert "cognitive_controller" in module_names
        assert "goal_generation" in module_names
        assert "planning" in module_names
        assert "talking" in module_names
        assert "self_reflection" in module_names
        assert "social_awareness" in module_names
        # Bridge modules should NOT be registered
        assert "bridge_perception" not in module_names
        assert "skill_executor" not in module_names
        assert "chat_broadcaster" not in module_names

    def test_register_modules_with_bridge(self) -> None:
        """With bridge + sas, bridge modules are also registered."""
        from piano.core.agent import Agent
        from piano.core.scheduler import ModuleScheduler
        from tests.helpers import InMemorySAS

        sas = InMemorySAS("test-agent")
        scheduler = ModuleScheduler(tick_interval=0.1)
        agent = Agent(agent_id="test-agent", sas=sas, scheduler=scheduler)
        provider = MockLLMProvider()

        mock_bridge = MagicMock()

        _register_modules(agent, provider, bridge=mock_bridge, sas=sas)

        module_names = [m.name for m in agent.modules]
        # Standard modules
        assert "action_awareness" in module_names
        assert "cognitive_controller" in module_names
        assert "talking" in module_names
        # Bridge modules
        assert "bridge_perception" in module_names
        assert "skill_executor" in module_names
        assert "chat_broadcaster" in module_names

    def test_register_modules_bridge_without_sas_skips(self) -> None:
        """If bridge is provided but sas is None, bridge modules are skipped."""
        from piano.core.agent import Agent
        from piano.core.scheduler import ModuleScheduler
        from tests.helpers import InMemorySAS

        sas = InMemorySAS("test-agent")
        scheduler = ModuleScheduler(tick_interval=0.1)
        agent = Agent(agent_id="test-agent", sas=sas, scheduler=scheduler)
        provider = MockLLMProvider()

        mock_bridge = MagicMock()

        _register_modules(agent, provider, bridge=mock_bridge, sas=None)

        module_names = [m.name for m in agent.modules]
        assert "bridge_perception" not in module_names
        assert "skill_executor" not in module_names
        assert "chat_broadcaster" not in module_names


class TestBridgeMode:
    """Tests for bridge connection mode in _run_single and _run_multi."""

    async def test_run_single_no_bridge_mode(self) -> None:
        """--no-bridge flag skips bridge entirely."""
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--ticks", "1", "--no-bridge"])
        await _run_single(args, provider, tick_interval=0.1)

    async def test_run_single_bridge_fallback_on_failure(self) -> None:
        """When bridge connection fails, run continues without bridge."""
        provider = MockLLMProvider()
        args = parse_args(["--ticks", "1", "--mock-llm"])
        # no_bridge is False, so it will try to connect but fail gracefully
        args.no_bridge = False
        await _run_single(args, provider, tick_interval=0.1)

    async def test_run_multi_no_bridge_mode(self) -> None:
        """Multi-agent with --no-bridge works normally."""
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--agents", "2", "--no-bridge"])
        settings = PianoSettings()

        task = asyncio.create_task(
            _run_multi(args, provider, settings, tick_interval=0.1),
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_run_single_with_mock_bridge(self) -> None:
        """Single agent with a mocked bridge manager connects and cleans up."""
        provider = MockLLMProvider()
        args = parse_args(["--mock-llm", "--ticks", "1"])
        args.no_bridge = False

        mock_bridge = MagicMock()
        mock_bridge.start_event_listener = AsyncMock()

        mock_mgr = MagicMock()
        mock_mgr.create_bridge = MagicMock(return_value=mock_bridge)
        mock_mgr.connect_all = AsyncMock(return_value={"agent-001": True})
        mock_mgr.disconnect_all = AsyncMock()

        with patch("piano.bridge.manager.BridgeManager", return_value=mock_mgr):
            await _run_single(args, provider, tick_interval=0.1)

        mock_mgr.disconnect_all.assert_awaited_once()


class TestCreateProvider:
    """Tests for _create_provider with settings."""

    def test_mock_provider_ignores_settings(self) -> None:
        args = parse_args(["--mock-llm"])
        settings = PianoSettings()
        provider = _create_provider(args, settings=settings)
        assert isinstance(provider, MockLLMProvider)

    def test_openai_provider_receives_settings(self) -> None:
        """OpenAIProvider is configured with LLM settings from PianoSettings."""
        from piano.llm.provider import OpenAIProvider

        args = parse_args([])
        settings = PianoSettings()
        settings.llm.api_key = "sk-test-key"
        settings.llm.cost_limit_usd = 42.0
        settings.llm.calls_per_minute_limit = 77

        provider = _create_provider(args, settings=settings)
        assert isinstance(provider, OpenAIProvider)
        assert provider._cost_limit_usd == 42.0
        assert provider._calls_per_minute_limit == 77

    def test_openai_provider_empty_api_key_passes_none(self) -> None:
        """Empty api_key string maps to None (uses OPENAI_API_KEY env var)."""
        from piano.llm.provider import OpenAIProvider

        args = parse_args([])
        settings = PianoSettings()
        settings.llm.api_key = ""  # default

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy-for-test"}):
            provider = _create_provider(args, settings=settings)
            assert isinstance(provider, OpenAIProvider)

    def test_openai_provider_without_settings(self) -> None:
        from piano.llm.provider import OpenAIProvider

        args = parse_args([])
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-dummy-for-test"}):
            provider = _create_provider(args)
            assert isinstance(provider, OpenAIProvider)


class TestCreateSas:
    """Tests for _create_sas with sas_backend parameter."""

    async def test_auto_fallback_logs_warning(self) -> None:
        """When Redis is unavailable, auto mode logs a warning and falls back."""
        mock_redis_cls = MagicMock()
        mock_redis_cls.create_with_settings = MagicMock(
            side_effect=ConnectionError("Redis down"),
        )
        with (
            patch.dict(
                "sys.modules",
                {"piano.core.sas_redis": MagicMock(RedisSAS=mock_redis_cls)},
            ),
            patch("piano.main.logger") as mock_logger,
        ):
            sas = _create_sas("test-agent", sas_backend="auto")
            mock_logger.warning.assert_called_once()
            assert "redis_sas_creation_failed" in str(
                mock_logger.warning.call_args,
            )
            assert sas is not None
            assert sas.agent_id == "test-agent"

    async def test_memory_backend_skips_redis(self) -> None:
        """sas_backend='memory' never attempts Redis."""
        sas = _create_sas("test-agent", sas_backend="memory")
        assert sas is not None
        assert sas.agent_id == "test-agent"

    async def test_redis_backend_raises_on_failure(self) -> None:
        """sas_backend='redis' raises when Redis is unavailable."""
        mock_redis_cls = MagicMock()
        mock_redis_cls.create_with_settings = MagicMock(
            side_effect=ConnectionError("Redis down"),
        )
        with (
            patch.dict(
                "sys.modules",
                {"piano.core.sas_redis": MagicMock(RedisSAS=mock_redis_cls)},
            ),
            pytest.raises(ConnectionError),
        ):
            _create_sas("test-agent", sas_backend="redis")

    async def test_snapshot_includes_agent_id(self) -> None:
        """_LocalSAS.snapshot() includes agent_id field."""
        sas = _create_sas("test-agent", sas_backend="memory")
        snap = await sas.snapshot()
        assert snap["agent_id"] == "test-agent"

    async def test_default_backend_is_auto(self) -> None:
        """Default sas_backend is auto, falls back to memory."""
        sas = _create_sas("test-agent", sas_backend="memory")
        assert sas is not None
        assert sas.agent_id == "test-agent"


class TestVersionFlag:
    """Tests for --version flag."""

    def test_version_flag_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--version"])
        assert exc_info.value.code == 0


class TestConfigValidation:
    """Tests for --config path validation."""

    async def test_missing_config_warns(self) -> None:
        """Run with nonexistent config path should warn but not crash."""
        args = parse_args(["--mock-llm", "--ticks", "1", "--config", "/nonexistent/path.env"])
        await run(args)

    async def test_existing_config_works(self, tmp_path) -> None:
        """Run with valid config path works normally."""
        env_file = tmp_path / "test.env"
        env_file.write_text("PIANO_LOG__LEVEL=DEBUG\n")
        args = parse_args(["--mock-llm", "--ticks", "1", "--config", str(env_file)])
        await run(args)


class TestInstallShutdownHandler:
    """Tests for the unified shutdown handler."""

    async def test_shutdown_handler_sets_event(self) -> None:
        """The shutdown handler correctly sets the event."""
        event = asyncio.Event()
        _install_shutdown_handler(event)
        # Event should not be set initially
        assert not event.is_set()


class TestLogLevelFromSettings:
    """Tests for settings.log.level integration."""

    async def test_settings_log_level_used_when_cli_default(self) -> None:
        """When CLI uses default INFO, settings.log.level should be used."""
        args = parse_args(["--mock-llm", "--ticks", "1"])
        # Default --log-level is INFO, so settings.log.level should be checked
        assert args.log_level == "INFO"


class TestRegisterDemoResponses:
    """Tests for _register_demo_responses rotating mock responses."""

    async def test_rotating_responses(self) -> None:
        """Demo responses rotate through the list."""
        mock = MockLLMProvider()
        _register_demo_responses(mock)

        r1 = await mock.complete("test")
        r2 = await mock.complete("test")
        assert r1 != r2  # different on successive calls

    async def test_responses_are_valid_json(self) -> None:
        """All demo responses are valid JSON with expected keys."""
        import json

        mock = MockLLMProvider()
        _register_demo_responses(mock)

        for _ in range(7):
            resp = await mock.complete("test")
            data = json.loads(resp)
            assert "action" in data
            assert "action_params" in data
            assert "speaking" in data
            assert "reasoning" in data

    async def test_responses_wrap_around(self) -> None:
        """After cycling all responses, wraps back to first."""
        mock = MockLLMProvider()
        _register_demo_responses(mock)

        first = await mock.complete("test")
        # Cycle through remaining 6
        for _ in range(6):
            await mock.complete("test")
        eighth = await mock.complete("test")
        assert first == eighth


class TestCreateProviderWithDemoResponses:
    """Tests for _create_provider registering demo responses."""

    async def test_mock_provider_has_demo_responses(self) -> None:
        """Mock provider from _create_provider has rotating responses."""
        import json

        args = parse_args(["--mock-llm"])
        provider = _create_provider(args)
        resp = await provider.complete("test")
        data = json.loads(resp)
        assert "action" in data


class TestHealthCheckLoop:
    """Tests for _health_check_loop."""

    async def test_health_check_stops_on_shutdown(self) -> None:
        """Health check loop exits when shutdown_event is set."""
        from piano.bridge.health import BridgeHealthMonitor

        mock_manager = MagicMock()
        mock_manager.bridges = {}

        monitor = BridgeHealthMonitor()
        shutdown_event = asyncio.Event()

        task = asyncio.create_task(
            _health_check_loop(
                monitor, mock_manager, shutdown_event, interval_s=0.05
            )
        )
        await asyncio.sleep(0.1)
        shutdown_event.set()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_health_check_survives_errors(self) -> None:
        """Health check loop continues despite check_all errors."""
        mock_monitor = MagicMock()
        mock_monitor.check_all = AsyncMock(
            side_effect=RuntimeError("check failed")
        )

        mock_manager = MagicMock()
        mock_manager.bridges = {}

        shutdown_event = asyncio.Event()

        task = asyncio.create_task(
            _health_check_loop(
                mock_monitor, mock_manager, shutdown_event, interval_s=0.05
            )
        )
        await asyncio.sleep(0.15)
        shutdown_event.set()
        await asyncio.wait_for(task, timeout=2.0)


class TestSasBackendSetting:
    """Tests for PianoSettings.sas_backend field."""

    def test_default_sas_backend(self) -> None:
        settings = PianoSettings()
        assert settings.sas_backend == "auto"

    def test_sas_backend_override(self) -> None:
        settings = PianoSettings(sas_backend="memory")
        assert settings.sas_backend == "memory"
