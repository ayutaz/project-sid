"""Tests for the PIANO simulation launcher."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from piano.config.settings import PianoSettings
from piano.llm.mock import MockLLMProvider
from piano.main import (
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
                "--log-level",
                "WARNING",
            ]
        )
        assert args.agents == 3
        assert args.ticks == 50
        assert args.mock_llm is True
        assert args.config == "test.env"
        assert args.no_bridge is True
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
